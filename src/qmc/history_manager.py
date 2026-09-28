"""SQLite persistence with Google Drive mirroring and corruption recovery.

Why a local working copy (ADR-0002): SQLite relies on POSIX file locking and fsync semantics
that the Colab Google Drive FUSE mount does not guarantee, which can corrupt a live database.
So the live DB sits on local disk and a consistent snapshot (``sqlite3`` backup API -> temp
file -> atomic rename) is written to Drive after every turn. On startup the Drive snapshot is
restored when the local copy is missing or older (i.e. after a Colab restart).
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    meta        TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content     TEXT NOT NULL DEFAULT '',
    reasoning   TEXT,
    intent      TEXT,
    model       TEXT,
    created_at  REAL NOT NULL,
    deleted     INTEGER NOT NULL DEFAULT 0,
    meta        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS images (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id  INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('uploaded', 'generated', 'edited')),
    rel_path    TEXT NOT NULL,
    width       INTEGER,
    height      INTEGER,
    parent_id   TEXT REFERENCES images(id) ON DELETE SET NULL,
    root_id     TEXT NOT NULL,
    revision    INTEGER NOT NULL DEFAULT 0,
    seed        INTEGER,
    sha256      TEXT,
    created_at  REAL NOT NULL,
    meta        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_images_session ON images(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_images_parent ON images(parent_id);

CREATE TABLE IF NOT EXISTS generations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id        INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    image_id          TEXT REFERENCES images(id) ON DELETE SET NULL,
    kind              TEXT NOT NULL CHECK (kind IN ('generate', 'edit')),
    prompt            TEXT NOT NULL,
    effective_prompt  TEXT,
    source_image_ids  TEXT NOT NULL DEFAULT '[]',
    params            TEXT NOT NULL DEFAULT '{}',
    model             TEXT,
    duration_s        REAL,
    created_at        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS model_settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
"""


class HistoryStore:
    def __init__(self, local_path: Path, mirror_path: Path | None = None):
        self.local_path = Path(local_path)
        self.mirror_path = Path(mirror_path) if mirror_path else None
        self.notes: list[str] = []  # recovery notes shown in the UI
        self._lock = threading.RLock()
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self._restore_from_mirror_if_needed()
        self.conn = self._open_checked()

    # ------------------------------------------------------------------ open / recover
    def _restore_from_mirror_if_needed(self) -> None:
        m = self.mirror_path
        if not m or not m.exists():
            return
        if not self.local_path.exists() or m.stat().st_mtime > self.local_path.stat().st_mtime + 1:
            if _integrity_ok(m):
                shutil.copy2(m, self.local_path)
                self.notes.append(f"履歴を復元しました: {m}")
            else:
                self.notes.append(f"Drive上の履歴DBが破損していたため復元をスキップしました: {m}")

    def _open_checked(self) -> sqlite3.Connection:
        if self.local_path.exists() and not _integrity_ok(self.local_path):
            broken = self.local_path.with_suffix(f".corrupt-{int(time.time())}.db")
            self.local_path.replace(broken)
            self.notes.append(f"履歴DBが破損していたため退避しました: {broken.name}")
            if self.mirror_path and self.mirror_path.exists() and _integrity_ok(self.mirror_path):
                shutil.copy2(self.mirror_path, self.local_path)
                self.notes.append("Drive上のバックアップから復元しました")
            else:
                self.notes.append("新しい履歴DBを作成しました")
        conn = sqlite3.connect(self.local_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        # rollback journal (not WAL): the DB is a single file, so restore-by-copy is always safe
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.executescript(SCHEMA)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
        return conn

    # ------------------------------------------------------------------ access
    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.conn.execute("BEGIN")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    # ------------------------------------------------------------------ mirror
    def sync(self) -> bool:
        """Write a consistent snapshot to the mirror path (Drive). Never raises."""
        if not self.mirror_path:
            return False
        try:
            self.mirror_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.mirror_path.with_suffix(".db.tmp")
            with self._lock:
                dst = sqlite3.connect(tmp)
                try:
                    self.conn.backup(dst)
                finally:
                    dst.close()
            os.replace(tmp, self.mirror_path)
            return True
        except Exception as exc:  # Drive disconnected / quota
            log.warning("History mirror to %s failed: %s", self.mirror_path, exc)
            return False

    def close(self) -> None:
        with self._lock:
            self.conn.close()


def _integrity_ok(path: Path) -> bool:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
            return bool(row) and row[0] == "ok"
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False
