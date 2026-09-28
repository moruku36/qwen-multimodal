"""Conversation state: sessions, messages, images and their revision lineage.

Image lineage is a tree stored with ``parent_id`` / ``root_id`` / ``revision``:

    original (rev 0) -> revision 1 -> revision 2
                     \\-> revision 1' (a different edit of the original)

so "go back", "use the image two steps ago" or "compare with the original" are simple queries.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .history_manager import HistoryStore
from .imaging import save_png

TITLE_MAX = 40


@dataclass
class ImageRecord:
    id: str
    session_id: str
    message_id: int | None
    kind: str  # uploaded | generated | edited
    rel_path: str
    width: int | None
    height: int | None
    parent_id: str | None
    root_id: str
    revision: int
    seed: int | None
    created_at: float
    meta: dict = field(default_factory=dict)

    @property
    def caption(self) -> str:
        if self.meta.get("pdf_page"):
            return f"PDF {self.meta.get('pdf_name', '')} p{self.meta['pdf_page']}"
        base = {"uploaded": "アップロード画像", "generated": "生成画像", "edited": "編集画像"}[self.kind]
        rev = f" rev{self.revision}" if self.revision else ""
        prompt = self.meta.get("prompt")
        return f"{base}{rev}" + (f" / prompt: {prompt[:80]}" if prompt else "")


@dataclass
class MessageRecord:
    id: int
    session_id: str
    role: str
    content: str
    reasoning: str | None
    intent: str | None
    model: str | None
    created_at: float
    meta: dict = field(default_factory=dict)
    images: list[ImageRecord] = field(default_factory=list)


def _now() -> float:
    return time.time()


def _row_to_image(row) -> ImageRecord:
    return ImageRecord(
        id=row["id"],
        session_id=row["session_id"],
        message_id=row["message_id"],
        kind=row["kind"],
        rel_path=row["rel_path"],
        width=row["width"],
        height=row["height"],
        parent_id=row["parent_id"],
        root_id=row["root_id"],
        revision=row["revision"],
        seed=row["seed"],
        created_at=row["created_at"],
        meta=json.loads(row["meta"] or "{}"),
    )


class SessionManager:
    def __init__(self, store: HistoryStore, data_dir: Path):
        self.store = store
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------ sessions
    def create_session(self, title: str = "New chat") -> str:
        sid = uuid.uuid4().hex[:12]
        now = _now()
        with self.store.tx() as c:
            c.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (sid, title, now, now),
            )
        return sid

    def session_exists(self, session_id: str) -> bool:
        return self.store.query_one("SELECT 1 FROM sessions WHERE id = ?", (session_id,)) is not None

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.store.query(
            """SELECT s.id, s.title, s.updated_at,
                      (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id AND m.deleted = 0) AS n
               FROM sessions s ORDER BY s.updated_at DESC LIMIT ?""",
            (limit,),
        )
        return [dict(r) for r in rows]

    def rename_session(self, session_id: str, title: str) -> None:
        with self.store.tx() as c:
            c.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title[:120], _now(), session_id),
            )

    def delete_session(self, session_id: str) -> None:
        with self.store.tx() as c:
            c.execute("DELETE FROM generations WHERE session_id = ?", (session_id,))
            c.execute("DELETE FROM images WHERE session_id = ?", (session_id,))
            c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        shutil.rmtree(self.data_dir / "sessions" / session_id, ignore_errors=True)

    def _touch(self, c, session_id: str) -> None:
        c.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))

    # ------------------------------------------------------------------ messages
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        intent: str | None = None,
        model: str | None = None,
        reasoning: str | None = None,
        meta: dict | None = None,
    ) -> int:
        with self.store.tx() as c:
            cur = c.execute(
                """INSERT INTO messages(session_id, role, content, reasoning, intent, model, created_at, meta)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, role, content, reasoning, intent, model, _now(), json.dumps(meta or {})),
            )
            if role == "user" and content.strip():
                row = c.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
                if row and row["title"] == "New chat":
                    title = " ".join(content.split())
                    title = title[:TITLE_MAX] + ("…" if len(title) > TITLE_MAX else "")
                    c.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
            self._touch(c, session_id)
            return int(cur.lastrowid)

    def update_message(self, message_id: int, **fields: Any) -> None:
        allowed = {"content", "reasoning", "intent", "model", "meta"}
        sets, vals = [], []
        for k, v in fields.items():
            if k not in allowed:
                raise KeyError(k)
            sets.append(f"{k} = ?")
            vals.append(json.dumps(v) if k == "meta" else v)
        if not sets:
            return
        with self.store.tx() as c:
            c.execute(f"UPDATE messages SET {', '.join(sets)} WHERE id = ?", (*vals, message_id))

    def get_messages(self, session_id: str) -> list[MessageRecord]:
        rows = self.store.query(
            "SELECT * FROM messages WHERE session_id = ? AND deleted = 0 ORDER BY id", (session_id,)
        )
        images_by_msg: dict[int, list[ImageRecord]] = {}
        for r in self.store.query(
            "SELECT * FROM images WHERE session_id = ? ORDER BY created_at", (session_id,)
        ):
            img = _row_to_image(r)
            if img.message_id is not None:
                images_by_msg.setdefault(img.message_id, []).append(img)
        return [
            MessageRecord(
                id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                reasoning=r["reasoning"],
                intent=r["intent"],
                model=r["model"],
                created_at=r["created_at"],
                meta=json.loads(r["meta"] or "{}"),
                images=images_by_msg.get(r["id"], []),
            )
            for r in rows
        ]

    def delete_messages_from(self, session_id: str, message_id: int) -> None:
        """Soft-delete ``message_id`` and everything after it (Regenerate)."""
        with self.store.tx() as c:
            c.execute(
                "UPDATE messages SET deleted = 1 WHERE session_id = ? AND id >= ?", (session_id, message_id)
            )
            self._touch(c, session_id)

    def clear_session(self, session_id: str) -> None:
        with self.store.tx() as c:
            c.execute("UPDATE messages SET deleted = 1 WHERE session_id = ?", (session_id,))
            self._touch(c, session_id)

    def last_user_message(self, session_id: str) -> MessageRecord | None:
        for m in reversed(self.get_messages(session_id)):
            if m.role == "user":
                return m
        return None

    # ------------------------------------------------------------------ images
    def add_image(
        self,
        session_id: str,
        image: Image.Image,
        kind: str,
        *,
        message_id: int | None = None,
        parent_id: str | None = None,
        seed: int | None = None,
        meta: dict | None = None,
    ) -> ImageRecord:
        image_id = uuid.uuid4().hex[:8]
        rel = Path("sessions") / session_id / "images" / f"{image_id}.png"
        digest = save_png(image, self.data_dir / rel)
        parent = self.get_image(parent_id) if parent_id else None
        root_id = parent.root_id if parent else image_id
        revision = parent.revision + 1 if parent else 0
        now = _now()
        with self.store.tx() as c:
            c.execute(
                """INSERT INTO images(id, session_id, message_id, kind, rel_path, width, height, parent_id, root_id,
                                      revision, seed, sha256, created_at, meta)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    image_id, session_id, message_id, kind, rel.as_posix(), image.width, image.height,
                    parent.id if parent else None, root_id, revision, seed, digest, now, json.dumps(meta or {}),
                ),
            )  # fmt: skip
            self._touch(c, session_id)
        return self.get_image(image_id)

    def attach_image_to_message(self, image_id: str, message_id: int) -> None:
        with self.store.tx() as c:
            c.execute("UPDATE images SET message_id = ? WHERE id = ?", (message_id, image_id))

    def update_image_meta(self, image_id: str, meta: dict) -> None:
        with self.store.tx() as c:
            c.execute("UPDATE images SET meta = ? WHERE id = ?", (json.dumps(meta), image_id))

    def get_image(self, image_id: str) -> ImageRecord:
        row = self.store.query_one("SELECT * FROM images WHERE id = ?", (image_id,))
        if row is None:
            raise KeyError(f"image {image_id} not found")
        return _row_to_image(row)

    def image_path(self, image: ImageRecord) -> Path:
        return self.data_dir / image.rel_path

    def load_image(self, image: ImageRecord) -> Image.Image:
        path = self.image_path(image)
        if not path.exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません（Drive未マウント？）: {path}")
        with Image.open(path) as im:
            return im.copy()

    def session_images(self, session_id: str) -> list[ImageRecord]:
        """Images of visible (non-deleted) messages, oldest first."""
        rows = self.store.query(
            """SELECT i.* FROM images i JOIN messages m ON m.id = i.message_id
               WHERE i.session_id = ? AND m.deleted = 0 ORDER BY i.created_at, i.rowid""",
            (session_id,),
        )
        return [_row_to_image(r) for r in rows]

    def latest_image(self, session_id: str) -> ImageRecord | None:
        imgs = self.session_images(session_id)
        return imgs[-1] if imgs else None

    def nth_previous_image(self, session_id: str, n: int) -> ImageRecord | None:
        """n=0 -> latest, n=1 -> the one before, ... (chronological, across lineages)."""
        imgs = self.session_images(session_id)
        return imgs[-1 - n] if 0 <= n < len(imgs) else None

    def lineage(self, image_id: str) -> list[ImageRecord]:
        """Root -> ... -> image_id."""
        chain: list[ImageRecord] = []
        current: str | None = image_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            img = self.get_image(current)
            chain.append(img)
            current = img.parent_id
        return list(reversed(chain))

    def children(self, image_id: str) -> list[ImageRecord]:
        return [
            _row_to_image(r)
            for r in self.store.query("SELECT * FROM images WHERE parent_id = ?", (image_id,))
        ]

    # ------------------------------------------------------------------ generations
    def add_generation(
        self,
        session_id: str,
        *,
        kind: str,
        prompt: str,
        effective_prompt: str | None,
        params: dict,
        model: str,
        duration_s: float,
        message_id: int | None = None,
        image_id: str | None = None,
        source_image_ids: list[str] | None = None,
    ) -> int:
        with self.store.tx() as c:
            cur = c.execute(
                """INSERT INTO generations(session_id, message_id, image_id, kind, prompt, effective_prompt,
                                           source_image_ids, params, model, duration_s, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, message_id, image_id, kind, prompt, effective_prompt,
                    json.dumps(source_image_ids or []), json.dumps(params), model, duration_s, _now(),
                ),
            )  # fmt: skip
            return int(cur.lastrowid)

    def generations(self, session_id: str) -> list[dict]:
        rows = self.store.query("SELECT * FROM generations WHERE session_id = ? ORDER BY id", (session_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["params"] = json.loads(d["params"])
            d["source_image_ids"] = json.loads(d["source_image_ids"])
            out.append(d)
        return out

    # ------------------------------------------------------------------ settings
    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.store.query_one("SELECT value FROM model_settings WHERE key = ?", (key,))
        return json.loads(row["value"]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with self.store.tx() as c:
            c.execute(
                """INSERT INTO model_settings(key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
                (key, json.dumps(value), _now()),
            )
