import sqlite3

from PIL import Image

from qmc.history_manager import HistoryStore
from qmc.session_manager import SessionManager


def _sm(tmp_path, mirror=True):
    store = HistoryStore(tmp_path / "local" / "h.db", tmp_path / "drive" / "history.db" if mirror else None)
    return SessionManager(store, tmp_path / "drive"), store


def test_session_crud_and_auto_title(tmp_path):
    sm, _ = _sm(tmp_path)
    sid = sm.create_session()
    sm.add_message(sid, "user", "TerraformとPulumiの違いを教えて")
    sm.add_message(sid, "assistant", "違いは…", model="qwen", intent="chat")
    sessions = sm.list_sessions()
    assert sessions[0]["title"].startswith("TerraformとPulumi")
    assert sessions[0]["n"] == 2
    msgs = sm.get_messages(sid)
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].model == "qwen"
    sm.rename_session(sid, "IaC")
    assert sm.list_sessions()[0]["title"] == "IaC"
    sm.delete_session(sid)
    assert sm.list_sessions() == []


def test_image_revision_tracking(tmp_path):
    sm, _ = _sm(tmp_path)
    sid = sm.create_session()
    m1 = sm.add_message(sid, "assistant", "生成しました")
    orig = sm.add_image(
        sid, Image.new("RGB", (32, 32)), "generated", message_id=m1, seed=1, meta={"prompt": "city"}
    )
    m2 = sm.add_message(sid, "assistant", "編集しました")
    rev1 = sm.add_image(sid, Image.new("RGB", (32, 32)), "edited", message_id=m2, parent_id=orig.id, seed=2)
    m3 = sm.add_message(sid, "assistant", "編集しました")
    rev2 = sm.add_image(sid, Image.new("RGB", (32, 32)), "edited", message_id=m3, parent_id=rev1.id, seed=3)

    assert (orig.revision, rev1.revision, rev2.revision) == (0, 1, 2)
    assert rev2.root_id == orig.id
    assert [i.id for i in sm.lineage(rev2.id)] == [orig.id, rev1.id, rev2.id]
    assert sm.latest_image(sid).id == rev2.id
    assert sm.nth_previous_image(sid, 2).id == orig.id
    assert sm.nth_previous_image(sid, 5) is None
    assert [c.id for c in sm.children(orig.id)] == [rev1.id]
    assert sm.image_path(rev2).exists()
    assert "prompt: city" in orig.caption


def test_regenerate_soft_delete_hides_images(tmp_path):
    sm, _ = _sm(tmp_path)
    sid = sm.create_session()
    u = sm.add_message(sid, "user", "draw")
    a = sm.add_message(sid, "assistant", "done")
    sm.add_image(sid, Image.new("RGB", (8, 8)), "generated", message_id=a)
    sm.delete_messages_from(sid, a)
    assert [m.id for m in sm.get_messages(sid)] == [u]
    assert sm.latest_image(sid) is None
    assert sm.last_user_message(sid).id == u


def test_generations_and_settings(tmp_path):
    sm, _ = _sm(tmp_path)
    sid = sm.create_session()
    sm.add_generation(
        sid,
        kind="generate",
        prompt="p",
        effective_prompt="ep",
        params={"steps": 4},
        model="m",
        duration_s=1.5,
        source_image_ids=["x"],
    )
    g = sm.generations(sid)[0]
    assert g["params"] == {"steps": 4} and g["source_image_ids"] == ["x"]
    sm.set_setting("mode", {"a": 1})
    sm.set_setting("mode", {"a": 2})
    assert sm.get_setting("mode") == {"a": 2}
    assert sm.get_setting("missing", 7) == 7


def test_restore_after_restart_from_drive_mirror(tmp_path):
    sm, store = _sm(tmp_path)
    sid = sm.create_session()
    sm.add_message(sid, "user", "hello")
    assert store.sync()
    store.close()
    # Colab restart: local disk wiped, Drive keeps the mirror
    (tmp_path / "local" / "h.db").unlink()
    sm2, store2 = _sm(tmp_path)
    assert [m.content for m in sm2.get_messages(sid)] == ["hello"]
    assert any("復元" in n for n in store2.notes)


def test_corrupted_local_db_is_recovered_from_mirror(tmp_path):
    sm, store = _sm(tmp_path)
    sid = sm.create_session()
    sm.add_message(sid, "user", "keep me")
    store.sync()
    store.close()
    local = tmp_path / "local" / "h.db"
    local.write_bytes(b"garbage" * 100)
    # make the mirror look older so the restore-by-mtime path is not taken
    import os

    os.utime(tmp_path / "drive" / "history.db", (0, 0))
    sm2, store2 = _sm(tmp_path)
    assert [m.content for m in sm2.get_messages(sid)] == ["keep me"]
    assert any("破損" in n for n in store2.notes)
    assert list((tmp_path / "local").glob("*.corrupt-*"))


def test_corrupted_without_backup_starts_fresh(tmp_path):
    local = tmp_path / "local" / "h.db"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"x" * 4096)
    store = HistoryStore(local, None)
    assert SessionManager(store, tmp_path).list_sessions() == []
    assert any("新しい" in n for n in store.notes)


def test_sync_without_mirror_is_noop(tmp_path):
    _, store = _sm(tmp_path, mirror=False)
    assert store.sync() is False


def test_mirror_is_valid_sqlite(tmp_path):
    sm, store = _sm(tmp_path)
    sm.create_session()
    store.sync()
    conn = sqlite3.connect(tmp_path / "drive" / "history.db")
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
