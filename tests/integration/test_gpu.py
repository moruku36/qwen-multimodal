"""GPU integration tests (Colab). Run with:  pytest -m gpu tests/integration

They download the real models (~60GB on first run) and need an A100 or L4.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA GPU required", allow_module_level=True)

from qmc.app import build_app  # noqa: E402
from qmc.config import load_config  # noqa: E402
from qmc.controller import TurnOptions  # noqa: E402
from qmc.image_engine import ImageOptions  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    d = tmp_path_factory.mktemp("gpu")
    return build_app(load_config(data_dir=d / "data", local_db_path=d / "h.db"))


def _run(app, sid, text, files=None, steps=12):
    return list(
        app.controller.handle(sid, text, files, TurnOptions(image=ImageOptions(steps=steps, band=768)))
    )


def test_profile_detected(app):
    assert app.profile.key in {"a100_80", "a100_40", "l4"}


def test_mvp_on_gpu(app):
    sid = app.sessions.create_session()
    ev = _run(app, sid, "TerraformとPulumiの違いを一言で")
    assert "".join(e.data for e in ev if e.kind == "text").strip()
    ev = _run(app, sid, "東京の夜景を背景にした未来的なデータセンターを生成して")
    assert any(e.kind == "image" for e in ev), [e for e in ev if e.kind == "error"]
    ev = _run(app, sid, "もう少し夜を暗くして、ネオンを増やして")
    assert app.sessions.latest_image(sid).kind == "edited", [e for e in ev if e.kind == "error"]
    ev = _run(app, sid, "元画像と今の画像の違いを説明して")
    assert "".join(e.data for e in ev if e.kind == "text").strip()
