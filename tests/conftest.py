import pytest
from PIL import Image


@pytest.fixture
def make_png(tmp_path):
    def _make(name="img.png", size=(64, 48), color=(200, 30, 30), mode="RGB"):
        path = tmp_path / name
        Image.new(mode, size, color if mode == "RGB" else color + (255,)).save(path)
        return path

    return _make
