import pytest
from PIL import Image

from qmc.imaging import InvalidImageError, flatten_alpha, load_user_image, save_png, to_data_uri


def test_load_and_downscale(make_png):
    p = make_png(size=(4000, 1000))
    img, info = load_user_image(p, max_side=2048)
    assert img.size == (2048, 512)
    assert info["downscaled"]


def test_rejects_non_image(tmp_path):
    p = tmp_path / "x.png"
    p.write_text("not an image")
    with pytest.raises(InvalidImageError):
        load_user_image(p)


def test_rejects_too_large_file(make_png):
    p = make_png()
    with pytest.raises(InvalidImageError):
        load_user_image(p, max_mb=0)


def test_missing_file(tmp_path):
    with pytest.raises(InvalidImageError):
        load_user_image(tmp_path / "nope.png")


def test_data_uri_and_save(make_png, tmp_path):
    p = make_png()
    assert to_data_uri(p).startswith("data:image/jpeg;base64,")
    digest = save_png(Image.new("RGB", (8, 8)), tmp_path / "out" / "a.png")
    assert len(digest) == 64 and (tmp_path / "out" / "a.png").exists()


def test_flatten_alpha():
    assert flatten_alpha(Image.new("RGBA", (4, 4), (1, 2, 3, 255))).mode == "RGB"
    assert flatten_alpha(Image.new("RGBA", (4, 4), (1, 2, 3, 0))).mode == "RGBA"
