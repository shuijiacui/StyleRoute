import io

import numpy as np
import pytest
from PIL import Image

from src.models.preprocessing import image_fingerprint, load_rgb, prepare_content, prepare_style


def test_transparency_and_grayscale():
    assert load_rgb(Image.new("RGBA", (2, 2), (255, 0, 0, 128))).getpixel((0, 0)) == (255, 127, 127)
    gray = load_rgb(Image.new("L", (2, 2), 80))
    assert gray.mode == "RGB" and gray.getpixel((0, 0)) == (80, 80, 80)


def test_exif_orientation():
    source = Image.new("RGB", (20, 30), "red")
    exif = source.getexif()
    exif[274] = 6
    stream = io.BytesIO()
    source.save(stream, format="JPEG", exif=exif)
    assert load_rgb(stream.getvalue()).size == (30, 20)


def test_content_crop_has_recorded_geometry():
    image = Image.new("RGB", (100, 50), "blue")
    image.paste("red", (25, 0, 75, 50))
    prepared = prepare_content(image, (64, 64))
    assert prepared.metadata["resized_wh"] == (128, 64)
    assert prepared.metadata["crop_ltrb"] == (32, 0, 96, 64)
    assert prepared.rgb.getpixel((32, 32)) == (255, 0, 0)
    assert prepared.tensor.shape == (1, 3, 64, 64)
    assert np.allclose(prepared.tensor[0, :, 32, 32].numpy(), [1, 0, 0])


def test_style_ratio_and_extreme_narrow_rejection():
    assert prepare_style(Image.new("RGB", (400, 200))).rgb.size == (256, 128)
    with pytest.raises(ValueError, match="too narrow"):
        prepare_style(Image.new("RGB", (4000, 10)))


def test_bad_image_and_missing_file(tmp_path):
    with pytest.raises(ValueError, match="decode"):
        load_rgb(b"not really a png")
    with pytest.raises(FileNotFoundError):
        load_rgb(tmp_path / "missing.jpg")


def test_same_name_different_bytes_change_identity(tmp_path):
    target = tmp_path / "same.png"
    Image.new("RGB", (4, 4), "red").save(target)
    first = image_fingerprint(load_rgb(target))
    Image.new("RGB", (4, 4), "blue").save(target)
    assert image_fingerprint(load_rgb(target)) != first
