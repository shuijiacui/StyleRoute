import io

import pytest
from PIL import Image, ImageOps

from src.postcard.render import CANVAS_WH, IMAGE_BOXES, TEMPLATES, render_postcard
from src.project import project_path


@pytest.fixture
def font():
    path = project_path("resources/fonts/NotoSerifCJKsc-Regular.otf")
    if not path.is_file():
        pytest.skip("Actual CJK font is not configured")
    return path


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("size", [(256, 256), (256, 128)])
def test_png_reopens_and_text_does_not_change_picture(font, template, size):
    image = Image.linear_gradient("L").resize(size).convert("RGB")
    first = render_postcard(
        image, "山海来信", "你好，世界！\nA postcard from the mountains.", template, font
    )
    second = render_postcard(image, "另一封信", "把风景寄给你。", template, font)
    with Image.open(io.BytesIO(first)) as one, Image.open(io.BytesIO(second)) as two:
        one.load()
        assert one.format == "PNG" and one.mode == "RGB" and one.size == CANVAS_WH
        box = IMAGE_BOXES[template]
        assert one.crop(box).tobytes() == two.crop(box).tobytes()
        left, top, right, bottom = box
        fitted = ImageOps.contain(image, (right - left, bottom - top), Image.Resampling.LANCZOS)
        x, y = left + (right - left - fitted.width) // 2, top + (bottom - top - fitted.height) // 2
        assert one.crop((x, y, x + fitted.width, y + fitted.height)).tobytes() == fitted.tobytes()
        assert one.tobytes() != two.tobytes()


def test_overflow_and_missing_font_are_explicit(font, tmp_path):
    image = Image.new("RGB", (256, 256))
    with pytest.raises(ValueError, match="five lines"):
        render_postcard(image, "标题", "长正文" * 150, "minimal", font)
    with pytest.raises(ValueError, match="one line"):
        render_postcard(image, "长标题" * 20, "", "minimal", font)
    with pytest.raises(FileNotFoundError):
        render_postcard(image, "", "", "minimal", tmp_path / "missing.ttf")


def test_empty_text_and_unsupported_glyph(font):
    image = Image.new("RGB", (256, 256))
    assert render_postcard(image, "", "", "minimal", font).startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="support character"):
        render_postcard(image, "", "\U0010ffff", "minimal", font)
