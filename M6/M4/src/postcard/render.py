"""Landscape stationery with separate, uncropped picture and typeset letter areas."""

import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.models.preprocessing import png_bytes, tensor_to_pil

TEMPLATES = ("minimal", "border", "journal")
TEMPLATE_LABELS = {"minimal": "山海画报", "border": "远方邮笺", "journal": "旅途手记"}
CANVAS_WH = (1500, 1000)
DESIGN_VERSION = "stationery-v2"
IMAGE_BOXES = {
    "minimal": (64, 108, 856, 900),
    "border": (68, 152, 788, 872),
    "journal": (658, 150, 1438, 930),
}
INK, MUTED, VERMILION = "#29463e", "#697268", "#a8513d"


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Break at measured glyph widths; preserve explicit and empty lines."""
    lines = []
    for paragraph in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        current = ""
        for character in paragraph:
            if font.getlength(character) > max_width:
                raise ValueError("A character is wider than the text area")
            if current and font.getlength(current + character) > max_width:
                lines.append(current)
                current = character
            else:
                current += character
        lines.append(current)
    return lines


def _validate_glyphs(text, font):
    missing = bytes(font.getmask("\U0010ffff"))
    for character in set(text):
        if not character.isspace() and bytes(font.getmask(character)) == missing:
            raise ValueError(
                f"Font does not support character {character!r}; choose a suitable font"
            )


def _waves(draw, x, y, width=160, color=MUTED):
    for row in range(3):
        points = [(x + n, y + row * 13 + int(4 * math.sin(n / 13))) for n in range(width)]
        draw.line(points, fill=color, width=2)


def _postmark(draw, center, font, paper):
    x, y = center
    draw.ellipse((x - 64, y - 64, x + 64, y + 64), fill=paper, outline=VERMILION, width=2)
    draw.ellipse((x - 55, y - 55, x + 55, y + 55), outline=VERMILION, width=1)
    draw.text((x, y - 27), "山 海", font=font, fill=VERMILION, anchor="mt")
    draw.text((x, y + 9), "来 信", font=font, fill=VERMILION, anchor="mt")


def _stamp(draw, x, y, font, paper):
    draw.rectangle((x, y, x + 112, y + 142), fill="#dfe5d9")
    for offset in range(8, 113, 16):
        for yy in (y, y + 142):
            draw.ellipse((x + offset - 4, yy - 4, x + offset + 4, yy + 4), fill=paper)
    for offset in range(8, 143, 16):
        for xx in (x, x + 112):
            draw.ellipse((xx - 4, y + offset - 4, xx + 4, y + offset + 4), fill=paper)
    draw.rectangle((x + 12, y + 12, x + 100, y + 130), outline=INK, width=1)
    draw.ellipse((x + 65, y + 30, x + 81, y + 46), fill=VERMILION)
    draw.line(
        [(x + 22, y + 91), (x + 47, y + 49), (x + 65, y + 77), (x + 79, y + 62), (x + 93, y + 91)],
        fill=INK,
        width=2,
    )
    draw.text((x + 56, y + 103), "山 海", font=font, fill=INK, anchor="mt")


def render_postcard(
    image_rgb: Image.Image | torch.Tensor,
    title: str,
    message: str,
    template: str,
    font_path: str | Path,
    *,
    title_font_path: str | Path | None = None,
) -> bytes:
    """Return a 3:2 PNG without cropping, tinting or overlaying the generated image.

    Pixel-wrap the body into at most five lines. Long titles shrink to a legible
    minimum, then raise explicitly. Empty text is valid. Export resolution describes
    the stationery canvas, not additional generated image detail.
    """
    if template not in TEMPLATES:
        raise ValueError(f"Template must be one of {TEMPLATES}")
    font_path = Path(font_path)
    heading_path = Path(title_font_path) if title_font_path else font_path
    for path in (font_path, heading_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Chinese font: {path}")
    if len(title) > 100 or len(message) > 1500:
        raise ValueError("Title/body exceed the input safety limits (100/1500 characters)")
    body_font = ImageFont.truetype(str(font_path), 28)
    small = ImageFont.truetype(str(font_path), 20)
    tiny = ImageFont.truetype(str(font_path), 17)
    width = 466 if template == "minimal" else 520
    title_font = ImageFont.truetype(str(heading_path), 64)
    try:
        _validate_glyphs(title, title_font)
    except ValueError:
        # Rare characters outside the brush face use the complete serif font.
        heading_path = font_path
    for size in range(64, 35, -2):
        title_font = ImageFont.truetype(str(heading_path), size)
        if title_font.getlength(title) <= width:
            break
    _validate_glyphs(title, title_font)
    _validate_glyphs(message, body_font)
    if len(wrap_text(title, title_font, width)) > 1:
        raise ValueError("Title must fit one line; 标题过长，请缩短到一行。")
    body_lines = wrap_text(message, body_font, width)
    if len(body_lines) > 5:
        raise ValueError("Body exceeds five lines; 正文超过五行，请缩短寄语或减少空行。")
    image = (
        tensor_to_pil(image_rgb)
        if isinstance(image_rgb, torch.Tensor)
        else image_rgb.convert("RGB")
    )
    paper = {"minimal": "#faf7ee", "border": "#f6efdf", "journal": "#eae4d5"}[template]
    canvas = Image.new("RGB", CANVAS_WH, paper)
    draw = ImageDraw.Draw(canvas)
    if template == "minimal":
        draw.text(
            (64, 48), "山 海 来 信   /   LANDSCAPE LETTERS", font=small, fill=INK, anchor="lt"
        )
        draw.text((1436, 48), "一帧风景，一纸思念", font=small, fill=MUTED, anchor="rt")
        draw.line((920, 108, 920, 900), fill="#d8d5c8", width=1)
        draw.text((970, 165), "A LITTLE NOTE FROM AFAR", font=tiny, fill=VERMILION, anchor="lt")
        tx, ty, by = 970, 254, 412
        _waves(draw, 970, 352, 104)
        _postmark(draw, (1320, 790), small, paper)
        draw.text((970, 708), "此刻，见字如面。", font=small, fill=MUTED, anchor="lt")
        draw.text((64, 936), "把眼前的山海，寄给心里的远方。", font=small, fill=MUTED, anchor="lt")
        draw.text((1436, 936), "01  /  山海画报", font=tiny, fill=MUTED, anchor="rt")
    elif template == "border":
        for offset in range(20, 1480, 54):
            color = VERMILION if (offset // 54) % 2 else "#536d74"
            for yy in (20, 970):
                draw.polygon(
                    [
                        (offset, yy),
                        (offset + 20, yy),
                        (offset + 9, yy + 10),
                        (offset - 11, yy + 10),
                    ],
                    fill=color,
                )
        draw.text(
            (68, 68),
            "POST CARD",
            font=ImageFont.truetype(str(font_path), 38),
            fill=INK,
            anchor="lt",
        )
        draw.text((350, 82), "远 方 邮 笺", font=small, fill=MUTED, anchor="lt")
        for y in range(152, 872, 13):
            draw.line((836, y, 836, y + 4), fill="#c6beab", width=2)
        _stamp(draw, 1310, 78, tiny, paper)
        _waves(draw, 1115, 148, 162, VERMILION)
        draw.text((900, 258), "TO / 远方的你", font=small, fill=VERMILION, anchor="lt")
        tx, ty, by = 900, 320, 458
        for y in range(498, 740, 56):
            draw.line((900, y, 1420, y), fill="#d7cfbb", width=1)
        draw.text((900, 806), "FROM / 山海之间", font=small, fill=MUTED, anchor="lt")
        draw.text(
            (68, 914), "山海来信  ·  愿每一份想念，都有回响。", font=small, fill=INK, anchor="lt"
        )
        draw.text((1420, 914), "02  /  远方邮笺", font=tiny, fill=MUTED, anchor="rt")
    else:
        draw.rectangle((0, 0, 20, 1000), fill=VERMILION)
        draw.text(
            (74, 72),
            "NOTES FROM",
            font=ImageFont.truetype(str(font_path), 30),
            fill=INK,
            anchor="lt",
        )
        draw.text(
            (74, 115),
            "THE JOURNEY",
            font=ImageFont.truetype(str(font_path), 30),
            fill=INK,
            anchor="lt",
        )
        draw.line((74, 216, 220, 216), fill=VERMILION, width=3)
        tx, ty, by = 74, 290, 438
        draw.rectangle((638, 130, 1458, 950), fill="#fffcf4")
        draw.polygon([(927, 110), (1142, 119), (1136, 147), (922, 140)], fill="#cdbd99")
        _waves(draw, 74, 786, 155)
        draw.text((74, 860), "沿途有光，落笔成信。", font=small, fill=MUTED, anchor="lt")
        draw.text((74, 930), "03  /  旅途手记", font=tiny, fill=VERMILION, anchor="lt")
    draw.text((tx, ty), title, font=title_font, fill=INK, anchor="lt")
    for index, line in enumerate(body_lines):
        draw.text((tx, by + index * 56), line, font=body_font, fill=INK, anchor="lt")
    left, top, right, bottom = IMAGE_BOXES[template]
    fitted = ImageOps.contain(image, (right - left, bottom - top), Image.Resampling.LANCZOS)
    canvas.paste(
        fitted,
        (left + (right - left - fitted.width) // 2, top + (bottom - top - fitted.height) // 2),
    )
    return png_bytes(canvas)
