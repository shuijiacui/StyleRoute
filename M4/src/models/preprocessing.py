"""Deterministic RGB preprocessing shared by generation, segmentation and scoring."""

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

ImageInput = str | Path | bytes | Image.Image


@dataclass(frozen=True)
class PreprocessedImage:
    rgb: Image.Image
    tensor: torch.Tensor
    metadata: dict


def load_rgb(source: ImageInput) -> Image.Image:
    """Decode eagerly, apply EXIF orientation and composite transparency on white."""
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        else:
            stream = io.BytesIO(source) if isinstance(source, bytes) else Path(source)
            with Image.open(stream) as opened:
                opened.load()
                image = opened.copy()
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
        return image.convert("RGB")
    except FileNotFoundError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"Cannot decode image: {error}") from error


def image_fingerprint(image: Image.Image) -> str:
    """Canonical decoded RGB identity, independent of uploaded filename."""
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def image_tensor(image: Image.Image) -> torch.Tensor:
    # This normalized-VGG checkpoint expects RGB [0,1], without ImageNet mean/std.
    pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().unsqueeze(0)


def prepare_content(source: ImageInput, hw=(256, 256)) -> PreprocessedImage:
    image = load_rgb(source)
    height, width = hw
    if any(not isinstance(v, int) or v < 32 or v % 8 for v in hw):
        raise ValueError("Content dimensions must be multiples of 8 and at least 32")
    scale = max(width / image.width, height / image.height)
    resized = (math.ceil(image.width * scale), math.ceil(image.height * scale))
    left, top = (resized[0] - width) // 2, (resized[1] - height) // 2
    box = (left, top, left + width, top + height)
    result = image.resize(resized, Image.Resampling.LANCZOS).crop(box)
    return PreprocessedImage(
        result,
        image_tensor(result),
        {
            "source_rgb_sha256": image_fingerprint(image),
            "original_wh": image.size,
            "resize_mode": "fit_center_crop",
            "resized_wh": resized,
            "crop_ltrb": box,
            "output_hw": hw,
            "exif_transposed": True,
            "transparency_background": "white",
            "processed_rgb_sha256": image_fingerprint(result),
        },
    )


def prepare_style(source: ImageInput, long_side=256, minimum_short_side=32) -> PreprocessedImage:
    image = load_rgb(source)
    scale = long_side / max(image.size)
    size = tuple(max(1, round(v * scale)) for v in image.size)
    if min(size) < minimum_short_side:
        raise ValueError(
            f"Style is too narrow after resize: {size}; minimum is {minimum_short_side}"
        )
    result = image.resize(size, Image.Resampling.LANCZOS)
    return PreprocessedImage(
        result,
        image_tensor(result),
        {
            "source_rgb_sha256": image_fingerprint(image),
            "original_wh": image.size,
            "resize_mode": "preserve_aspect_long_side",
            "output_hw": (size[1], size[0]),
            "processed_rgb_sha256": image_fingerprint(result),
        },
    )


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim != 4 or tensor.shape[:2] != (1, 3):
        raise ValueError("Expected a [1,3,H,W] image tensor")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("Image contains NaN or infinity")
    pixels = tensor.detach().clamp(0, 1).mul(255).round().to(torch.uint8)
    return Image.fromarray(pixels[0].permute(1, 2, 0).cpu().numpy())


def png_bytes(image: Image.Image | torch.Tensor) -> bytes:
    if isinstance(image, torch.Tensor):
        image = tensor_to_pil(image)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()
