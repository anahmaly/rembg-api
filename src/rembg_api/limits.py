from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError


class ImageLimitError(ValueError):
    """A caller-safe image size or decode limit violation."""


@dataclass(frozen=True)
class ImageLimits:
    max_width: int
    max_height: int
    max_pixels: int
    max_encoded_bytes: int | None = None

    def validate_dimensions(self, width: int, height: int, *, subject: str) -> None:
        if width < 1 or height < 1:
            raise ImageLimitError(f"{subject} image has invalid dimensions")
        if (
            width > self.max_width
            or height > self.max_height
            or width * height > self.max_pixels
        ):
            raise ImageLimitError(f"{subject} image dimensions exceed configured limits")

    def validate_encoded_bytes(self, size: int, *, subject: str) -> None:
        if self.max_encoded_bytes is not None and size > self.max_encoded_bytes:
            raise ImageLimitError(f"{subject} image exceeds configured byte limit")


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def input_limits_from_env() -> ImageLimits:
    return ImageLimits(
        max_width=_positive_int_env("REMBG_MAX_INPUT_WIDTH", 10_000),
        max_height=_positive_int_env("REMBG_MAX_INPUT_HEIGHT", 10_000),
        max_pixels=_positive_int_env("REMBG_MAX_INPUT_PIXELS", 40_000_000),
    )


def output_limits_from_env() -> ImageLimits:
    return ImageLimits(
        max_width=_positive_int_env("REMBG_MAX_OUTPUT_WIDTH", 10_000),
        max_height=_positive_int_env("REMBG_MAX_OUTPUT_HEIGHT", 10_000),
        max_pixels=_positive_int_env("REMBG_MAX_OUTPUT_PIXELS", 40_000_000),
        max_encoded_bytes=_positive_int_env("REMBG_MAX_OUTPUT_BYTES", 40_000_000),
    )


def max_upload_bytes_from_env() -> int:
    return _positive_int_env("REMBG_MAX_UPLOAD_BYTES", 20_000_000)


def max_request_bytes_from_env() -> int:
    """Return the whole HTTP body limit, including multipart framing."""
    max_upload_bytes = max_upload_bytes_from_env()
    max_request_bytes = _positive_int_env("REMBG_MAX_REQUEST_BYTES", 21_000_000)
    if max_request_bytes < max_upload_bytes:
        raise ValueError(
            "REMBG_MAX_REQUEST_BYTES must be at least REMBG_MAX_UPLOAD_BYTES"
        )
    return max_request_bytes


def validate_image_bytes(data: bytes, limits: ImageLimits, *, subject: str) -> tuple[int, int]:
    """Read only image headers and reject decompression bombs before pixel allocation."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as opened:
                limits.validate_dimensions(opened.width, opened.height, subject=subject)
                return opened.width, opened.height
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ImageLimitError(f"{subject} image dimensions exceed configured limits") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, ImageLimitError):
            raise
        raise ValueError(f"{subject} bytes are not a valid image") from exc
