from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from rembg_api.limits import (
    ImageLimitError,
    ImageLimits,
    max_request_bytes_from_env,
    validate_image_bytes,
)


def png_bytes(size: tuple[int, int] = (2, 2)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size).save(output, "PNG")
    return output.getvalue()


def test_request_byte_limit_must_allow_file_limit(monkeypatch) -> None:
    monkeypatch.setenv("REMBG_MAX_UPLOAD_BYTES", "20")
    monkeypatch.setenv("REMBG_MAX_REQUEST_BYTES", "19")

    with pytest.raises(ValueError, match="REMBG_MAX_REQUEST_BYTES"):
        max_request_bytes_from_env()


def test_decoded_pixel_limit_allows_exact_edge() -> None:
    limits = ImageLimits(max_width=2, max_height=2, max_pixels=4)
    validate_image_bytes(png_bytes((2, 2)), limits, subject="upload")


def test_decoded_pixel_limit_rejects_header_before_decode() -> None:
    limits = ImageLimits(max_width=2, max_height=2, max_pixels=4)
    with pytest.raises(ImageLimitError, match="image dimensions exceed"):
        validate_image_bytes(png_bytes((3, 2)), limits, subject="upload")


def test_decompression_bomb_is_a_safe_limit_error(monkeypatch) -> None:
    limits = ImageLimits(max_width=100, max_height=100, max_pixels=10_000)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ImageLimitError, match="image dimensions exceed"):
        validate_image_bytes(png_bytes((2, 2)), limits, subject="upload")
