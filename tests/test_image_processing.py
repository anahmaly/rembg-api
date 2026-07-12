from __future__ import annotations

from io import BytesIO

from PIL import Image

from rembg_api.image_processing import (
    AlphaOptions,
    DespillOptions,
    composite_background,
    parse_hex_color,
    process_png_bytes,
    refine_alpha,
)

from helpers import make_png


def read_image(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data))


def test_parse_hex_color_accepts_hash_and_plain_values() -> None:
    assert parse_hex_color("#aabbcc") == (170, 187, 204)
    assert parse_hex_color("00ff10") == (0, 255, 16)


def test_alpha_threshold_binarizes_alpha() -> None:
    image = Image.new("RGBA", (2, 1))
    image.putdata([(255, 0, 0, 120), (255, 0, 0, 200)])

    refined = refine_alpha(image, AlphaOptions(threshold=128))

    assert list(refined.getchannel("A").get_flattened_data()) == [0, 255]


def test_composite_background_preserves_rgba_png_shape() -> None:
    image = Image.new("RGBA", (1, 1), (255, 0, 0, 128))

    composited = composite_background(image, "white", "ffffff")

    assert composited.mode == "RGBA"
    assert composited.getpixel((0, 0))[3] == 255


def test_process_png_bytes_can_return_alpha_mask() -> None:
    output = process_png_bytes(
        make_png((10, 20, 30, 128)),
        alpha=AlphaOptions(),
        despill=DespillOptions(),
        background_color="transparent",
        background_hex="ffffff",
        return_alpha=True,
        return_checker_preview=False,
        checker_size=8,
        max_encoded_bytes=1_000_000,
    )

    image = read_image(output)
    assert image.mode == "L"
    assert image.getpixel((0, 0)) == 128


def test_process_png_bytes_checker_preview_is_opaque() -> None:
    output = process_png_bytes(
        make_png((10, 20, 30, 128), size=(4, 4)),
        alpha=AlphaOptions(),
        despill=DespillOptions(),
        background_color="transparent",
        background_hex="ffffff",
        return_alpha=False,
        return_checker_preview=True,
        checker_size=2,
        max_encoded_bytes=1_000_000,
    )

    image = read_image(output).convert("RGBA")
    assert image.getpixel((0, 0))[3] == 255
