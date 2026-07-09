from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Literal, cast

from PIL import Image, ImageFilter

BackgroundColor = Literal["transparent", "white", "black", "custom"]
DespillColor = Literal["black", "white", "green", "blue", "custom"]


@dataclass(frozen=True)
class AlphaOptions:
    blur: float = 0.0
    erode: int = 0
    dilate: int = 0
    threshold: int = 0


@dataclass(frozen=True)
class DespillOptions:
    enabled: bool = False
    color: DespillColor = "black"
    hex_color: str = "000000"


def png_bytes(image: Image.Image) -> bytes:
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def load_rgba_png(data: bytes) -> Image.Image:
    with Image.open(BytesIO(data)) as image:
        return image.convert("RGBA")


def parse_hex_color(value: str) -> tuple[int, int, int]:
    cleaned = value.strip().lstrip("#")
    if len(cleaned) != 6:
        raise ValueError("hex colors must be exactly 6 characters")
    try:
        red = int(cleaned[0:2], 16)
        green = int(cleaned[2:4], 16)
        blue = int(cleaned[4:6], 16)
    except ValueError as exc:
        raise ValueError("hex colors must contain only hexadecimal characters") from exc
    return red, green, blue


def resolve_background_color(mode: BackgroundColor, hex_color: str) -> tuple[int, int, int, int] | None:
    if mode == "transparent":
        return None
    if mode == "white":
        return (255, 255, 255, 255)
    if mode == "black":
        return (0, 0, 0, 255)
    red, green, blue = parse_hex_color(hex_color)
    return (red, green, blue, 255)


def resolve_despill_color(mode: DespillColor, hex_color: str) -> tuple[int, int, int]:
    if mode == "black":
        return (0, 0, 0)
    if mode == "white":
        return (255, 255, 255)
    if mode == "green":
        return (0, 255, 0)
    if mode == "blue":
        return (0, 0, 255)
    return parse_hex_color(hex_color)


def refine_alpha(image: Image.Image, options: AlphaOptions) -> Image.Image:
    rgba = image.convert("RGBA")
    red, green, blue, alpha = rgba.split()

    if options.erode:
        alpha = alpha.filter(ImageFilter.MinFilter(options.erode * 2 + 1))
    if options.dilate:
        alpha = alpha.filter(ImageFilter.MaxFilter(options.dilate * 2 + 1))
    if options.blur:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=options.blur))
    if options.threshold:
        threshold = options.threshold
        alpha = alpha.point(lambda pixel: 255 if cast(int, pixel) >= threshold else 0)

    return Image.merge("RGBA", (red, green, blue, alpha))


def despill_image(image: Image.Image, options: DespillOptions) -> Image.Image:
    if not options.enabled:
        return image.convert("RGBA")

    rgba = image.convert("RGBA")
    target = resolve_despill_color(options.color, options.hex_color)
    target_channel = max(range(3), key=lambda index: target[index])
    if target[target_channel] == 0:
        return rgba

    pixels = rgba.load()
    if pixels is None:
        return rgba
    width, height = rgba.size
    for y in range(height):
        for x in range(width):
            red, green, blue, alpha = pixels[x, y]
            if alpha == 0:
                continue
            channels = [red, green, blue]
            dominant = channels[target_channel]
            other_channels = [channels[index] for index in range(3) if index != target_channel]
            spill_limit = max(other_channels)
            if dominant > spill_limit:
                edge_factor = 1.0 - (alpha / 255.0)
                reduction = int((dominant - spill_limit) * max(edge_factor, 0.25))
                channels[target_channel] = max(spill_limit, dominant - reduction)
                pixels[x, y] = (channels[0], channels[1], channels[2], alpha)
    return rgba


def composite_background(image: Image.Image, mode: BackgroundColor, hex_color: str) -> Image.Image:
    rgba = image.convert("RGBA")
    background = resolve_background_color(mode, hex_color)
    if background is None:
        return rgba
    canvas = Image.new("RGBA", rgba.size, background)
    canvas.alpha_composite(rgba)
    return canvas


def alpha_channel_png(image: Image.Image) -> bytes:
    return png_bytes(image.convert("RGBA").getchannel("A"))


def checker_preview(image: Image.Image, checker_size: int) -> Image.Image:
    rgba = image.convert("RGBA")
    size = max(1, checker_size)
    light = Image.new("RGBA", rgba.size, (230, 230, 230, 255))
    dark = Image.new("RGBA", rgba.size, (180, 180, 180, 255))
    mask = Image.new("1", rgba.size)
    mask_pixels = mask.load()
    if mask_pixels is None:
        return rgba
    for y in range(rgba.height):
        for x in range(rgba.width):
            mask_pixels[x, y] = ((x // size) + (y // size)) % 2
    background = Image.composite(dark, light, mask)
    background.alpha_composite(rgba)
    return background


def process_png_bytes(
    data: bytes,
    *,
    alpha: AlphaOptions,
    despill: DespillOptions,
    background_color: BackgroundColor,
    background_hex: str,
    return_alpha: bool,
    return_checker_preview: bool,
    checker_size: int,
) -> bytes:
    image = load_rgba_png(data)
    image = refine_alpha(image, alpha)
    image = despill_image(image, despill)

    if return_alpha:
        return alpha_channel_png(image)
    if return_checker_preview:
        return png_bytes(checker_preview(image, checker_size))

    image = composite_background(image, background_color, background_hex)
    return png_bytes(image)
