from __future__ import annotations

from io import BytesIO

from PIL import Image


def make_png(color=(255, 0, 0, 128), size=(2, 2)) -> bytes:
    out = BytesIO()
    Image.new("RGBA", size, color).save(out, format="PNG")
    return out.getvalue()
