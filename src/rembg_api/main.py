from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
import onnxruntime as ort
from rembg import new_session, remove

from rembg_api.image_processing import AlphaOptions, DespillOptions, process_png_bytes

logger = logging.getLogger(__name__)

SupportedModel = Literal["isnet-general-use", "u2net", "u2netp", "isnet-anime", "silueta"]
OutputFormat = Literal["png"]
BackgroundColor = Literal["transparent", "white", "black", "custom"]
DespillColor = Literal["black", "white", "green", "blue", "custom"]

SUPPORTED_MODELS: tuple[str, ...] = (
    "isnet-general-use",
    "u2net",
    "u2netp",
    "isnet-anime",
    "silueta",
)

app = FastAPI(
    title="rembg-api",
    description="Thin bytes-in/bytes-out HTTP wrapper around rembg.",
    version="0.1.0",
)


@lru_cache(maxsize=len(SUPPORTED_MODELS))
def get_session(model: str):
    return new_session(model)


def get_onnxruntime_provider_info() -> dict[str, str | bool | list[str]]:
    providers = list(ort.get_available_providers())
    if "CUDAExecutionProvider" in providers:
        preferred_provider = "CUDAExecutionProvider"
    elif providers:
        preferred_provider = providers[0]
    else:
        preferred_provider = "unavailable"

    return {
        "onnxruntime_available_providers": providers,
        "preferred_provider": preferred_provider,
        "gpu_available": "CUDAExecutionProvider" in providers,
    }


@app.get("/health")
def health() -> dict[str, str | bool | list[str]]:
    return {"status": "ok", **get_onnxruntime_provider_info()}


@app.get("/models")
def models() -> dict[str, str | list[str]]:
    return {
        "default": "isnet-general-use",
        "supported": list(SUPPORTED_MODELS),
    }


@app.post(
    "/remove-background/",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}},
        400: {"description": "Invalid request"},
        500: {"description": "Internal processing error"},
    },
)
async def remove_background(
    file: Annotated[UploadFile, File(description="Input image file bytes")],
    model: Annotated[SupportedModel, Query(description="rembg model name")] = "isnet-general-use",
    only_mask: Annotated[bool, Query(description="Return rembg's raw mask output")] = False,
    post_process_mask: Annotated[bool, Query(description="Enable rembg mask post-processing")] = False,
    alpha_matting: Annotated[bool, Query(description="Enable rembg alpha matting")] = False,
    alpha_matting_foreground_threshold: Annotated[int, Query(ge=0, le=255)] = 240,
    alpha_matting_background_threshold: Annotated[int, Query(ge=0, le=255)] = 10,
    alpha_matting_erode_size: Annotated[int, Query(ge=0)] = 10,
    output_format: Annotated[OutputFormat, Query(description="Output image format; v1 supports PNG")] = "png",
    background_color: Annotated[BackgroundColor, Query(description="Optional background compositing mode")] = "transparent",
    background_hex: Annotated[str, Query(pattern=r"^#?[0-9a-fA-F]{6}$")] = "ffffff",
    alpha_blur: Annotated[float, Query(ge=0, le=20)] = 0.0,
    alpha_erode: Annotated[int, Query(ge=0, le=100)] = 0,
    alpha_dilate: Annotated[int, Query(ge=0, le=100)] = 0,
    alpha_threshold: Annotated[int, Query(ge=0, le=255)] = 0,
    despill: Annotated[bool, Query(description="Reduce selected color spill on foreground edges")] = False,
    despill_color: Annotated[DespillColor, Query(description="Spill color to reduce")] = "black",
    despill_hex: Annotated[str, Query(pattern=r"^#?[0-9a-fA-F]{6}$")] = "000000",
    return_alpha: Annotated[bool, Query(description="Return grayscale alpha PNG bytes")] = False,
    return_checker_preview: Annotated[bool, Query(description="Return checker-composited preview PNG bytes")] = False,
    checker_size: Annotated[int, Query(ge=2, le=128)] = 32,
) -> Response:
    if output_format != "png":
        raise HTTPException(status_code=400, detail="Only png output_format is supported")

    input_bytes = await file.read()
    if not input_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        session = get_session(model)
        removed = remove(
            input_bytes,
            session=session,
            only_mask=only_mask,
            post_process_mask=post_process_mask,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
        )
        if not isinstance(removed, bytes):
            raise RuntimeError(f"rembg.remove returned {type(removed)!r}, expected bytes")

        output_bytes = process_png_bytes(
            removed,
            alpha=AlphaOptions(
                blur=alpha_blur,
                erode=alpha_erode,
                dilate=alpha_dilate,
                threshold=alpha_threshold,
            ),
            despill=DespillOptions(
                enabled=despill,
                color=despill_color,
                hex_color=despill_hex,
            ),
            background_color=background_color,
            background_hex=background_hex,
            return_alpha=return_alpha,
            return_checker_preview=return_checker_preview,
            checker_size=checker_size,
        )
        return Response(content=output_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("remove-background failed: %r", exc)
        raise HTTPException(status_code=500, detail="Internal image processing error") from exc
