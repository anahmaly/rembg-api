from __future__ import annotations

import asyncio
import gc
import logging
import sys
import threading
from functools import lru_cache, partial
from typing import Annotated, Literal

from anyio import to_thread

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
import onnxruntime as ort
from rembg import new_session, remove

from rembg_api.bria_rmbg import (
    BRIA_RMBG_2_MODEL_ID,
    BriaDevice,
    BriaDType,
    clear_bria_backend_cache,
    configured_model_path,
    get_torch_status,
    get_bria_rmbg_2_backend,
    local_model_status,
    release_request_memory,
    remove_with_bria_rmbg_2,
    should_release_cuda_cache_after_request,
)
from rembg_api.birefnet_hr import (
    BIREFNET_MODEL_NAME,
    BiRefNetConfig,
    clear_cache as clear_birefnet_cache,
    health_info as birefnet_health_info,
    remove_with_birefnet,
)
from rembg_api.image_processing import AlphaOptions, DespillOptions, process_png_bytes
from rembg_api.limits import (
    ImageLimitError,
    InvalidImageError,
    input_limits_from_env,
    max_request_bytes_from_env,
    max_upload_bytes_from_env,
    output_limits_from_env,
    validate_image_bytes,
    validate_output_image_bytes,
)

logger = logging.getLogger(__name__)


class _RequestBodyTooLarge(BaseException):
    """Escape multipart parsing without being rewritten as a malformed form."""


class RequestBodyLimitMiddleware:
    """Bound every byte consumed for upload endpoints, including multipart overhead."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["path"] != "/remove-background/":
            await self.app(scope, receive, send)
            return

        max_bytes = max_request_bytes_from_env()
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared is not None:
            if not declared or any(byte < ord("0") or byte > ord("9") for byte in declared):
                await JSONResponse(
                    status_code=400, content={"detail": "Invalid Content-Length header"}
                )(scope, receive, send)
                return
            declared_size = int(declared)
            if declared_size > max_bytes:
                await JSONResponse(
                    status_code=413,
                    content={"detail": "Request body is larger than this service accepts"},
                )(scope, receive, send)
                return

        consumed = 0
        response_started = False

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > max_bytes:
                    raise _RequestBodyTooLarge
            return message

        async def tracked_send(message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            if response_started:
                # Multipart parsing completes before this route starts a response;
                # guard anyway so a future streaming route cannot double-send.
                raise RuntimeError("request limit exceeded after response start")
            await JSONResponse(
                status_code=413,
                content={"detail": "Request body is larger than this service accepts"},
            )(scope, receive, send)


SupportedModel = Literal[
    "isnet-general-use",
    "u2net",
    "u2netp",
    "isnet-anime",
    "silueta",
    "bria-rmbg-2.0",
    "birefnet-hr-matting",
]
OutputFormat = Literal["png"]
BackgroundColor = Literal["transparent", "white", "black", "custom"]
DespillColor = Literal["black", "white", "green", "blue", "custom"]

REMBG_MODELS: tuple[str, ...] = (
    "isnet-general-use",
    "u2net",
    "u2netp",
    "isnet-anime",
    "silueta",
)
SUPPORTED_MODELS: tuple[str, ...] = (
    *REMBG_MODELS,
    BRIA_RMBG_2_MODEL_ID,
    BIREFNET_MODEL_NAME,
)

app = FastAPI(
    title="rembg-api",
    description="Thin bytes-in/bytes-out HTTP wrapper around rembg.",
    version="0.1.0",
)
app.add_middleware(RequestBodyLimitMiddleware)

_UPLOAD_CHUNK_BYTES = 64 * 1024
_birefnet_admission_lock = threading.Lock()
_birefnet_admissions: dict[BiRefNetConfig, threading.BoundedSemaphore] = {}


def _birefnet_admission(config: BiRefNetConfig) -> threading.BoundedSemaphore:
    """Return the process-wide, non-queueing heavy-path admission gate."""
    with _birefnet_admission_lock:
        return _birefnet_admissions.setdefault(
            config, threading.BoundedSemaphore(config.max_concurrency)
        )


async def _read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413, detail="Uploaded image is larger than this service accepts"
            )
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return b"".join(chunks)



def _consume_background_task_result(task: asyncio.Task[bytes]) -> None:
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        # The original requester has already been notified or disconnected. The
        # worker's finally block has released admission before this callback runs.
        pass


@lru_cache(maxsize=len(REMBG_MODELS))
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


def get_bria_model_info() -> dict[str, object]:
    status = local_model_status()
    return {
        "model_path": status.path,
        "model_path_exists": status.exists,
        "model_path_is_dir": status.is_dir,
        "model_path_readable": status.readable,
        "model_path_available": status.available,
        **get_torch_status(),
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        **get_onnxruntime_provider_info(),
        "bria_rmbg_2": get_bria_model_info(),
        "birefnet_hr_matting": birefnet_health_info(),
    }


@app.get("/models")
def models() -> dict[str, object]:
    return {
        "default": "isnet-general-use",
        "supported": list(SUPPORTED_MODELS),
        "details": {
            BRIA_RMBG_2_MODEL_ID: {
                "backend": "torch-transformers-local",
                "configured_path": configured_model_path(),
                **get_bria_model_info(),
            },
            BIREFNET_MODEL_NAME: {
                "backend": "torch-transformers",
                **birefnet_health_info(),
            },
        },
    }


@app.post("/cache/clear")
def clear_caches(
    release_cuda_cache: Annotated[
        bool,
        Query(
            description="Run torch.cuda.empty_cache() after clearing caches when CUDA is available"
        ),
    ] = True,
) -> dict[str, object]:
    """Clear cached rembg sessions and BRIA backends for LAN-local resource recovery."""
    get_session.cache_clear()
    bria_was_loaded = get_bria_rmbg_2_backend.cache_info().currsize > 0
    clear_bria_backend_cache(release_cuda_cache=False)
    birefnet_cleared = clear_birefnet_cache()
    gc.collect()
    cuda_cache_released = False
    if release_cuda_cache and (bria_was_loaded or birefnet_cleared > 0):
        # Loaded torch backends necessarily imported torch. Do not import it just
        # because an operator clears a service that has never used one.
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            cuda_cache_released = True
    return {
        "status": "ok",
        "rembg_sessions_cleared": True,
        "bria_backends_cleared": True,
        "birefnet_backends_cleared": True,
        "cuda_cache_release_requested": release_cuda_cache,
        "cuda_cache_released": cuda_cache_released,
    }


@app.post(
    "/remove-background/",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}},
        400: {"description": "Invalid request or Content-Length header"},
        413: {"description": "HTTP request body, image, or upload exceeds service limits"},
        429: {"description": "BiRefNet capacity is currently unavailable"},
        500: {"description": "Internal processing error"},
    },
)
async def remove_background(
    file: Annotated[UploadFile, File(description="Input image file bytes")],
    model: Annotated[
        SupportedModel, Query(description="Background-removal model name")
    ] = "isnet-general-use",
    only_mask: Annotated[
        bool,
        Query(description="Return rembg's raw mask output; ignored for bria-rmbg-2.0"),
    ] = False,
    post_process_mask: Annotated[
        bool,
        Query(
            description="Enable rembg mask post-processing; ignored for bria-rmbg-2.0"
        ),
    ] = False,
    alpha_matting: Annotated[
        bool, Query(description="Enable rembg alpha matting; ignored for bria-rmbg-2.0")
    ] = False,
    alpha_matting_foreground_threshold: Annotated[int, Query(ge=0, le=255)] = 240,
    alpha_matting_background_threshold: Annotated[int, Query(ge=0, le=255)] = 10,
    alpha_matting_erode_size: Annotated[int, Query(ge=0)] = 10,
    model_input_size: Annotated[
        int, Query(ge=512, le=2048, description="BRIA RMBG-2.0 square model input size")
    ] = 1024,
    device: Annotated[
        BriaDevice, Query(description="BRIA RMBG-2.0 device selection")
    ] = "auto",
    dtype: Annotated[
        BriaDType, Query(description="BRIA RMBG-2.0 model precision")
    ] = "auto",
    output_format: Annotated[
        OutputFormat, Query(description="Output image format; v1 supports PNG")
    ] = "png",
    background_color: Annotated[
        BackgroundColor, Query(description="Optional background compositing mode")
    ] = "transparent",
    background_hex: Annotated[str, Query(pattern=r"^#?[0-9a-fA-F]{6}$")] = "ffffff",
    alpha_blur: Annotated[float, Query(ge=0, le=20)] = 0.0,
    alpha_erode: Annotated[int, Query(ge=0, le=100)] = 0,
    alpha_dilate: Annotated[int, Query(ge=0, le=100)] = 0,
    alpha_threshold: Annotated[int, Query(ge=0, le=255)] = 0,
    despill: Annotated[
        bool, Query(description="Reduce selected color spill on foreground edges")
    ] = False,
    despill_color: Annotated[
        DespillColor, Query(description="Spill color to reduce")
    ] = "black",
    despill_hex: Annotated[str, Query(pattern=r"^#?[0-9a-fA-F]{6}$")] = "000000",
    return_alpha: Annotated[
        bool, Query(description="Return grayscale alpha PNG bytes")
    ] = False,
    return_checker_preview: Annotated[
        bool, Query(description="Return checker-composited preview PNG bytes")
    ] = False,
    checker_size: Annotated[int, Query(ge=2, le=128)] = 32,
    release_cuda_cache: Annotated[
        bool | None,
        Query(
            description=(
                "BRIA RMBG-2.0 only: call torch.cuda.empty_cache() after the request. "
                "Defaults to BRIA_RELEASE_CUDA_CACHE_AFTER_REQUEST, true when unset."
            )
        ),
    ] = None,
    birefnet_inference_size: Annotated[
        int | None,
        Query(
            ge=512,
            le=4096,
            description="BiRefNet square input size; env/default is 2048",
        ),
    ] = None,
    birefnet_foreground_refinement: Annotated[
        bool | None,
        Query(
            description="BiRefNet only: clear hidden RGB for fully transparent pixels; alpha is unchanged"
        ),
    ] = None,
) -> Response:
    if output_format != "png":
        raise HTTPException(
            status_code=400, detail="Only png output_format is supported"
        )

    max_upload_bytes = max_upload_bytes_from_env()
    input_limits = input_limits_from_env()
    output_limits = output_limits_from_env()
    if output_limits.max_encoded_bytes is None:
        raise RuntimeError("output byte limit is required")
    bria_request = model == BRIA_RMBG_2_MODEL_ID
    should_release_bria_cuda_cache = (
        should_release_cuda_cache_after_request()
        if release_cuda_cache is None
        else release_cuda_cache
    )
    process = partial(
        process_png_bytes,
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
        max_encoded_bytes=output_limits.max_encoded_bytes,
    )

    admission: threading.BoundedSemaphore | None = None
    try:
        if model == BIREFNET_MODEL_NAME:
            # This instant, process-wide gate has no waiting queue. It is acquired
            # before file reading and remains owned by the worker until *all* heavy
            # BiRefNet work (load/decode/preprocess/infer/postprocess) has finished.
            config = BiRefNetConfig.from_env()
            candidate_admission = _birefnet_admission(config)
            if not candidate_admission.acquire(blocking=False):
                raise HTTPException(
                    status_code=429,
                    detail="BiRefNet is busy; please try again shortly",
                )
            admission = candidate_admission
            input_bytes = await _read_upload_limited(file, max_upload_bytes)
            worker_admission = admission

            def work() -> bytes:
                try:
                    removed = remove_with_birefnet(
                        input_bytes,
                        inference_size=birefnet_inference_size,
                        foreground_refinement=birefnet_foreground_refinement,
                        config=config,
                        input_limits=input_limits,
                        output_limits=output_limits,
                    )
                    validate_output_image_bytes(removed, output_limits)
                    encoded = process(removed)
                    output_limits.validate_encoded_bytes(len(encoded), subject="output")
                    validate_output_image_bytes(encoded, output_limits)
                    return encoded
                finally:
                    worker_admission.release()

            # A shielded task is the bounded ownership handoff: cancellation stops
            # response waiting, but never cancels or abandons the admitted worker.
            task = asyncio.create_task(to_thread.run_sync(work))
            task.add_done_callback(_consume_background_task_result)
            admission = None
            output_bytes = await asyncio.shield(task)
        else:
            input_bytes = await _read_upload_limited(file, max_upload_bytes)
            validate_image_bytes(input_bytes, input_limits, subject="upload")
            if bria_request:
                removed = remove_with_bria_rmbg_2(
                    input_bytes,
                    model_input_size=model_input_size,
                    device=device,
                    dtype=dtype,
                    release_cuda_cache=release_cuda_cache,
                    cleanup_after_request=False,
                )
            else:
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
                raise RuntimeError(
                    f"background removal returned {type(removed)!r}, expected bytes"
                )
            validate_output_image_bytes(removed, output_limits)
            output_bytes = process(removed)
            output_limits.validate_encoded_bytes(len(output_bytes), subject="output")
            validate_output_image_bytes(output_bytes, output_limits)
        return Response(content=output_bytes, media_type="image/png")
    except HTTPException:
        raise
    except ImageLimitError as exc:
        raise HTTPException(status_code=413, detail="Image exceeds this service's limits") from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc
    except Exception as exc:
        logger.exception(
            "remove-background failed: model=%s model_input_size=%s device=%s dtype=%s error=%r",
            model,
            model_input_size,
            device,
            dtype,
            exc,
        )
        raise HTTPException(
            status_code=500, detail="Internal image processing error"
        ) from exc
    finally:
        if admission is not None:
            admission.release()
        if bria_request:
            release_request_memory(release_cuda_cache=should_release_bria_cuda_cache)
