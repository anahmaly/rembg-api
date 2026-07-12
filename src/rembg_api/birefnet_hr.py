from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal, Protocol

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

BIREFNET_MODEL_NAME = "birefnet-hr-matting"
DEFAULT_MODEL_ID = "ZhengPeng7/BiRefNet_HR-matting"
DEFAULT_REVISION = "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f"
DEFAULT_LOCAL_PATH = "/models/ZhengPeng7/BiRefNet_HR-matting"
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

Device = Literal["auto", "cuda", "cpu"]
Precision = Literal["auto", "fp16", "fp32"]
ResolvedDevice = Literal["cuda", "cpu"]
ResolvedPrecision = Literal["fp16", "fp32"]


class ModelLoader(Protocol):
    def __call__(self, source: str, **kwargs: object) -> object: ...


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class BiRefNetConfig:
    source: str
    revision: str
    local_files_only: bool
    trust_remote_code: bool
    cache_dir: str | None
    device: Device
    precision: Precision
    inference_size: int
    foreground_refinement: bool
    max_concurrency: int

    @classmethod
    def from_env(cls) -> "BiRefNetConfig":
        local_only = _bool_env("BIREFNET_LOCAL_FILES_ONLY", True)
        source = os.getenv("BIREFNET_MODEL_PATH") or (
            DEFAULT_LOCAL_PATH
            if local_only
            else os.getenv("BIREFNET_MODEL_ID", DEFAULT_MODEL_ID)
        )
        config = cls(
            source=source,
            revision=os.getenv("BIREFNET_REVISION", DEFAULT_REVISION),
            local_files_only=local_only,
            trust_remote_code=_bool_env("BIREFNET_TRUST_REMOTE_CODE", False),
            cache_dir=os.getenv("BIREFNET_CACHE_DIR") or None,
            device=os.getenv("BIREFNET_DEVICE", "auto"),  # type: ignore[arg-type]
            precision=os.getenv("BIREFNET_PRECISION", "auto"),  # type: ignore[arg-type]
            inference_size=int(os.getenv("BIREFNET_INFERENCE_SIZE", "2048")),
            foreground_refinement=_bool_env("BIREFNET_FOREGROUND_REFINEMENT", False),
            max_concurrency=int(os.getenv("BIREFNET_MAX_CONCURRENCY", "1")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("BIREFNET_DEVICE must be auto, cuda, or cpu")
        if self.precision not in {"auto", "fp16", "fp32"}:
            raise ValueError("BIREFNET_PRECISION must be auto, fp16, or fp32")
        if not 512 <= self.inference_size <= 4096:
            raise ValueError("BIREFNET_INFERENCE_SIZE must be between 512 and 4096")
        if not 1 <= self.max_concurrency <= 16:
            raise ValueError("BIREFNET_MAX_CONCURRENCY must be between 1 and 16")
        if not self.local_files_only and not self.trust_remote_code:
            raise ValueError(
                "online bootstrap requires BIREFNET_TRUST_REMOTE_CODE=true"
            )
        if self.local_files_only and not Path(self.source).is_absolute():
            raise ValueError("offline BiRefNet source must be an absolute mounted path")


def _torch_status() -> dict[str, object]:
    try:
        import torch

        return {
            "torch_available": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count())
            if torch.cuda.is_available()
            else 0,
        }
    except Exception:
        return {
            "torch_available": False,
            "cuda_available": False,
            "cuda_device_count": 0,
        }


def _safe_source_name(source: str) -> str:
    return Path(source.rstrip("/")).name or "configured"


def resolve_runtime(
    device: Device, precision: Precision
) -> tuple[ResolvedDevice, ResolvedPrecision]:
    import torch

    resolved_device: ResolvedDevice = (
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )  # type: ignore[assignment]
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BiRefNet CUDA was requested but is unavailable")
    resolved_precision: ResolvedPrecision = (
        "fp16"
        if precision == "auto" and resolved_device == "cuda"
        else "fp32"
        if precision == "auto"
        else precision
    )
    if resolved_device == "cpu" and resolved_precision == "fp16":
        raise RuntimeError("BiRefNet fp16 is supported only on CUDA")
    return resolved_device, resolved_precision


class BiRefNetBackend:
    def __init__(
        self,
        config: BiRefNetConfig,
        device: ResolvedDevice,
        precision: ResolvedPrecision,
        *,
        loader: ModelLoader | None = None,
    ) -> None:
        import torch

        if config.local_files_only:
            path = Path(config.source)
            if not path.is_dir() or not os.access(path, os.R_OK):
                raise FileNotFoundError(
                    "configured offline BiRefNet model mount is unavailable"
                )
        if loader is None:
            from transformers import AutoModelForImageSegmentation

            loader = AutoModelForImageSegmentation.from_pretrained
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self._semaphore = threading.BoundedSemaphore(config.max_concurrency)
        kwargs: dict[str, object] = {
            "revision": config.revision,
            "trust_remote_code": config.trust_remote_code,
            "local_files_only": config.local_files_only,
        }
        if config.cache_dir:
            kwargs["cache_dir"] = config.cache_dir
        self.model = loader(config.source, **kwargs)
        self.model.to(device=self.device, dtype=self.dtype)  # type: ignore[attr-defined]
        self.model.eval()  # type: ignore[attr-defined]

    def remove_background(
        self,
        data: bytes,
        *,
        inference_size: int,
        foreground_refinement: bool,
    ) -> bytes:
        import torch
        from torchvision.transforms.functional import (
            normalize,
            pil_to_tensor,
            resize,
            to_pil_image,
        )

        try:
            with Image.open(BytesIO(data)) as opened:
                opened.load()
                if opened.width < 1 or opened.height < 1:
                    raise ValueError("input image has invalid dimensions")
                original = opened.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("uploaded bytes are not a valid image") from exc

        tensor = (
            resize(
                pil_to_tensor(original),
                [inference_size, inference_size],
                antialias=True,
            ).to(torch.float32)
            / 255.0
        )
        tensor = (
            normalize(tensor, NORMALIZE_MEAN, NORMALIZE_STD)
            .unsqueeze(0)
            .to(self.device, dtype=self.dtype)
        )
        with self._semaphore, torch.inference_mode():
            outputs = self.model(tensor)  # type: ignore[operator]
        try:
            prediction = outputs[-1].sigmoid()
        except (IndexError, TypeError, AttributeError) as exc:
            raise RuntimeError(
                "BiRefNet returned an invalid segmentation output"
            ) from exc
        if prediction.numel() == 0 or prediction.ndim not in {2, 3, 4}:
            raise RuntimeError("BiRefNet returned an invalid alpha tensor")
        prediction = prediction.detach().to(device="cpu", dtype=torch.float32)
        while prediction.ndim > 2:
            if prediction.shape[0] != 1:
                raise RuntimeError("BiRefNet returned an ambiguous alpha tensor")
            prediction = prediction[0]
        if not bool(torch.isfinite(prediction).all()):
            raise RuntimeError("BiRefNet returned non-finite alpha values")
        prediction = prediction.clamp(0, 1)
        prediction = resize(
            prediction.unsqueeze(0), [original.height, original.width], antialias=True
        ).squeeze(0)
        alpha = to_pil_image(prediction, mode="L")

        rgba = original.convert("RGBA")
        rgba.putalpha(alpha)
        if foreground_refinement:
            # Conservative alpha-aware edge refinement: remove hidden RGB from fully
            # transparent pixels while retaining source colors for every visible pixel.
            pixels = rgba.load()
            for y in range(rgba.height):
                for x in range(rgba.width):
                    r, g, b, a = pixels[x, y]
                    if a == 0:
                        pixels[x, y] = (0, 0, 0, 0)
        output = BytesIO()
        rgba.save(output, "PNG")
        return output.getvalue()


BackendKey = tuple[
    str, str, bool, bool, str | None, ResolvedDevice, ResolvedPrecision, int
]


def _load_backend(key: BackendKey) -> BiRefNetBackend:
    (
        source,
        revision,
        local_files_only,
        trust_remote_code,
        cache_dir,
        device,
        precision,
        max_concurrency,
    ) = key
    config = BiRefNetConfig(
        source,
        revision,
        local_files_only,
        trust_remote_code,
        cache_dir,
        device,
        precision,
        2048,
        False,
        max_concurrency,
    )
    return BiRefNetBackend(config, device, precision)


class BackendCache:
    """Thread-safe bounded LRU cache with atomic lazy construction per key."""

    def __init__(self, maxsize: int = 8) -> None:
        self.maxsize = maxsize
        self._lock = threading.RLock()
        self._entries: OrderedDict[BackendKey, BiRefNetBackend] = OrderedDict()
        self._loading: dict[BackendKey, threading.Event] = {}
        self._generation = 0

    def get(self, key: BackendKey) -> BiRefNetBackend:
        while True:
            with self._lock:
                backend = self._entries.get(key)
                if backend is not None:
                    self._entries.move_to_end(key)
                    return backend
                ready = self._loading.get(key)
                if ready is None:
                    ready = self._loading[key] = threading.Event()
                    generation = self._generation
                    break
            ready.wait()

        try:
            backend = _load_backend(key)
        except BaseException:
            with self._lock:
                self._loading.pop(key).set()
            raise

        with self._lock:
            # A clear racing a load drops the cache reference, but the requesting
            # worker can still safely use the backend it just constructed.
            if generation == self._generation:
                self._entries[key] = backend
                if len(self._entries) > self.maxsize:
                    # In-flight calls retain their own reference until inference and
                    # semaphore release finish; eviction only drops the cache reference.
                    self._entries.popitem(last=False)
            self._loading.pop(key).set()
        return backend

    def contains(self, key: BackendKey) -> bool:
        with self._lock:
            return key in self._entries

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._generation += 1
            return count


_backend_cache = BackendCache(maxsize=8)


def _cache_key(config: BiRefNetConfig) -> BackendKey:
    device, precision = resolve_runtime(config.device, config.precision)
    return (
        config.source,
        config.revision,
        config.local_files_only,
        config.trust_remote_code,
        config.cache_dir,
        device,
        precision,
        config.max_concurrency,
    )


def get_backend(config: BiRefNetConfig) -> BiRefNetBackend:
    return _backend_cache.get(_cache_key(config))


def clear_cache() -> int:
    """Drop cached references while allowing active calls to finish safely."""
    return _backend_cache.clear()


def remove_with_birefnet(
    data: bytes,
    *,
    inference_size: int | None = None,
    foreground_refinement: bool | None = None,
    config: BiRefNetConfig | None = None,
) -> bytes:
    config = config or BiRefNetConfig.from_env()
    size = config.inference_size if inference_size is None else inference_size
    if not 512 <= size <= 4096:
        raise ValueError("BiRefNet inference size must be between 512 and 4096")
    refinement = (
        config.foreground_refinement
        if foreground_refinement is None
        else foreground_refinement
    )
    return get_backend(config).remove_background(
        data, inference_size=size, foreground_refinement=refinement
    )


def health_info(config: BiRefNetConfig | None = None) -> dict[str, object]:
    try:
        config = config or BiRefNetConfig.from_env()
        device, precision = resolve_runtime(config.device, config.precision)
        key = (
            config.source,
            config.revision,
            config.local_files_only,
            config.trust_remote_code,
            config.cache_dir,
            device,
            precision,
            config.max_concurrency,
        )
        mounted = (
            Path(config.source).is_dir() and os.access(config.source, os.R_OK)
            if config.local_files_only
            else None
        )
        return {
            "configured": True,
            "model": _safe_source_name(config.source),
            "revision": config.revision,
            "device": device,
            "precision": precision,
            "local_files_only": config.local_files_only,
            "trust_remote_code": config.trust_remote_code,
            "model_mount_available": mounted,
            "loaded": _backend_cache.contains(key),
            "ready": (bool(mounted) and config.trust_remote_code)
            if config.local_files_only
            else config.trust_remote_code,
            "max_concurrency": config.max_concurrency,
            **_torch_status(),
        }
    except Exception as exc:
        return {
            "configured": False,
            "ready": False,
            "loaded": False,
            "error": str(exc),
            **_torch_status(),
        }
