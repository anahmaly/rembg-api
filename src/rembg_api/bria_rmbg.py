from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
import importlib
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image

logger = logging.getLogger(__name__)

BRIA_RMBG_2_MODEL_ID = "bria-rmbg-2.0"
DEFAULT_BRIA_RMBG_2_MODEL_PATH = "/models/briaai/RMBG-2.0"
BRIA_RMBG_2_MODEL_PATH_ENV = "BRIA_RMBG_2_MODEL_PATH"
BRIA_RMBG_2_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
BRIA_RMBG_2_NORMALIZE_STD = (0.229, 0.224, 0.225)
BRIA_RMBG_2_REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "transformers",
    "timm",
    "kornia",
)

BriaDevice = Literal["auto", "cuda", "cpu"]
BriaDType = Literal["auto", "fp16", "fp32"]


@dataclass(frozen=True)
class LocalModelStatus:
    path: str
    exists: bool
    is_dir: bool
    readable: bool
    available: bool


def configured_model_path() -> str:
    return os.environ.get(BRIA_RMBG_2_MODEL_PATH_ENV, DEFAULT_BRIA_RMBG_2_MODEL_PATH)


def local_model_status(path: str | None = None) -> LocalModelStatus:
    model_path = path or configured_model_path()
    resolved = Path(model_path)
    exists = resolved.exists()
    is_dir = resolved.is_dir()
    readable = os.access(resolved, os.R_OK) if exists else False
    return LocalModelStatus(
        path=model_path,
        exists=exists,
        is_dir=is_dir,
        readable=readable,
        available=exists and is_dir and readable,
    )


def get_torch_status() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime image
        return {
            "torch_available": False,
            "torch_error": repr(exc),
            "cuda_available": False,
        }

    return {
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _resolve_device(requested: BriaDevice):
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("BRIA RMBG-2.0 requested device=cuda but CUDA is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(requested: BriaDType, device):
    import torch

    if requested == "fp16":
        if device.type != "cuda":
            raise RuntimeError("BRIA RMBG-2.0 dtype=fp16 requires a CUDA device")
        return torch.float16
    if requested == "fp32":
        return torch.float32
    return torch.float16 if device.type == "cuda" else torch.float32


def _check_bria_runtime_dependencies() -> None:
    missing: list[str] = []
    errors: dict[str, str] = {}
    for module_name in BRIA_RMBG_2_REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or (exc.name is not None and exc.name.startswith(f"{module_name}.")):
                missing.append(module_name)
            else:
                errors[module_name] = repr(exc)
        except Exception as exc:  # pragma: no cover - depends on binary runtime
            errors[module_name] = repr(exc)

    if missing or errors:
        detail_parts: list[str] = []
        if missing:
            detail_parts.append(f"missing={','.join(missing)}")
        if errors:
            detail_parts.append(
                "import_errors=" + ",".join(f"{name}:{error}" for name, error in errors.items())
            )
        raise RuntimeError(
            "BRIA RMBG-2.0 runtime dependencies are unavailable; "
            + " ".join(detail_parts)
            + ". Rebuild the container after installing pyproject dependencies."
        )


def build_bria_preprocess_transform(model_input_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((model_input_size, model_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(BRIA_RMBG_2_NORMALIZE_MEAN, BRIA_RMBG_2_NORMALIZE_STD),
        ]
    )


class BriaRmbg2Backend:
    def __init__(self, model_path: str, device: BriaDevice, dtype: BriaDType) -> None:
        status = local_model_status(model_path)
        if not status.available:
            raise FileNotFoundError(
                "BRIA RMBG-2.0 local model path is unavailable: "
                f"path={status.path!r} exists={status.exists} is_dir={status.is_dir} readable={status.readable}"
            )

        _check_bria_runtime_dependencies()

        import torch
        from transformers import AutoModelForImageSegmentation

        self.torch = torch
        self.model_path = model_path
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype, self.device)
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    def remove_background(self, input_bytes: bytes, *, model_input_size: int) -> bytes:
        import torch
        from torchvision import transforms
        from torchvision.transforms.functional import resize as resize_tensor

        with Image.open(BytesIO(input_bytes)) as image:
            original = image.convert("RGBA")
        rgb = original.convert("RGB")
        original_size = (rgb.height, rgb.width)

        transform = build_bria_preprocess_transform(model_input_size)
        inputs = transform(rgb).unsqueeze(0).to(device=self.device, dtype=self.dtype)

        with torch.inference_mode():
            outputs = self.model(inputs)
            pred = outputs[-1].sigmoid().detach().to("cpu", dtype=torch.float32)

        pred = resize_tensor(pred, original_size, antialias=True)
        pred = pred.squeeze()
        pred_min = pred.min()
        pred_max = pred.max()
        denom = pred_max - pred_min
        if float(denom) > 0:
            pred = (pred - pred_min) / denom
        mask = transforms.ToPILImage()(pred).convert("L")

        result = original.copy()
        result.putalpha(mask)
        output = BytesIO()
        result.save(output, format="PNG")
        return output.getvalue()


@lru_cache(maxsize=4)
def get_bria_rmbg_2_backend(model_path: str, device: BriaDevice, dtype: BriaDType) -> BriaRmbg2Backend:
    logger.info(
        "Loading BRIA RMBG-2.0 backend from local path %s with device=%s dtype=%s",
        model_path,
        device,
        dtype,
    )
    return BriaRmbg2Backend(model_path, device, dtype)


def remove_with_bria_rmbg_2(
    input_bytes: bytes,
    *,
    model_input_size: int,
    device: BriaDevice,
    dtype: BriaDType,
    model_path: str | None = None,
) -> bytes:
    path = model_path or configured_model_path()
    backend = get_bria_rmbg_2_backend(path, device, dtype)
    return backend.remove_background(input_bytes, model_input_size=model_input_size)
