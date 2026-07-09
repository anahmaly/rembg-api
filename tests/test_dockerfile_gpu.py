from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CPU_DOCKERFILE = REPO_ROOT / "Dockerfile.cpu"
DEFAULT_DOCKERFILE = REPO_ROOT / "Dockerfile"
GPU_DOCKERFILE = REPO_ROOT / "Dockerfile.gpu"
COMPOSE_FILE = REPO_ROOT / "compose.yml"


def test_cpu_dockerfile_is_explicit_and_no_default_dockerfile_exists() -> None:
    assert CPU_DOCKERFILE.is_file()
    assert not DEFAULT_DOCKERFILE.exists()


def test_compose_services_reference_explicit_dockerfiles() -> None:
    compose = COMPOSE_FILE.read_text(encoding="utf-8")

    assert "dockerfile: Dockerfile.cpu" in compose
    assert "dockerfile: Dockerfile.gpu" in compose
    assert "dockerfile: Dockerfile\n" not in compose


def test_gpu_dockerfile_uses_cuda_13_cudnn_runtime_base() -> None:
    dockerfile = GPU_DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04" in dockerfile
    assert "FROM python:" not in dockerfile


def test_gpu_dockerfile_pins_onnxruntime_gpu_matching_cuda_runtime() -> None:
    dockerfile = GPU_DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG ONNXRUNTIME_GPU_VERSION=1.27.0" in dockerfile
    assert 'onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}' in dockerfile
    assert "libcudart.so.13" in dockerfile


def test_gpu_dockerfile_installs_torch_transformers_dependencies() -> None:
    dockerfile = GPU_DOCKERFILE.read_text(encoding="utf-8")

    assert "download.pytorch.org/whl/cu128" in dockerfile
    assert "torch torchvision" in dockerfile
    assert "pip install ." in dockerfile


def test_compose_mounts_bria_rmbg_2_model_path() -> None:
    compose = COMPOSE_FILE.read_text(encoding="utf-8")

    assert "${HOME}/models/briaai/RMBG-2.0:/models/briaai/RMBG-2.0:ro" in compose
    assert "BRIA_RMBG_2_MODEL_PATH=/models/briaai/RMBG-2.0" in compose
