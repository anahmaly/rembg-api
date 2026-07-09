from __future__ import annotations

from pathlib import Path


GPU_DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile.gpu"


def test_gpu_dockerfile_uses_cuda_13_cudnn_runtime_base() -> None:
    dockerfile = GPU_DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04" in dockerfile
    assert "FROM python:" not in dockerfile


def test_gpu_dockerfile_pins_onnxruntime_gpu_matching_cuda_runtime() -> None:
    dockerfile = GPU_DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG ONNXRUNTIME_GPU_VERSION=1.27.0" in dockerfile
    assert 'onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}' in dockerfile
    assert "libcudart.so.13" in dockerfile
