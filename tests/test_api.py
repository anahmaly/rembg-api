from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from fastapi.testclient import TestClient
from PIL import Image
from rembg_api import main
from rembg_api.bria_rmbg import (
    BRIA_RMBG_2_NORMALIZE_MEAN,
    BRIA_RMBG_2_NORMALIZE_STD,
    BRIA_RMBG_2_REQUIRED_MODULES,
    _check_bria_runtime_dependencies,
    get_torch_status,
    resolve_bria_backend_cache_key,
    should_release_cuda_cache_after_request,
    local_model_status,
)

from helpers import make_png


def test_health(monkeypatch) -> None:
    monkeypatch.setattr(
        main.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(main, "get_bria_model_info", lambda: {"model_path_available": False})
    monkeypatch.setattr(main, "birefnet_health_info", lambda: {"loaded": False})
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "onnxruntime_available_providers": ["CPUExecutionProvider"],
        "preferred_provider": "CPUExecutionProvider",
        "gpu_available": False,
        "bria_rmbg_2": {"model_path_available": False},
        "birefnet_hr_matting": {"loaded": False},
    }


def test_health_reports_cuda_provider_when_available(monkeypatch) -> None:
    monkeypatch.setattr(
        main.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(main, "get_bria_model_info", lambda: {"model_path_available": True})
    monkeypatch.setattr(main, "birefnet_health_info", lambda: {"loaded": False})
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "onnxruntime_available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "preferred_provider": "CUDAExecutionProvider",
        "gpu_available": True,
        "bria_rmbg_2": {"model_path_available": True},
        "birefnet_hr_matting": {"loaded": False},
    }


def test_models_lists_supported_default(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_bria_model_info", lambda: {"model_path_available": False})
    monkeypatch.setattr(main, "birefnet_health_info", lambda: {"loaded": False})
    client = TestClient(main.app)
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert body["default"] == "isnet-general-use"
    assert "u2net" in body["supported"]
    assert "bria-rmbg-2.0" in body["supported"]
    assert body["details"]["bria-rmbg-2.0"]["backend"] == "torch-transformers-local"
    assert body["details"]["bria-rmbg-2.0"]["model_path_available"] is False


def test_cache_clear_endpoint_clears_rembg_and_bria_caches(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_clear_bria_backend_cache(*, release_cuda_cache: bool) -> None:
        calls["release_cuda_cache"] = release_cuda_cache

    main.get_session.cache_clear()
    monkeypatch.setattr(main, "new_session", lambda model: f"session:{model}")
    assert main.get_session("u2net") == "session:u2net"
    assert main.get_session.cache_info().currsize == 1
    monkeypatch.setattr(main, "clear_bria_backend_cache", fake_clear_bria_backend_cache)

    client = TestClient(main.app)
    response = client.post("/cache/clear?release_cuda_cache=false")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "rembg_sessions_cleared": True,
        "bria_backends_cleared": True,
        "birefnet_backends_cleared": True,
    }
    assert main.get_session.cache_info().currsize == 0
    assert calls["release_cuda_cache"] is False


def test_remove_background_bytes_in_bytes_out(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_new_session(model: str) -> str:
        calls["model"] = model
        return f"session:{model}"

    def fake_remove(data: bytes, **kwargs) -> bytes:
        calls["data"] = data
        calls["kwargs"] = kwargs
        return make_png((255, 0, 0, 128))

    main.get_session.cache_clear()
    monkeypatch.setattr(main, "new_session", fake_new_session)
    monkeypatch.setattr(main, "remove", fake_remove)

    client = TestClient(main.app)
    response = client.post(
        "/remove-background/?model=u2net&background_color=white&alpha_threshold=1",
        files={"file": ("input.png", make_png(), "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
    assert calls["model"] == "u2net"
    kwargs = cast(dict[str, Any], calls["kwargs"])
    assert kwargs["session"] == "session:u2net"
    assert kwargs["alpha_matting_foreground_threshold"] == 240


def test_bria_rmbg_routes_to_local_backend_and_returns_png(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_bria_remove(data: bytes, **kwargs) -> bytes:
        calls["data"] = data
        calls["kwargs"] = kwargs
        return make_png((10, 20, 30, 128), size=(3, 3))

    monkeypatch.setattr(main, "remove_with_bria_rmbg_2", fake_bria_remove)

    client = TestClient(main.app)
    response = client.post(
        "/remove-background/?model=bria-rmbg-2.0&model_input_size=1024&device=auto&dtype=auto&return_checker_preview=true",
        files={"file": ("input.png", make_png(), "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
    kwargs = cast(dict[str, Any], calls["kwargs"])
    assert calls["data"] == make_png()
    assert kwargs == {
        "model_input_size": 1024,
        "device": "auto",
        "dtype": "auto",
        "release_cuda_cache": None,
        "cleanup_after_request": False,
    }
    with Image.open(BytesIO(response.content)) as image:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 255


def test_bria_release_cuda_cache_query_override(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_bria_remove(data: bytes, **kwargs) -> bytes:
        calls["kwargs"] = kwargs
        return make_png((10, 20, 30, 128), size=(1, 1))

    monkeypatch.setattr(main, "remove_with_bria_rmbg_2", fake_bria_remove)
    monkeypatch.setattr(
        main,
        "release_request_memory",
        lambda *, release_cuda_cache: calls.setdefault("release_cuda_cache", release_cuda_cache),
    )

    client = TestClient(main.app)
    response = client.post(
        "/remove-background/?model=bria-rmbg-2.0&release_cuda_cache=false",
        files={"file": ("input.png", make_png(), "image/png")},
    )

    assert response.status_code == 200
    kwargs = cast(dict[str, Any], calls["kwargs"])
    assert kwargs["release_cuda_cache"] is False
    assert kwargs["cleanup_after_request"] is False
    assert calls["release_cuda_cache"] is False


def test_bria_release_cuda_cache_env_default(monkeypatch) -> None:
    monkeypatch.delenv("BRIA_RELEASE_CUDA_CACHE_AFTER_REQUEST", raising=False)
    assert should_release_cuda_cache_after_request() is True
    monkeypatch.setenv("BRIA_RELEASE_CUDA_CACHE_AFTER_REQUEST", "false")
    assert should_release_cuda_cache_after_request() is False


def test_bria_remove_releases_request_memory_with_env_default(monkeypatch) -> None:
    from rembg_api import bria_rmbg

    calls: dict[str, object] = {}

    class FakeBackend:
        def remove_background(self, data: bytes, *, model_input_size: int) -> bytes:
            calls["data"] = data
            calls["model_input_size"] = model_input_size
            return b"png"

    monkeypatch.setenv("BRIA_RELEASE_CUDA_CACHE_AFTER_REQUEST", "false")
    monkeypatch.setattr(bria_rmbg, "resolve_bria_backend_cache_key", lambda path, device, dtype: (path, "cuda", "fp16"))
    monkeypatch.setattr(bria_rmbg, "get_bria_rmbg_2_backend", lambda *key: FakeBackend())
    monkeypatch.setattr(
        bria_rmbg,
        "release_request_memory",
        lambda *, release_cuda_cache: calls.setdefault("release_cuda_cache", release_cuda_cache),
    )

    assert bria_rmbg.remove_with_bria_rmbg_2(
        b"input",
        model_input_size=1024,
        device="auto",
        dtype="auto",
    ) == b"png"

    assert calls == {
        "data": b"input",
        "model_input_size": 1024,
        "release_cuda_cache": False,
    }


def test_bria_cache_key_canonicalizes_auto_to_resolved_cuda(monkeypatch) -> None:
    class FakeDevice:
        def __init__(self, value: str) -> None:
            self.type = value

    fake_torch = SimpleNamespace(
        float16=object(),
        float32=object(),
        cuda=SimpleNamespace(is_available=lambda: True),
        device=FakeDevice,
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    assert resolve_bria_backend_cache_key("/models/bria", "auto", "auto") == (
        "/models/bria",
        "cuda",
        "fp16",
    )
    assert resolve_bria_backend_cache_key("/models/bria", "cuda", "fp16") == (
        "/models/bria",
        "cuda",
        "fp16",
    )


def test_health_torch_status_includes_cuda_memory_stats(monkeypatch) -> None:
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        memory_allocated=lambda device: 11,
        memory_reserved=lambda device: 22,
        max_memory_allocated=lambda device: 33,
        max_memory_reserved=lambda device: 44,
    )
    fake_torch = SimpleNamespace(__version__="test", cuda=fake_cuda)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    status = get_torch_status()

    assert status["cuda_available"] is True
    assert status["cuda_memory"] == {
        "available": True,
        "device": 0,
        "allocated_bytes": 11,
        "reserved_bytes": 22,
        "max_allocated_bytes": 33,
        "max_reserved_bytes": 44,
    }


def test_bria_rmbg_return_alpha_reuses_alpha_post_processing(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "remove_with_bria_rmbg_2",
        lambda data, **kwargs: make_png((10, 20, 30, 128), size=(1, 1)),
    )

    client = TestClient(main.app)
    response = client.post(
        "/remove-background/?model=bria-rmbg-2.0&return_alpha=true&alpha_threshold=129",
        files={"file": ("input.png", make_png(), "image/png")},
    )

    assert response.status_code == 200
    with Image.open(BytesIO(response.content)) as image:
        assert image.mode == "L"
        assert image.getpixel((0, 0)) == 0


def test_bria_query_validation_rejects_out_of_bounds_input_size() -> None:
    client = TestClient(main.app)
    response = client.post(
        "/remove-background/?model=bria-rmbg-2.0&model_input_size=128",
        files={"file": ("input.png", make_png(), "image/png")},
    )
    assert response.status_code == 422


def test_model_enum_validation_rejects_unknown_model() -> None:
    client = TestClient(main.app)
    response = client.post(
        "/remove-background/?model=unknown",
        files={"file": ("input.png", make_png(), "image/png")},
    )
    assert response.status_code == 422


def test_local_model_status_reports_missing_and_available_paths(tmp_path: Path) -> None:
    missing = local_model_status(str(tmp_path / "missing"))
    assert missing.available is False
    assert missing.exists is False

    available = local_model_status(str(tmp_path))
    assert available.available is True
    assert available.exists is True
    assert available.is_dir is True
    assert available.readable is True


def test_bria_preprocess_normalization_matches_model_card() -> None:
    assert BRIA_RMBG_2_NORMALIZE_MEAN == (0.485, 0.456, 0.406)
    assert BRIA_RMBG_2_NORMALIZE_STD == (0.229, 0.224, 0.225)


def test_bria_runtime_dependency_check_reports_missing_timm_and_kornia(monkeypatch) -> None:
    def fake_import_module(name: str):
        if name in {"timm", "kornia"}:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return object()

    monkeypatch.setattr("rembg_api.bria_rmbg.importlib.import_module", fake_import_module)

    try:
        _check_bria_runtime_dependencies()
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected missing dependencies to raise RuntimeError")

    assert "missing=timm,kornia" in message
    assert "Rebuild the container" in message


def test_bria_required_runtime_modules_include_custom_code_dependencies() -> None:
    assert {"torch", "torchvision", "transformers", "timm", "kornia"}.issubset(
        BRIA_RMBG_2_REQUIRED_MODULES
    )


def test_bria_backend_failure_returns_generic_500_and_logs_exact_error(monkeypatch, caplog) -> None:
    def fail_bria_remove(data: bytes, **kwargs) -> bytes:
        raise FileNotFoundError("exact missing /models/briaai/RMBG-2.0")

    monkeypatch.setattr(main, "remove_with_bria_rmbg_2", fail_bria_remove)

    client = TestClient(main.app)
    response = client.post(
        "/remove-background/?model=bria-rmbg-2.0",
        files={"file": ("input.png", make_png(), "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal image processing error"
    assert "exact missing /models/briaai/RMBG-2.0" in caplog.text


def test_empty_file_returns_400() -> None:
    client = TestClient(main.app)
    response = client.post(
        "/remove-background/",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty"


def test_invalid_query_validation_returns_422() -> None:
    client = TestClient(main.app)
    response = client.post(
        "/remove-background/?alpha_matting_foreground_threshold=999",
        files={"file": ("input.png", make_png(), "image/png")},
    )
    assert response.status_code == 422
