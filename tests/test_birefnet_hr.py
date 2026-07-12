from __future__ import annotations

import threading
import time
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rembg_api import birefnet_hr, main
from rembg_api.birefnet_hr import (
    DEFAULT_REVISION,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    BiRefNetBackend,
    BiRefNetConfig,
    health_info,
    resolve_runtime,
)
from helpers import make_png


def config(tmp_path, **overrides):
    values = dict(
        source=str(tmp_path),
        revision=DEFAULT_REVISION,
        local_files_only=True,
        trust_remote_code=False,
        cache_dir=None,
        device="cpu",
        precision="fp32",
        inference_size=2048,
        foreground_refinement=False,
        max_concurrency=1,
    )
    values.update(overrides)
    return BiRefNetConfig(**values)


class FakePrediction:
    pass


def test_safe_offline_defaults(monkeypatch):
    for name in [
        "BIREFNET_MODEL_PATH",
        "BIREFNET_MODEL_ID",
        "BIREFNET_LOCAL_FILES_ONLY",
        "BIREFNET_TRUST_REMOTE_CODE",
    ]:
        monkeypatch.delenv(name, raising=False)
    cfg = BiRefNetConfig.from_env()
    assert cfg.local_files_only is True
    assert cfg.trust_remote_code is False
    assert cfg.source == "/models/ZhengPeng7/BiRefNet_HR-matting"
    assert cfg.revision == DEFAULT_REVISION
    assert cfg.inference_size == 2048


def test_online_bootstrap_requires_explicit_remote_code(monkeypatch):
    monkeypatch.setenv("BIREFNET_LOCAL_FILES_ONLY", "false")
    monkeypatch.setenv("BIREFNET_TRUST_REMOTE_CODE", "false")
    with pytest.raises(ValueError, match="online bootstrap"):
        BiRefNetConfig.from_env()


def test_cpu_never_allows_half(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_runtime("auto", "auto") == ("cpu", "fp32")
    with pytest.raises(RuntimeError, match="only on CUDA"):
        resolve_runtime("cpu", "fp16")


def test_cuda_auto_uses_fp16(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_runtime("auto", "auto") == ("cuda", "fp16")


def test_loader_receives_pinned_offline_contract(tmp_path):
    import torch

    calls = {}

    class Model:
        def to(self, **kwargs):
            calls["to"] = kwargs
            return self

        def eval(self):
            calls["eval"] = True
            return self

    def loader(source, **kwargs):
        calls["source"] = source
        calls["kwargs"] = kwargs
        return Model()

    BiRefNetBackend(config(tmp_path), "cpu", "fp32", loader=loader)
    assert calls["source"] == str(tmp_path)
    assert calls["kwargs"] == {
        "revision": DEFAULT_REVISION,
        "trust_remote_code": False,
        "local_files_only": True,
    }
    assert calls["to"]["dtype"] is torch.float32
    assert calls["eval"] is True


def test_preprocessing_prediction_and_exact_rgba_output(tmp_path):
    import torch

    seen = {}

    class Model:
        def to(self, **kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, tensor):
            seen["tensor"] = tensor.detach().cpu()
            # logits; backend must choose [-1] and sigmoid it
            return [torch.full((1, 1, 2, 2), -99.0), torch.zeros((1, 1, 2, 2))]

    backend = BiRefNetBackend(
        config(tmp_path), "cpu", "fp32", loader=lambda *a, **k: Model()
    )
    source = Image.new("RGB", (7, 5), (255, 0, 0))
    buf = BytesIO()
    source.save(buf, "PNG")
    result = backend.remove_background(
        buf.getvalue(), inference_size=512, foreground_refinement=False
    )

    tensor = seen["tensor"]
    assert tensor.shape == (1, 3, 512, 512)
    assert tensor.dtype is torch.float32
    assert tensor[0, 0, 0, 0].item() == pytest.approx(
        (1 - NORMALIZE_MEAN[0]) / NORMALIZE_STD[0]
    )
    assert tensor[0, 1, 0, 0].item() == pytest.approx(
        (0 - NORMALIZE_MEAN[1]) / NORMALIZE_STD[1]
    )
    with Image.open(BytesIO(result)) as output:
        assert output.mode == "RGBA"
        assert output.size == (7, 5)
        assert 126 <= output.getchannel("A").getextrema()[0] <= 128


def test_malformed_output_is_rejected(tmp_path):
    class Model:
        def to(self, **kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, tensor):
            return []

    backend = BiRefNetBackend(
        config(tmp_path), "cpu", "fp32", loader=lambda *a, **k: Model()
    )
    with pytest.raises(RuntimeError, match="invalid segmentation output"):
        backend.remove_background(
            make_png(), inference_size=512, foreground_refinement=False
        )


def test_invalid_image_is_rejected_before_inference(tmp_path):
    class Model:
        def to(self, **kwargs):
            return self

        def eval(self):
            return self

    backend = BiRefNetBackend(
        config(tmp_path), "cpu", "fp32", loader=lambda *a, **k: Model()
    )
    with pytest.raises(ValueError, match="valid image"):
        backend.remove_background(
            b"not an image", inference_size=512, foreground_refinement=False
        )


def test_concurrency_is_bounded(tmp_path):
    import torch

    active = 0
    maximum = 0
    lock = threading.Lock()

    class Model:
        def to(self, **kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, tensor):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return [torch.zeros((1, 1, 2, 2))]

    backend = BiRefNetBackend(
        config(tmp_path, max_concurrency=1),
        "cpu",
        "fp32",
        loader=lambda *a, **k: Model(),
    )
    threads = [
        threading.Thread(
            target=backend.remove_background,
            args=(make_png(),),
            kwargs={"inference_size": 512, "foreground_refinement": False},
        )
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert maximum == 1


def test_health_does_not_load_and_hides_local_path(tmp_path, monkeypatch):
    monkeypatch.setattr(birefnet_hr, "resolve_runtime", lambda *args: ("cpu", "fp32"))
    birefnet_hr.clear_cache()
    info = health_info(config(tmp_path, trust_remote_code=True))
    assert info["loaded"] is False
    assert info["model"] == tmp_path.name
    assert str(tmp_path) not in str(info)
    assert info["revision"] == DEFAULT_REVISION
    assert info["ready"] is True


def test_api_and_openapi_select_birefnet(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        main, "birefnet_health_info", lambda: {"loaded": False, "ready": False}
    )
    monkeypatch.setattr(
        main,
        "remove_with_birefnet",
        lambda data, **kwargs: (
            calls.update(kwargs) or make_png((1, 2, 3, 128), size=(4, 3))
        ),
    )
    client = TestClient(main.app)
    response = client.post(
        "/remove-background/?model=birefnet-hr-matting&birefnet_inference_size=2048&birefnet_foreground_refinement=true",
        files={"file": ("in.png", make_png(), "image/png")},
    )
    assert response.status_code == 200
    with Image.open(BytesIO(response.content)) as output:
        assert output.mode == "RGBA"
        assert output.size == (4, 3)
    assert calls == {"inference_size": 2048, "foreground_refinement": True}
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/remove-background/"]["post"]
    names = {parameter["name"] for parameter in operation["parameters"]}
    assert {
        "model",
        "birefnet_inference_size",
        "birefnet_foreground_refinement",
    } <= names
    model_schema = next(p for p in operation["parameters"] if p["name"] == "model")[
        "schema"
    ]
    assert "birefnet-hr-matting" in model_schema["enum"]


def test_lazy_cache_loads_once(tmp_path, monkeypatch):
    calls = []
    fake = object()
    monkeypatch.setattr(birefnet_hr, "resolve_runtime", lambda *args: ("cpu", "fp32"))
    monkeypatch.setattr(
        birefnet_hr, "BiRefNetBackend", lambda *args: calls.append(args) or fake
    )
    birefnet_hr.clear_cache()
    cfg = config(tmp_path)
    assert birefnet_hr.get_backend(cfg) is fake
    assert birefnet_hr.get_backend(cfg) is fake
    assert len(calls) == 1
