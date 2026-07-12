from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

from rembg_api import main
from rembg_api.birefnet_hr import BiRefNetConfig, DEFAULT_REVISION
from rembg_api.limits import ImageLimits
from helpers import make_png


def test_upload_content_length_is_rejected_before_read() -> None:
    request = type("Request", (), {"headers": {"content-length": "11"}})()
    with pytest.raises(Exception) as caught:
        main._reject_oversized_content_length(request, 10)
    assert caught.value.status_code == 413


def test_lazy_chunked_upload_is_rejected_at_actual_byte_limit() -> None:
    class Upload:
        def __init__(self) -> None:
            self.chunks = [b"abc", b"def", b""]
            self.calls = 0

        async def read(self, size: int) -> bytes:
            self.calls += 1
            assert size == main._UPLOAD_CHUNK_BYTES
            return self.chunks.pop(0)

    async def scenario() -> None:
        upload = Upload()
        with pytest.raises(Exception) as caught:
            await main._read_upload_limited(upload, 5)
        assert caught.value.status_code == 413
        assert upload.calls == 2

    asyncio.run(scenario())


def test_encoded_output_limit_returns_413_without_large_allocation(monkeypatch) -> None:
    monkeypatch.setattr(main, "max_upload_bytes_from_env", lambda: 1_000_000)
    monkeypatch.setattr(main, "input_limits_from_env", lambda: ImageLimits(10, 10, 100))
    monkeypatch.setattr(
        main, "output_limits_from_env", lambda: ImageLimits(10, 10, 100, 3)
    )
    monkeypatch.setattr(main, "new_session", lambda model: object())
    monkeypatch.setattr(main, "remove", lambda *args, **kwargs: make_png())
    monkeypatch.setattr(main, "process_png_bytes", lambda *args, **kwargs: b"four")

    from fastapi.testclient import TestClient

    response = TestClient(main.app).post(
        "/remove-background/?model=u2net",
        files={"file": ("in.png", make_png(), "image/png")},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "Image exceeds this service's limits"


def test_birefnet_admission_is_nonblocking_and_recovers_after_cancellation(monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    state = {"active": 0, "maximum": 0, "calls": 0}
    lock = threading.Lock()
    config = BiRefNetConfig(
        source="/models/test",
        revision=DEFAULT_REVISION,
        local_files_only=True,
        trust_remote_code=True,
        cache_dir=None,
        device="cpu",
        precision="fp32",
        inference_size=512,
        foreground_refinement=False,
        max_concurrency=1,
    )
    main._birefnet_admissions.clear()
    monkeypatch.setattr(main.BiRefNetConfig, "from_env", lambda: config)
    monkeypatch.setattr(main, "get_onnxruntime_provider_info", lambda: {})
    monkeypatch.setattr(main, "get_bria_model_info", lambda: {})
    monkeypatch.setattr(main, "birefnet_health_info", lambda: {"loaded": True})

    def slow_remove(data: bytes, **kwargs: object) -> bytes:
        with lock:
            state["calls"] += 1
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        entered.set()
        release.wait(timeout=2)
        with lock:
            state["active"] -= 1
        return make_png()

    monkeypatch.setattr(main, "remove_with_birefnet", slow_remove)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(
                client.post(
                    "/remove-background/?model=birefnet-hr-matting",
                    files={"file": ("in.png", make_png(), "image/png")},
                )
            )
            assert await asyncio.to_thread(entered.wait, 1)
            health = await asyncio.wait_for(client.get("/health"), timeout=0.25)
            assert health.status_code == 200
            saturated = await asyncio.gather(
                *[
                    client.post(
                        "/remove-background/?model=birefnet-hr-matting",
                        files={"file": ("in.png", make_png(), "image/png")},
                    )
                    for _ in range(8)
                ]
            )
            assert {response.status_code for response in saturated} == {429}
            assert state["calls"] == 1  # no decode/preprocess/model call after admission fails
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            release.set()
            for _ in range(100):
                with lock:
                    if state["active"] == 0:
                        break
                await asyncio.sleep(0.01)
            recovered = await asyncio.wait_for(
                client.post(
                    "/remove-background/?model=birefnet-hr-matting",
                    files={"file": ("in.png", make_png(), "image/png")},
                ),
                timeout=1,
            )
            assert recovered.status_code == 200

    asyncio.run(scenario())
    assert state["maximum"] == 1
    assert state["calls"] == 2
