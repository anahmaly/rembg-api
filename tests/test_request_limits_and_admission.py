from __future__ import annotations

import asyncio
import threading
from io import BytesIO
import json

import httpx
import pytest
from PIL import Image

from rembg_api import birefnet_hr, main
from rembg_api.birefnet_hr import BiRefNetConfig, DEFAULT_REVISION
from rembg_api.limits import ImageLimits
from helpers import make_png


def _multipart_body(*parts: tuple[str, str, bytes, str]) -> tuple[bytes, str]:
    boundary = "actual-stream-boundary"
    chunks: list[bytes] = []
    for name, filename, content, content_type in parts:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


async def _stream_asgi_request(
    body: bytes, boundary: str, *, content_length: int | None = None
) -> tuple[int, dict[str, object], int]:
    headers = [(b"content-type", f"multipart/form-data; boundary={boundary}".encode())]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/remove-background/",
        "raw_path": b"/remove-background/",
        "query_string": b"model=u2net",
        "headers": headers,
        "client": ("test", 1),
        "server": ("test", 80),
    }
    midpoint = max(1, len(body) // 2)
    messages = [
        {"type": "http.request", "body": body[:midpoint], "more_body": True},
        {"type": "http.request", "body": body[midpoint:], "more_body": False},
    ]
    reads = 0
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        nonlocal reads
        reads += 1
        return messages.pop(0)

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await main.app(scope, receive, send)
    start = next(message for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    status = int(start["status"])
    parsed = json.loads(response_body) if status != 200 else {}
    return status, parsed, reads


@pytest.mark.parametrize(
    "parts",
    [
        (("ignored", "large.bin", b"x" * 500, "application/octet-stream"),),
        (("file", "large.png", b"x" * 500, "image/png"),),
    ],
)
def test_streamed_whole_multipart_limit_rejects_all_parts_before_route(
    monkeypatch, parts
) -> None:
    body, boundary = _multipart_body(*parts)
    monkeypatch.setattr(main, "max_request_bytes_from_env", lambda: len(body) - 1)
    monkeypatch.setattr(main, "get_session", lambda model: pytest.fail("route called"))

    status, detail, reads = asyncio.run(_stream_asgi_request(body, boundary))

    assert status == 413
    assert detail["detail"] == "Request body is larger than this service accepts"
    assert reads == 2


def test_declared_whole_request_limit_rejects_without_consuming_body(monkeypatch) -> None:
    body, boundary = _multipart_body(("file", "in.png", make_png(), "image/png"))
    monkeypatch.setattr(main, "max_request_bytes_from_env", lambda: len(body) - 1)
    monkeypatch.setattr(main, "get_session", lambda model: pytest.fail("route called"))

    status, detail, reads = asyncio.run(
        _stream_asgi_request(body, boundary, content_length=len(body))
    )

    assert status == 413
    assert detail["detail"] == "Request body is larger than this service accepts"
    assert reads == 0


def test_streamed_under_limit_multipart_reaches_backend(monkeypatch) -> None:
    body, boundary = _multipart_body(("file", "in.png", make_png(), "image/png"))
    calls = 0
    monkeypatch.setattr(main, "max_request_bytes_from_env", lambda: len(body))
    monkeypatch.setattr(main, "max_upload_bytes_from_env", lambda: len(body))
    monkeypatch.setattr(main, "new_session", lambda model: object())

    def fake_remove(*args, **kwargs):
        nonlocal calls
        calls += 1
        return make_png()

    monkeypatch.setattr(main, "remove", fake_remove)
    main.get_session.cache_clear()

    status, _, reads = asyncio.run(_stream_asgi_request(body, boundary))

    assert status == 200
    assert calls == 1
    assert reads == 2


def test_multipart_content_length_over_file_limit_does_not_reject_near_limit_file(
    monkeypatch,
) -> None:
    payload = make_png()
    monkeypatch.setattr(main, "max_upload_bytes_from_env", lambda: len(payload))
    monkeypatch.setattr(main, "remove", lambda *args, **kwargs: make_png())
    monkeypatch.setattr(main, "new_session", lambda model: object())

    from fastapi.testclient import TestClient

    response = TestClient(main.app).post(
        "/remove-background/?model=u2net",
        files={"file": ("in.png", payload, "image/png")},
    )

    assert response.status_code == 200


def test_oversized_declared_request_length_rejects_before_upload_read(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        main, "max_request_bytes_from_env", lambda: 100, raising=False
    )

    async def fail_if_read(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("oversized declared request must not read the upload")

    monkeypatch.setattr(main, "_read_upload_limited", fail_if_read)

    from fastapi.testclient import TestClient

    response = TestClient(main.app).post(
        "/remove-background/?model=u2net",
        files={"file": ("in.png", make_png(), "image/png")},
        headers={"Content-Length": "101"},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body is larger than this service accepts"


def test_oversized_multipart_upload_returns_413_at_route_level(monkeypatch) -> None:
    monkeypatch.setattr(main, "max_upload_bytes_from_env", lambda: 10)

    from fastapi.testclient import TestClient

    response = TestClient(main.app).post(
        "/remove-background/?model=u2net",
        files={"file": ("in.png", b"x" * 11, "image/png")},
    )

    assert response.status_code == 413


def test_birefnet_oversized_headers_return_413_before_backend_load(monkeypatch) -> None:
    output = BytesIO()
    Image.new("RGB", (10_001, 1)).save(output, "PNG")
    payload = output.getvalue()
    assert len(payload) < 1_000
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
    calls = 0
    main._birefnet_admissions.clear()
    monkeypatch.setattr(main.BiRefNetConfig, "from_env", lambda: config)
    monkeypatch.setattr(main, "input_limits_from_env", lambda: ImageLimits(10_000, 10_000, 40_000_000))

    def fail_if_loaded(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("BiRefNet backend must not load for rejected headers")

    monkeypatch.setattr(birefnet_hr, "get_backend", fail_if_loaded)

    from fastapi.testclient import TestClient

    response = TestClient(main.app).post(
        "/remove-background/?model=birefnet-hr-matting",
        files={"file": ("oversized.png", bytes(payload), "image/png")},
    )

    assert response.status_code == 413
    assert calls == 0


def test_birefnet_decompression_bomb_returns_413_before_backend_load(monkeypatch) -> None:
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
    monkeypatch.setattr(birefnet_hr, "get_backend", lambda *args: pytest.fail("backend loaded"))
    monkeypatch.setattr("PIL.Image.MAX_IMAGE_PIXELS", 1)

    from fastapi.testclient import TestClient

    response = TestClient(main.app).post(
        "/remove-background/?model=birefnet-hr-matting",
        files={"file": ("bomb.png", make_png(), "image/png")},
    )

    assert response.status_code == 413


@pytest.mark.parametrize("kind", ["not-image", "truncated-jpeg"])
def test_malformed_client_image_returns_safe_400_before_backend(
    monkeypatch, kind
) -> None:
    if kind == "not-image":
        payload = b"not image"
    else:
        output = BytesIO()
        Image.new("RGB", (16, 16), (1, 2, 3)).save(output, "JPEG")
        payload = output.getvalue()[:-20]
    monkeypatch.setattr(main, "new_session", lambda model: pytest.fail("backend loaded"))
    main.get_session.cache_clear()

    from fastapi.testclient import TestClient

    response = TestClient(main.app).post(
        "/remove-background/?model=u2net",
        files={"file": ("bad.img", payload, "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file is not a valid image"}


def test_truncated_png_returns_safe_400_before_backend(monkeypatch) -> None:
    payload = make_png()[:-10]
    monkeypatch.setattr(main, "new_session", lambda model: pytest.fail("backend loaded"))
    main.get_session.cache_clear()

    from fastapi.testclient import TestClient

    response = TestClient(main.app).post(
        "/remove-background/?model=u2net",
        files={"file": ("bad.png", payload, "image/png")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file is not a valid image"}


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
