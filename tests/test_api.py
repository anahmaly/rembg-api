from __future__ import annotations

from typing import Any, cast

from fastapi.testclient import TestClient

from rembg_api import main

from helpers import make_png


def test_health() -> None:
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_lists_supported_default() -> None:
    client = TestClient(main.app)
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert body["default"] == "isnet-general-use"
    assert "u2net" in body["supported"]


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
