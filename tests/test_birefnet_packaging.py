from pathlib import Path

ROOT = Path(__file__).parents[1]
REVISION = "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f"
MOUNT = "${HOME}/models/ZhengPeng7/BiRefNet_HR-matting:/models/ZhengPeng7/BiRefNet_HR-matting:ro"


def test_compose_offline_read_only_contract():
    for filename in ("compose.yml", "compose.gpu.yml"):
        text = (ROOT / filename).read_text()
        assert MOUNT in text
        assert f"BIREFNET_REVISION={REVISION}" in text
        assert "BIREFNET_LOCAL_FILES_ONLY=true" in text
        assert "BIREFNET_TRUST_REMOTE_CODE=true" in text
        assert "HF_HUB_OFFLINE=1" in text
        assert "TRANSFORMERS_OFFLINE=1" in text


def test_docker_images_default_to_offline():
    for filename in ("Dockerfile.cpu", "Dockerfile.gpu"):
        text = (ROOT / filename).read_text()
        assert "BIREFNET_LOCAL_FILES_ONLY=true" in text
        assert "HF_HUB_OFFLINE=1" in text
        assert "TRANSFORMERS_OFFLINE=1" in text
