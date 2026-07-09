# rembg-api

Thin FastAPI bytes-in/bytes-out wrapper around [`rembg`](https://github.com/danielgatis/rembg). This project is a wrapper service, not a fork of `rembg`.

## API shape

- `POST /remove-background/` accepts multipart image bytes in a `file` field.
- The response is PNG bytes (`Content-Type: image/png`).
- Normal requests do not require temporary filesystem files; input and output are handled as bytes in process.
- OpenAPI docs are available from FastAPI at `/docs` and `/openapi.json`.
- `GET /health` reports the ONNX Runtime execution providers available inside the running container/process, so you can verify CPU vs CUDA without shelling into the container.

## Run locally

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uvicorn rembg_api.main:app --host 0.0.0.0 --port 8001
```

The first request for a model may download model weights into rembg/onnxruntime's normal cache.

## Run with Docker

### CPU image (default)

`Dockerfile` installs `onnxruntime` and is the safest default when no NVIDIA runtime is available.

```bash
docker build -t rembg-api:cpu -f Dockerfile .
docker run --rm -p 8001:8001 rembg-api:cpu
```

### GPU image

`Dockerfile.gpu` installs `onnxruntime-gpu`. Use it on a host with the NVIDIA Container Toolkit installed and a compatible driver/CUDA runtime available to Docker.

```bash
docker build -t rembg-api:gpu -f Dockerfile.gpu .
docker run --rm --gpus all -p 8001:8001 rembg-api:gpu
```

## Run with Docker Compose

### CPU compose service

```bash
docker compose up --build rembg-api
```

### GPU compose service

The GPU service is behind the `gpu` profile and uses `gpus: all` so you can choose it without editing compose files.

```bash
docker compose --profile gpu up --build rembg-api-gpu
```

The service listens on <http://localhost:8001> for both CPU and GPU modes.

## Verify ONNX Runtime / CUDA provider

From the host, call `/health`:

```bash
curl -sS http://localhost:8001/health
```

CPU images should report at least `CPUExecutionProvider`:

```json
{
  "status": "ok",
  "onnxruntime_available_providers": ["CPUExecutionProvider"],
  "preferred_provider": "CPUExecutionProvider",
  "gpu_available": false
}
```

GPU images should include `CUDAExecutionProvider` when Docker/NVIDIA runtime wiring is working:

```json
{
  "status": "ok",
  "onnxruntime_available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "preferred_provider": "CUDAExecutionProvider",
  "gpu_available": true
}
```

Optional container-side check:

```bash
docker exec <container-name> python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

`/health` only imports ONNX Runtime and asks for available providers; it does not create a rembg session or download model weights.

## Endpoints

### `GET /health`

Returns service status plus ONNX Runtime provider observability:

```json
{
  "status": "ok",
  "onnxruntime_available_providers": ["CPUExecutionProvider"],
  "preferred_provider": "CPUExecutionProvider",
  "gpu_available": false
}
```

### `GET /models`

Returns the default model and supported model names.

### `POST /remove-background/`

Basic transparent PNG output:

```bash
curl -sS -X POST "http://localhost:8001/remove-background/" \
  -F "file=@input.jpg" \
  --output output.png
```

Run rembg on an already-upscaled image:

```bash
curl -sS -X POST "http://localhost:8001/remove-background/?model=isnet-general-use" \
  -F "file=@upscaled.png" \
  --output upscaled-no-bg.png
```

Choose a model and composite over white:

```bash
curl -sS -X POST "http://localhost:8001/remove-background/?model=u2net&background_color=white" \
  -F "file=@input.jpg" \
  --output output-white.png
```

Return a grayscale alpha mask preview:

```bash
curl -sS -X POST "http://localhost:8001/remove-background/?return_alpha=true" \
  -F "file=@input.jpg" \
  --output alpha.png
```

Return a checkerboard-composited PNG preview:

```bash
curl -sS -X POST "http://localhost:8001/remove-background/?return_checker_preview=true&checker_size=24" \
  -F "file=@input.jpg" \
  --output preview.png
```

## Query parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `model` | `isnet-general-use` | One of `isnet-general-use`, `u2net`, `u2netp`, `isnet-anime`, `silueta`. Sessions are cached by model in process. |
| `only_mask` | `false` | Passed through to `rembg.remove`. |
| `post_process_mask` | `false` | Passed through to `rembg.remove`. |
| `alpha_matting` | `false` | Passed through to `rembg.remove`. |
| `alpha_matting_foreground_threshold` | `240` | Integer, `0..255`. |
| `alpha_matting_background_threshold` | `10` | Integer, `0..255`. |
| `alpha_matting_erode_size` | `10` | Integer, `>=0`. |
| `output_format` | `png` | v1 supports PNG only. |
| `background_color` | `transparent` | `transparent`, `white`, `black`, or `custom`. |
| `background_hex` | `ffffff` | Six-digit RGB hex for `background_color=custom`; leading `#` is optional. |
| `alpha_blur` | `0.0` | Gaussian blur radius, `0..20`; applied after rembg. |
| `alpha_erode` | `0` | Shrinks alpha mask, `0..100`; applied after rembg. |
| `alpha_dilate` | `0` | Expands alpha mask, `0..100`; applied after rembg. |
| `alpha_threshold` | `0` | `0` disables thresholding; otherwise binarizes alpha at `1..255`. |
| `despill` | `false` | Enables simple edge despill. |
| `despill_color` | `black` | `black`, `white`, `green`, `blue`, or `custom`. |
| `despill_hex` | `000000` | Six-digit RGB hex for `despill_color=custom`; leading `#` is optional. |
| `return_alpha` | `false` | Returns grayscale alpha PNG bytes after alpha refinement/despill and before background compositing. |
| `return_checker_preview` | `false` | Returns checker-composited PNG bytes after alpha refinement/despill and before background compositing. |
| `checker_size` | `32` | Checker square size, `2..128`. |

If both `return_alpha` and `return_checker_preview` are true, `return_alpha` takes precedence.

## Runtime notes

- First request per model may download weights/cache the model before inference begins.
- It is OK to have the RealESRGAN and rembg containers both started and models loaded, but avoid simultaneous inference if they share the same GPU/VRAM budget.
- For the intended upscale-then-cutout workflow, run rembg on the already-upscaled image (for example `upscaled.png`) so the output mask aligns with the final image size.

## Errors

- Empty uploads return `400` with a JSON detail message.
- FastAPI validation errors return `422`.
- Unexpected processing failures are logged server-side with `repr(e)` context and return a generic `500` JSON detail.

## Troubleshooting

- **Slow first request:** rembg downloads model weights the first time a model is used.
- **Model download/network failures:** pre-warm the model cache in the runtime environment or ensure outbound network access during first use.
- **GPU image still reports CPU only:** confirm the host has a working NVIDIA driver, NVIDIA Container Toolkit, and the container was started with `--gpus all` or the `rembg-api-gpu` compose service.
- **OpenCV/onnxruntime library errors in containers:** the Dockerfiles install `libglib2.0-0` and `libgl1`, which are commonly required by rembg's dependency stack.
- **Large image memory use:** start with smaller inputs or `u2netp` if memory is constrained.

## Development

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[test]"
python -m compileall src tests
pytest -q
```

Tests monkeypatch `rembg.new_session`, `rembg.remove`, and ONNX Runtime provider discovery so they do not download models or require GPU packages.
