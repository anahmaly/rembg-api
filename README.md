# rembg-api

Thin FastAPI bytes-in/bytes-out wrapper around [`rembg`](https://github.com/danielgatis/rembg). This project is a wrapper service, not a fork of `rembg`. It can also run BRIA RMBG-2.0 from a pre-downloaded local model directory.

## API shape

- `POST /remove-background/` accepts multipart image bytes in a `file` field.
- The response is PNG bytes (`Content-Type: image/png`).
- Normal requests do not require temporary filesystem files; input and output are handled as bytes in process.
- OpenAPI docs are available from FastAPI at `/docs` and `/openapi.json`; `/remove-background/` documents `413` size-limit and `429` BiRefNet-capacity responses.
- Request and image limits are applied before expensive image work; the defaults are intentionally finite and configurable below.
- `GET /health` reports ONNX Runtime provider availability plus BRIA RMBG-2.0 torch/CUDA, CUDA memory stats, and local-path status without downloading weights.
- `GET /models` lists supported model IDs and whether the configured BRIA RMBG-2.0 path is present/readable.
- `POST /cache/clear` clears cached rembg, BRIA RMBG-2.0, and BiRefNet backends, then reports whether optional CUDA allocator cleanup actually ran.

## Run locally

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uvicorn rembg_api.main:app --host 0.0.0.0 --port 8001
```

The first request for a rembg model may download model weights into rembg/onnxruntime's normal cache. BRIA RMBG-2.0 never downloads from Hugging Face at request time; it loads from the configured local path.

## BRIA RMBG-2.0 local model setup

BRIA RMBG-2.0 is a gated Hugging Face model with non-commercial terms unless you have a separate license. This service does not fetch gated weights or accept an HF token in the container. Download or otherwise place the licensed model files on the host first, then mount that exact directory read-only into the container.

Expected host path:

```bash
~/models/briaai/RMBG-2.0
```

Container path, configurable with `BRIA_RMBG_2_MODEL_PATH` and defaulting to this exact value:

```bash
/models/briaai/RMBG-2.0
```

## Run with Docker

### CPU image

`Dockerfile.cpu` installs the Python dependencies from `pyproject.toml`, including the torch/transformers/timm/kornia stack used by BRIA RMBG-2.0. It is the safest choice when no NVIDIA runtime is available; BRIA inference will be much slower on CPU.

There is intentionally no default `Dockerfile` in this repository. Use `-f Dockerfile.cpu` or `-f Dockerfile.gpu` so the selected runtime is explicit; plain `docker build .` is not documented or supported for choosing CPU vs GPU.

```bash
docker build -t rembg-api:cpu -f Dockerfile.cpu .
docker run --rm -p 8001:8001 \
  -v ~/models/briaai/RMBG-2.0:/models/briaai/RMBG-2.0:ro \
  rembg-api:cpu
```

### GPU image

`Dockerfile.gpu` uses an NVIDIA CUDA 13 + cuDNN runtime base image, installs CUDA PyTorch/torchvision, and installs a pinned `onnxruntime-gpu` wheel that expects CUDA runtime libraries such as `libcudart.so.13` to be present inside the container. Use it on a host with a compatible NVIDIA driver and the NVIDIA Container Toolkit installed.

```bash
docker build -t rembg-api:gpu -f Dockerfile.gpu .
docker run --rm --gpus all -p 8001:8001 \
  -v ~/models/briaai/RMBG-2.0:/models/briaai/RMBG-2.0:ro \
  rembg-api:gpu
curl -sS http://localhost:8001/health
```

## Run with Docker Compose

### CPU compose service

The CPU service builds from `Dockerfile.cpu` explicitly and mounts `${HOME}/models/briaai/RMBG-2.0` to `/models/briaai/RMBG-2.0:ro`.

```bash
docker compose up --build rembg-api
```

### Dedicated GPU compose file

`compose.gpu.yml` is the convenience path for GPU runs. It builds from `Dockerfile.gpu`, requests `gpus: all`, exposes `8001:8001`, mounts the BRIA RMBG-2.0 model read-only from `${HOME}/models/briaai/RMBG-2.0`, and keeps rembg model downloads in the named `rembg-model-cache` volume.

```bash
docker compose -f compose.gpu.yml up -d --build
docker compose -f compose.gpu.yml logs -f
curl -sS http://localhost:8001/health
docker compose -f compose.gpu.yml down
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
  "gpu_available": false,
  "bria_rmbg_2": {
    "model_path": "/models/briaai/RMBG-2.0",
    "model_path_available": true,
    "torch_available": true,
    "cuda_available": false
  }
}
```

GPU images should include `CUDAExecutionProvider` when Docker/NVIDIA runtime wiring is working, and `bria_rmbg_2.cuda_available` should be `true` when torch can see CUDA:

```json
{
  "status": "ok",
  "onnxruntime_available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "preferred_provider": "CUDAExecutionProvider",
  "gpu_available": true,
  "bria_rmbg_2": {
    "model_path": "/models/briaai/RMBG-2.0",
    "model_path_available": true,
    "torch_available": true,
    "cuda_available": true
  }
}
```

Optional container-side check:

```bash
docker exec <container-name> python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

When torch can see CUDA, `/health` also includes `bria_rmbg_2.cuda_memory` with integer byte counters for `allocated_bytes`, `reserved_bytes`, `max_allocated_bytes`, and `max_reserved_bytes` on the current CUDA device. These fields are best-effort observability and are omitted behind an `available: false` marker if torch/CUDA memory stats are unavailable.

`/health` imports ONNX Runtime and checks torch/local-path/CUDA memory availability; it does not create a rembg session, load BRIA RMBG-2.0, or download model weights.

## Endpoints

### `GET /health`

Returns service status plus ONNX Runtime provider observability and BRIA RMBG-2.0 local model status.

### `GET /models`

Returns the default model, supported model names, and BRIA RMBG-2.0 configured local-path availability.

### `POST /cache/clear`

Clears cached rembg sessions plus BRIA RMBG-2.0 and BiRefNet backends, then runs `gc.collect()`. With `release_cuda_cache=true` (the default), `torch.cuda.empty_cache()` runs only when a torch backend was loaded and CUDA is available; clearing an unused service does not import or load torch/models. The response reports both `cuda_cache_release_requested` and `cuda_cache_released`. The service has no built-in auth, so keep it LAN-local or behind your own trusted network boundary.

```bash
curl -sS -X POST "http://localhost:8001/cache/clear"
curl -sS -X POST "http://localhost:8001/cache/clear?release_cuda_cache=false"
```

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

Run BRIA RMBG-2.0 from the mounted local model path against a large PNG, for example a 10k PNG:

```bash
curl -sS -X POST "http://localhost:8001/remove-background/?model=bria-rmbg-2.0&model_input_size=1024&device=auto&dtype=auto" \
  -F "file=@10k.png" \
  --output 10k-bria-rmbg-2.png
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
| `model` | `isnet-general-use` | One of `isnet-general-use`, `u2net`, `u2netp`, `isnet-anime`, `silueta`, `bria-rmbg-2.0`. rembg sessions are cached by model; BRIA backend is cached by canonical local path/device/dtype after resolving `auto`. |
| `model_input_size` | `1024` | BRIA RMBG-2.0 square input size, `512..2048`. It affects preprocessing only and does not reload the model. |
| `device` | `auto` | BRIA RMBG-2.0 only: `auto`, `cuda`, or `cpu`. `auto` uses CUDA when torch sees it. |
| `dtype` | `auto` | BRIA RMBG-2.0 only: `auto`, `fp16`, or `fp32`. `auto` uses fp16 on CUDA and fp32 on CPU. |
| `release_cuda_cache` | env/default | BRIA RMBG-2.0 only: overrides `BRIA_RELEASE_CUDA_CACHE_AFTER_REQUEST` for this request. When true, the service runs `gc.collect()` and `torch.cuda.empty_cache()` after the request. The env var defaults to `true` when unset. |
| `only_mask` | `false` | Passed through to `rembg.remove`; ignored for `bria-rmbg-2.0`. |
| `post_process_mask` | `false` | Passed through to `rembg.remove`; ignored for `bria-rmbg-2.0`. |
| `alpha_matting` | `false` | Passed through to `rembg.remove`; ignored for `bria-rmbg-2.0`. |
| `alpha_matting_foreground_threshold` | `240` | Integer, `0..255`; rembg only. |
| `alpha_matting_background_threshold` | `10` | Integer, `0..255`; rembg only. |
| `alpha_matting_erode_size` | `10` | Integer, `>=0`; rembg only. |
| `output_format` | `png` | v1 supports PNG only. |
| `background_color` | `transparent` | `transparent`, `white`, `black`, or `custom`. |
| `background_hex` | `ffffff` | Six-digit RGB hex for `background_color=custom`; leading `#` is optional. |
| `alpha_blur` | `0.0` | Gaussian blur radius, `0..20`; applied after background removal. |
| `alpha_erode` | `0` | Shrinks alpha mask, `0..100`; applied after background removal. |
| `alpha_dilate` | `0` | Expands alpha mask, `0..100`; applied after background removal. |
| `alpha_threshold` | `0` | `0` disables thresholding; otherwise binarizes alpha at `1..255`. |
| `despill` | `false` | Enables simple edge despill. |
| `despill_color` | `black` | `black`, `white`, `green`, `blue`, or `custom`. |
| `despill_hex` | `000000` | Six-digit RGB hex for `despill_color=custom`; leading `#` is optional. |
| `return_alpha` | `false` | Returns grayscale alpha PNG bytes after alpha refinement/despill and before background compositing. |
| `return_checker_preview` | `false` | Returns checker-composited PNG bytes after alpha refinement/despill and before background compositing. |
| `checker_size` | `32` | Checker square size, `2..128`. |

If both `return_alpha` and `return_checker_preview` are true, `return_alpha` takes precedence.

## Request and image limits

The service applies these process-wide bounds to every `/remove-background/` request. A route-scoped ASGI guard rejects a declared multipart `Content-Length` over `REMBG_MAX_REQUEST_BYTES` before consuming the body and counts every actual `http.request` chunk while Starlette parses/spools multipart data. The whole envelope is therefore bounded even without a length header, including framing, ignored form parts, and the selected file; parsing may occur before BiRefNet admission, but only inside this finite request-byte bound. Malformed or negative declared lengths return `400`. This is deliberately separate from `REMBG_MAX_UPLOAD_BYTES`: multipart framing means total request bytes are not the same as the selected file's bytes. After parsing, BiRefNet saturation is rejected before the selected upload is read or decoded. The selected file is always read in 64 KiB chunks and stops at the exact file limit. Image dimensions are checked before pixel decode, corrupt/truncated client images return a safe `400`, and output dimensions are checked before BiRefNet alpha resizing/RGBA allocation and again before postprocessing. Final PNG encoding writes through a capped in-memory stream that raises `413` before its buffer can grow beyond `REMBG_MAX_OUTPUT_BYTES`.

| Variable | Default | Scope |
| --- | --- | --- |
| `REMBG_MAX_REQUEST_BYTES` | `21000000` | Whole HTTP request body, including multipart framing. Must be at least `REMBG_MAX_UPLOAD_BYTES`; the default leaves 1 MB for normal multipart overhead. |
| `REMBG_MAX_UPLOAD_BYTES` | `20000000` | Input file bytes read from multipart upload. |
| `REMBG_MAX_INPUT_WIDTH` | `10000` | Decoded source-image width. |
| `REMBG_MAX_INPUT_HEIGHT` | `10000` | Decoded source-image height. |
| `REMBG_MAX_INPUT_PIXELS` | `40000000` | Decoded source-image pixels. Pillow decompression-bomb warnings/errors are treated as a safe rejection. |
| `REMBG_MAX_OUTPUT_WIDTH` | `10000` | Decoded model-output width. |
| `REMBG_MAX_OUTPUT_HEIGHT` | `10000` | Decoded model-output height. |
| `REMBG_MAX_OUTPUT_PIXELS` | `40000000` | Decoded model-output pixels. |
| `REMBG_MAX_OUTPUT_BYTES` | `40000000` | Final encoded PNG response bytes. |

Limit rejections use `413` and a generic message; invalid request bodies continue to use normal `400`/`422` responses. For BiRefNet, admission has no wait queue: a request that cannot immediately reserve capacity returns `429`, so queued or cancelled clients cannot accumulate background workers.

## Runtime notes

- First request per rembg model may download weights/cache the model before inference begins.
- First request for `bria-rmbg-2.0` loads BRIA RMBG-2.0 from `BRIA_RMBG_2_MODEL_PATH`; if the mounted path is missing or unreadable, the API logs the exact path/status and returns a generic `500` response.
- BRIA RMBG-2.0 cache keys are canonicalized after resolving `device=auto` and `dtype=auto`, so equivalent requests such as `device=auto&dtype=auto` and `device=cuda&dtype=fp16` reuse one CUDA fp16 backend instead of loading duplicate model copies.
- After each BRIA request the service releases request-local tensors, runs `gc.collect()`, and by default calls `torch.cuda.empty_cache()`. Set `BRIA_RELEASE_CUDA_CACHE_AFTER_REQUEST=false` globally or pass `release_cuda_cache=false` on a request to leave PyTorch's CUDA allocator cache reserved.
- BRIA RMBG-2.0 uses the model-card preprocessing normalization (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`) and requires its custom-code dependencies (`torch`, `torchvision`, `transformers`, `timm`, and `kornia`) in the image.
- It is OK to have the RealESRGAN and rembg containers both started and models loaded, but avoid simultaneous inference if they share the same GPU/VRAM budget.
- For the intended upscale-then-cutout workflow, run background removal on the already-upscaled image (for example `upscaled.png`) so the output mask aligns with the final image size.

## Errors

- Empty uploads return `400` with a JSON detail message.
- FastAPI validation errors return `422`.
- Unexpected processing failures are logged server-side with `repr(e)` context and return a generic `500` JSON detail.

## Troubleshooting

- **Slow first request:** rembg downloads model weights the first time a rembg model is used; BRIA RMBG-2.0 loads local model files the first time `model=bria-rmbg-2.0` is used for a path/device/dtype tuple.
- **BRIA RMBG-2.0 request immediately returns generic `500`:** check container logs for the server-side error. If logs mention missing `timm` or `kornia`, pull this hotfix and rebuild the image (`docker build ... --no-cache` or `docker compose --profile gpu up --build rembg-api-gpu`) so pyproject dependencies are reinstalled.
- **BRIA RMBG-2.0 path unavailable:** confirm `~/models/briaai/RMBG-2.0` exists on the host and is mounted exactly as `/models/briaai/RMBG-2.0:ro`. Check `/health` or `/models` for `model_path_available`.
- **Model download/network failures:** pre-warm the rembg model cache in the runtime environment or ensure outbound network access during first use. BRIA RMBG-2.0 is intentionally local-path only.
- **GPU image fails at startup with `ImportError: libcudart.so.13`:** rebuild from the current `Dockerfile.gpu`. The GPU image intentionally uses an NVIDIA CUDA 13 + cuDNN runtime base so the CUDA runtime shared libraries required by the pinned `onnxruntime-gpu` wheel are present in the container instead of depending on host filesystem libraries.
- **GPU image still reports CPU only:** confirm the host has a working NVIDIA driver, NVIDIA Container Toolkit, and the container was started with `--gpus all` or the `rembg-api-gpu` compose service. Use `curl -sS http://localhost:8001/health` and look for `CUDAExecutionProvider` plus `bria_rmbg_2.cuda_available`.
- **BRIA RMBG-2.0 VRAM appears to grow:** PyTorch keeps a CUDA caching allocator, so `nvidia-smi` can show reserved VRAM even after tensors are freed. This is different from live allocated tensor memory. Check `curl -sS http://localhost:8001/health` and compare `bria_rmbg_2.cuda_memory.allocated_bytes` versus `reserved_bytes`; `reserved_bytes` may stay high by design. The service now canonicalizes BRIA cache keys to avoid duplicate model loads for equivalent `auto`/explicit device and dtype settings, and defaults to emptying the CUDA cache after each BRIA request.
- **Manually recover GPU memory:** for this LAN-local unauthenticated service, call `curl -sS -X POST http://localhost:8001/cache/clear` to clear cached rembg/BRIA backends and release CUDA cache. Watch host VRAM with `watch -n 1 nvidia-smi` before and after requests/cache clears. If you intentionally want PyTorch to keep allocator cache warm between BRIA requests, set `BRIA_RELEASE_CUDA_CACHE_AFTER_REQUEST=false` or pass `release_cuda_cache=false`.
- **OpenCV/onnxruntime library errors in containers:** the Dockerfiles install `libglib2.0-0` and `libgl1`, which are commonly required by rembg's dependency stack.
- **Large image memory use:** start with `model_input_size=1024`, lower it to `512` for BRIA RMBG-2.0 if memory is constrained, or try `u2netp` for the rembg backend.

## Development

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[test]"
python -m compileall src tests
pytest -q
git diff --check
```

Tests monkeypatch `rembg.new_session`, `rembg.remove`, ONNX Runtime provider discovery, and the BRIA/BiRefNet local backends so they do not download models or require GPU/model weights.

## BiRefNet HR matting

Select `model=birefnet-hr-matting` to use [ZhengPeng7/BiRefNet_HR-matting](https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting) (MIT). The service pins revision `5d6b6f8adcb5b417c871b1d84ceaae9871355b7f`. Native preprocessing is RGB, square resize (2048 by default), tensor conversion, and ImageNet normalization. The final `model(tensor)[-1].sigmoid()` alpha is clamped, validated, and resized to the exact original dimensions before an RGBA PNG is emitted.

### Recommended offline production setup

Download the pinned snapshot outside the service image (this is an explicit operator/bootstrap step, not an application startup action):

```bash
huggingface-cli download ZhengPeng7/BiRefNet_HR-matting \
  --revision 5d6b6f8adcb5b417c871b1d84ceaae9871355b7f \
  --local-dir "$HOME/models/ZhengPeng7/BiRefNet_HR-matting"
```

Review the pinned remote Python code before enabling it, then mount it read-only. The compose files provide the exact mount and offline variables. `trust_remote_code` is required by this model even from a local snapshot, so the application default is deliberately `false`; compose opts in for the reviewed pinned mount.

```bash
export BIREFNET_MODEL_PATH=/models/ZhengPeng7/BiRefNet_HR-matting
export BIREFNET_REVISION=5d6b6f8adcb5b417c871b1d84ceaae9871355b7f
export BIREFNET_LOCAL_FILES_ONLY=true
export BIREFNET_TRUST_REMOTE_CODE=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

docker compose up --build rembg-api                    # CPU / fp32
docker compose -f compose.gpu.yml up --build           # CUDA / fp16
```

The default cache key includes effective source, revision, offline/trust policy, cache directory, resolved device, resolved precision, and concurrency. Loading is lazy and concurrency-safe; no model is loaded by startup, `/health`, or `/models`. GPU inference defaults to one concurrent call to avoid overlapping roughly 444 MB of weights plus provisional multi-GB 2048px activation/working memory. Actual VRAM and image quality are not guaranteed until evaluated on the target GPU; lower `BIREFNET_INFERENCE_SIZE` if memory is constrained.

BiRefNet admission is process-wide and immediate: `BIREFNET_MAX_CONCURRENCY` reserves a slot before endpoint file reading and holds it through worker-thread model loading, header decode, preprocessing, inference, postprocessing, and encoding. There is deliberately no admission wait queue; a saturated request receives seller-safe `429` and does no BiRefNet preprocessing. If a client disconnects or its request task is cancelled after admission, response waiting stops promptly while the already-admitted worker keeps its reservation until it genuinely exits. This bounded task handoff prevents cancellation storms from accumulating abandoned workers; `/health` remains event-loop responsive and capacity recovers when the worker finishes.

```bash
curl -sS -X POST \
  'http://localhost:8001/remove-background/?model=birefnet-hr-matting&birefnet_inference_size=2048' \
  -F 'file=@input.png' --output output.png
```

`birefnet_foreground_refinement=false` (default) preserves the original RGB and changes only alpha. When true, the conservative refinement clears hidden RGB only where alpha is fully transparent; it does not alter alpha or visible foreground colors. `/health` reports only the configured source basename, pinned revision, effective device/precision, CUDA availability, readiness, and loaded state—never the full local path and never by loading/downloading weights.

Configuration variables:

| Variable | Safe default | Meaning |
| --- | --- | --- |
| `BIREFNET_MODEL_PATH` | `/models/ZhengPeng7/BiRefNet_HR-matting` | Absolute read-only offline snapshot path. |
| `BIREFNET_MODEL_ID` | `ZhengPeng7/BiRefNet_HR-matting` | Used only when offline mode is explicitly disabled. |
| `BIREFNET_REVISION` | pinned SHA above | Model/custom-code revision. |
| `BIREFNET_LOCAL_FILES_ONLY` | `true` | Prevent Hugging Face network access. |
| `BIREFNET_TRUST_REMOTE_CODE` | `false` | Must be explicitly enabled after reviewing pinned code. |
| `BIREFNET_DEVICE` | `auto` | `auto`, `cuda`, or `cpu`. |
| `BIREFNET_PRECISION` | `auto` | fp16 on CUDA, fp32 on CPU; CPU fp16 is rejected. |
| `BIREFNET_INFERENCE_SIZE` | `2048` | Square preprocessing size, 512–4096. |
| `BIREFNET_FOREGROUND_REFINEMENT` | `false` | Optional hidden-RGB cleanup described above. |
| `BIREFNET_CACHE_DIR` | unset | Optional Hugging Face cache location. |
| `BIREFNET_MAX_CONCURRENCY` | `1` | Per-loaded-backend bounded inference concurrency. |

Online bootstrap is intentionally opt-in and executes downloaded custom code: set `BIREFNET_LOCAL_FILES_ONLY=false`, `BIREFNET_TRUST_REMOTE_CODE=true`, and optionally `BIREFNET_MODEL_ID`. Do not use that mode for drift-resistant production. If readiness is false, verify the mount, permissions, exact revision snapshot, and trust flag. A generic API 500 protects operator details; inspect server logs for the safe failure category.
