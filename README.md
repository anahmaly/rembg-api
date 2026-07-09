# rembg-api

Thin FastAPI bytes-in/bytes-out wrapper around [`rembg`](https://github.com/danielgatis/rembg). This project is a wrapper service, not a fork of `rembg`.

## API shape

- `POST /remove-background/` accepts multipart image bytes in a `file` field.
- The response is PNG bytes (`Content-Type: image/png`).
- Normal requests do not require temporary filesystem files; input and output are handled as bytes in process.
- OpenAPI docs are available from FastAPI at `/docs` and `/openapi.json`.

## Run locally

```bash
uv venv
uv pip install -e .
uvicorn rembg_api.main:app --host 0.0.0.0 --port 8001
```

The first request for a model may download model weights into rembg/onnxruntime's normal cache.

## Run with Docker Compose

```bash
docker compose up --build
```

The service listens on <http://localhost:8001>.

## Endpoints

### `GET /health`

Returns a simple health payload:

```json
{"status":"ok"}
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

## Errors

- Empty uploads return `400` with a JSON detail message.
- FastAPI validation errors return `422`.
- Unexpected processing failures are logged server-side with `repr(e)` context and return a generic `500` JSON detail.

## Troubleshooting

- **Slow first request:** rembg downloads model weights the first time a model is used.
- **Model download/network failures:** pre-warm the model cache in the runtime environment or ensure outbound network access during first use.
- **OpenCV/onnxruntime library errors in containers:** the Dockerfile installs `libglib2.0-0` and `libgl1`, which are commonly required by rembg's dependency stack.
- **Large image memory use:** start with smaller inputs or `u2netp` if memory is constrained.

## Development

```bash
uv venv
uv pip install -e ".[test]"
python -m compileall src tests
pytest -q
```

Tests monkeypatch `rembg.new_session` and `rembg.remove` so they do not download models.
