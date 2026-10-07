# Basic Pitch — Local Docker Deployment

A slim, self-contained way to run Basic Pitch locally with Docker. The image
uses the lightweight **ONNX** runtime (no full TensorFlow), so it stays a few
hundred MB, and the ~2 MB model ships inside the `basic-pitch` package — no
separate download.

Two ways to use it:

- **HTTP service** — a small [FastAPI](https://fastapi.tiangolo.com/) app
  (`server.py`) that loads the model once and transcribes uploaded audio to MIDI.
- **CLI** — run the bundled `basic-pitch` command against mounted files.

---

## HTTP service

### Run with Docker Compose

```bash
docker compose -f deploy/docker-compose.yml up --build
```

### Or with plain Docker

```bash
# Build from the repo root (note the -f path and the "." context):
docker build -f deploy/Dockerfile -t basic-pitch-api .
docker run --rm -p 8000:8000 basic-pitch-api
```

The service listens on `http://localhost:8000`.

### Endpoints

| Method | Path          | Description                              |
|--------|---------------|------------------------------------------|
| `GET`  | `/health`     | Liveness/readiness probe.                |
| `POST` | `/transcribe` | Upload audio, get MIDI (or note JSON).   |
| `GET`  | `/docs`       | Interactive Swagger UI (auto-generated). |

### Examples

Transcribe to a MIDI file:

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@my_song.wav" \
  -o my_song.mid
```

Get note events as JSON instead:

```bash
curl -X POST "http://localhost:8000/transcribe?response_format=json" \
  -F "file=@my_song.wav"
```

Tune the prediction (all optional query params):

```bash
curl -X POST "http://localhost:8000/transcribe?onset_threshold=0.6&frame_threshold=0.3&minimum_note_length=100&multiple_pitch_bends=true" \
  -F "file=@my_song.wav" -o my_song.mid
```

| Query param            | Default  | Meaning                                             |
|------------------------|----------|-----------------------------------------------------|
| `response_format`      | `midi`   | `midi` (file) or `json` (note events).              |
| `onset_threshold`      | `0.5`    | Min likelihood for a note onset (0–1).              |
| `frame_threshold`      | `0.3`    | Min likelihood for a note to sustain (0–1).         |
| `minimum_note_length`  | `127.70` | Minimum note length, in milliseconds.               |
| `minimum_frequency`    | none     | Min allowed note frequency, in Hz.                  |
| `maximum_frequency`    | none     | Max allowed note frequency, in Hz.                  |
| `multiple_pitch_bends` | `false`  | Allow overlapping notes to have independent bends.  |
| `melodia_trick`        | `true`   | Apply the melodia post-processing step.             |

### Configuration (environment variables)

| Variable                       | Default | Description                                |
|--------------------------------|---------|--------------------------------------------|
| `BASIC_PITCH_MODEL`            | `onnx`  | Model serialization: `tf`/`coreml`/`tflite`/`onnx`. |
| `BASIC_PITCH_MAX_UPLOAD_BYTES` | `52428800` (50 MB) | Reject larger uploads with HTTP 413. |
| `PORT`                         | `8000`  | Port uvicorn binds inside the container.   |

---

## CLI usage

The same image includes the `basic-pitch` CLI. Mount an input folder and an
output folder:

```bash
docker build -f deploy/Dockerfile -t basic-pitch-api .

docker run --rm \
  -v "$PWD/audio:/in" \
  -v "$PWD/out:/out" \
  --entrypoint basic-pitch \
  basic-pitch-api /out /in/my_song.wav
```

This writes `/out/my_song_basic_pitch.mid`.

---

## Notes

- **Why Python 3.10.** The image is based on `python:3.10-slim` on purpose. On
  Python **3.11+**, `tensorflow` is a *base* dependency of `basic-pitch` (see
  `pyproject.toml`), which would bloat the image to multiple GB. On **3.10** the
  base dependency is the small `tflite-runtime`, and the `[onnx]` extra adds
  `onnxruntime` — so the image ships ONNX + TFLite runtimes and **no
  TensorFlow**.
- **Model format.** The image installs `basic-pitch[onnx]`, so inference uses
  the bundled `nmp.onnx`. Set `BASIC_PITCH_MODEL` to switch formats — `tflite`
  also works out of the box (tflite-runtime is present); `tf`/`coreml` would
  require installing the matching extra.
- **`.dockerignore`.** The repo's root `.dockerignore` excludes `saved_models`
  (intended for the training image). `deploy/Dockerfile.dockerignore` overrides
  that so the model is included — build with BuildKit (the default in modern
  Docker) for this override to take effect.
- **setuptools.** The Dockerfile upgrades `pip` but not `setuptools`: `resampy`
  imports `pkg_resources`, which setuptools >= 81 removed.
- **CPU only.** This is a small model; CPU inference is fast and no GPU is
  required.
