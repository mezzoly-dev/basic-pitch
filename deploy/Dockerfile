# Slim inference image for Basic Pitch, serving an HTTP transcription API.
#
# Uses the lightweight ONNX runtime instead of full TensorFlow, so the image is
# a few hundred MB rather than multiple GB. The ~2 MB ONNX model ships with the
# `basic-pitch` package, so no separate download is needed.
#
# NOTE: Python 3.10 is deliberate. On Python >= 3.11, `tensorflow` is a *base*
# dependency of basic-pitch (see pyproject.toml), which would bloat the image to
# multiple GB. On 3.10 the base dependency is the small `tflite-runtime`, and the
# [onnx] extra adds onnxruntime -- no TensorFlow.
#
# Build (from the repo root):
#   docker build -f deploy/Dockerfile -t basic-pitch-api .
# Run:
#   docker run --rm -p 8000:8000 basic-pitch-api
FROM python:3.10-slim

# System libraries needed for audio decoding (librosa/soundfile + ffmpeg/sox).
RUN apt-get update \
  && apt-get install --no-install-recommends -y \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-fmt-all \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the HTTP server deps first (better layer caching), then Basic Pitch.
# Note: we upgrade pip but deliberately NOT setuptools -- resampy (a transitive
# dependency) imports `pkg_resources`, which setuptools >= 81 removes.
COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r /app/deploy/requirements.txt

# Install Basic Pitch from the local source tree with the ONNX runtime.
COPY . /app
RUN pip install --no-cache-dir ".[onnx]"

ENV BASIC_PITCH_MODEL=onnx \
    PORT=8000

EXPOSE 8000

# Honour $PORT (e.g. when a platform injects it) but default to 8000.
CMD ["sh", "-c", "uvicorn deploy.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
