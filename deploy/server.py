#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2024 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A minimal HTTP service wrapping Basic Pitch audio-to-MIDI inference.

The model is loaded once at startup and reused across requests. Upload an
audio file to ``POST /transcribe`` and receive a MIDI file (or note-event
JSON) in return.
"""

import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
from basic_pitch.inference import Model, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("basic_pitch.server")

# Which serialized model to load. Defaults to ONNX so the image stays slim
# (no full TensorFlow dependency). Override with BASIC_PITCH_MODEL=tf|coreml|tflite|onnx.
_MODEL_SUFFIX = {
    "tf": FilenameSuffix.tf,
    "coreml": FilenameSuffix.coreml,
    "tflite": FilenameSuffix.tflite,
    "onnx": FilenameSuffix.onnx,
}[os.environ.get("BASIC_PITCH_MODEL", "onnx").lower()]

# Reject uploads larger than this (bytes) to avoid unbounded memory use.
MAX_UPLOAD_BYTES = int(os.environ.get("BASIC_PITCH_MAX_UPLOAD_BYTES", 50 * 1024 * 1024))

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = build_icassp_2022_model_path(_MODEL_SUFFIX)
    logger.info("Loading Basic Pitch model from %s", model_path)
    _state["model"] = Model(model_path)
    logger.info("Model loaded; service ready")
    yield
    _state.clear()


app = FastAPI(
    title="Basic Pitch",
    description="Audio-to-MIDI transcription HTTP service.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Liveness/readiness probe."""
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (wav, mp3, flac, ogg, ...)."),
    response_format: str = Query("midi", pattern="^(midi|json)$", description="midi | json"),
    onset_threshold: float = Query(0.5, ge=0.0, le=1.0),
    frame_threshold: float = Query(0.3, ge=0.0, le=1.0),
    minimum_note_length: float = Query(127.70, gt=0.0, description="Minimum note length in milliseconds."),
    minimum_frequency: Optional[float] = Query(None, gt=0.0),
    maximum_frequency: Optional[float] = Query(None, gt=0.0),
    multiple_pitch_bends: bool = Query(False),
    melodia_trick: bool = Query(True),
):
    """Transcribe an uploaded audio file to MIDI.

    Returns a ``.mid`` file by default, or the predicted note events as JSON
    when ``response_format=json``.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_BYTES} bytes.")

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        _, midi_data, note_events = predict(
            tmp_path,
            _state["model"],
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            multiple_pitch_bends=multiple_pitch_bends,
            melodia_trick=melodia_trick,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean 422 to the client
        logger.exception("Transcription failed")
        raise HTTPException(status_code=422, detail=f"Could not transcribe audio: {exc}") from exc
    finally:
        os.unlink(tmp_path)

    if response_format == "json":
        events = [
            {
                "start_time": float(start),
                "end_time": float(end),
                "pitch_midi": int(pitch),
                "amplitude": float(amplitude),
                "pitch_bends": [int(b) for b in bends] if bends else None,
            }
            for start, end, pitch, amplitude, bends in note_events
        ]
        return JSONResponse({"note_count": len(events), "notes": events})

    # Write to a temp file then read back: pretty_midi's file-like support
    # varies across versions, but writing to a path is universally supported.
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as midi_tmp:
        midi_path = midi_tmp.name
    try:
        midi_data.write(midi_path)
        with open(midi_path, "rb") as fh:
            buffer = io.BytesIO(fh.read())
    finally:
        os.unlink(midi_path)

    buffer.seek(0)
    stem = os.path.splitext(os.path.basename(file.filename or "transcription"))[0]
    return StreamingResponse(
        buffer,
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{stem}.mid"'},
    )
