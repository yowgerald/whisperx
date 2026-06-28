import gc
import os
import tempfile
from typing import Optional

import whisperx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="WhisperX CPU Transcription API")

DEVICE = "cpu"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")     # int8 is the recommended CPU setting
MODEL_SIZE = os.getenv("WHISPER_MODEL", "medium")     # large-v3 works but is much slower on CPU
HF_TOKEN = os.getenv("HF_TOKEN")                      # required only for diarization

# Models are loaded lazily and cached in-process so the (slow) first request
# warms things up, and every request after that reuses the loaded model.
_model = None
_diarize_model = None


def get_model():
    global _model
    if _model is None:
        _model = whisperx.load_model(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def get_diarize_model():
    global _diarize_model
    if _diarize_model is None:
        if not HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN env var is required for diarization. "
                "Accept the pyannote model terms on Hugging Face and set HF_TOKEN."
            )
        _diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    return _diarize_model


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_SIZE, "compute_type": COMPUTE_TYPE}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Form(True),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    language: Optional[str] = Form(None),  # leave empty to auto-detect
    batch_size: int = Form(8),             # lower this (e.g. 4) if you run out of RAM
):
    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    try:
        model = get_model()
        audio = whisperx.load_audio(audio_path)

        result = model.transcribe(audio, batch_size=batch_size, language=language or None)

        # Word-level alignment (wav2vec2)
        align_model, align_metadata = whisperx.load_align_model(
            language_code=result["language"], device=DEVICE
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            DEVICE,
            return_char_alignments=False,
        )
        del align_model
        gc.collect()

        # Speaker diarization (pyannote)
        if diarize:
            diarize_model = get_diarize_model()
            diarize_segments = diarize_model(
                audio_path, min_speakers=min_speakers, max_speakers=max_speakers
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)

        return JSONResponse(
            content={"language": result.get("language"), "segments": result["segments"]}
        )
    finally:
        os.remove(audio_path)
