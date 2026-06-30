import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from transcribe_core import (
    build_srt,
    build_txt,
    transcribe_audio,
    DEVICE,
    MODEL_SIZE,
    COMPUTE_TYPE,
    HF_TOKEN,
)

app = FastAPI(title="WhisperX Transcription API")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model": MODEL_SIZE,
        "compute_type": COMPUTE_TYPE,
        "diarization_available": HF_TOKEN is not None,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Form(True),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    language: Optional[str] = Form(None),
    batch_size: int = Form(8),
    response_format: Optional[str] = Form(None),
    max_line_width: int = Form(42),
    max_lines: int = Form(2),
    max_duration: float = Form(8.0),
    speaker_format: str = Form("inline"),
):
    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    try:
        result = transcribe_audio(
            audio_path,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            language=language,
            batch_size=batch_size,
        )

        if response_format == "srt":
            content = build_srt(result, max_line_width, max_lines, max_duration, speaker_format)
            return Response(content, media_type="text/plain",
                            headers={"Content-Disposition": "attachment; filename=transcription.srt"})
        elif response_format == "txt":
            content = build_txt(result)
            return Response(content, media_type="text/plain",
                            headers={"Content-Disposition": "attachment; filename=transcription.txt"})
        else:
            return JSONResponse(content={
                "language": result["language"],
                "segments": result["segments"],
            })
    finally:
        os.remove(audio_path)
