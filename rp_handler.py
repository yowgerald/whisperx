import os
import shutil
import tempfile
import time
import urllib.request
from typing import Optional

import runpod

from transcribe_core import (
    preload_models,
    transcribe_audio,
    build_srt,
    build_txt,
    DEVICE,
    MODEL_SIZE,
    COMPUTE_TYPE,
    HF_TOKEN,
)

# Preload models at module level so they stay warm in GPU VRAM across jobs.
print(f"[startup] device={DEVICE} model={MODEL_SIZE} compute={COMPUTE_TYPE} "
      f"diarization={'enabled' if HF_TOKEN else 'disabled (no HF_TOKEN)'}")
preload_models()
print("[startup] models loaded, worker ready")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def handler(job):
    """RunPod serverless handler.

    Expected ``job["input"]`` keys:

    - **audio_url** (str, required): URL to the audio file to transcribe.
    - **diarize** (bool, default True): Enable speaker diarization.
    - **min_speakers** (int, optional): Minimum speaker count for diarization.
    - **max_speakers** (int, optional): Maximum speaker count for diarization.
    - **language** (str, optional): Force language code (e.g. ``"en"``).
    - **batch_size** (int, default 8): Batch size for transcription.
    - **response_format** (str, optional): ``"srt"``, ``"txt"``, or None (JSON).
    - **max_line_width** (int, default 42): SRT line width.
    - **max_lines** (int, default 2): SRT max lines per block.
    - **max_duration** (float, default 8.0): SRT max block duration.
    - **speaker_format** (str, default "inline"): ``"inline"``, ``"newline"``, or ``"none"``.
    """
    job_input = job.get("input", {})

    audio_url = job_input.get("audio_url")
    if not audio_url:
        return {"error": "Missing required field: audio_url"}

    diarize = job_input.get("diarize", True)
    if diarize and not HF_TOKEN:
        _log("diarization requested but HF_TOKEN not set — disabling diarization")
        diarize = False

    min_speakers = job_input.get("min_speakers")
    max_speakers = job_input.get("max_speakers")
    language = job_input.get("language")
    batch_size = job_input.get("batch_size", 8)
    response_format = job_input.get("response_format")
    max_line_width = job_input.get("max_line_width", 42)
    max_lines = job_input.get("max_lines", 2)
    max_duration = job_input.get("max_duration", 8.0)
    speaker_format = job_input.get("speaker_format", "inline")

    # Download audio file
    suffix = os.path.splitext(audio_url.split("/")[-1].split("?")[0])[1] or ".wav"
    _log(f"downloading from {audio_url}")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_path = tmp.name
        try:
            with urllib.request.urlopen(audio_url, timeout=300) as resp:
                with open(audio_path, "wb") as f:
                    shutil.copyfileobj(resp, f)
        except Exception:
            os.remove(audio_path)
            raise

    try:
        _log(f"transcribing {audio_path} (diarize={diarize}, language={language or 'auto'})")
        t0 = time.time()
        result = transcribe_audio(
            audio_path,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            language=language,
            batch_size=batch_size,
        )
        _log(f"transcription done in {time.time() - t0:.1f}s, "
             f"language={result['language']}, segments={len(result['segments'])}")

        # Format response
        if response_format == "srt":
            srt = build_srt(result, max_line_width, max_lines, max_duration, speaker_format)
            return {"srt": srt, "language": result["language"]}
        elif response_format == "txt":
            txt = build_txt(result)
            return {"txt": txt, "language": result["language"]}
        else:
            return {
                "language": result["language"],
                "segments": result["segments"],
            }
    finally:
        os.remove(audio_path)
        _log("cleaned up temp file")


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
