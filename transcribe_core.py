import gc
import os

import torch

# PyTorch 2.6+ defaults torch.load to weights_only=True.
# pyannote.audio 3.3.2 checkpoints contain omegaconf objects not in
# the safe globals list. HuggingFace checkpoints are trusted here.
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

import whisperx
from whisperx.diarize import DiarizationPipeline

# --- Device / compute config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
MODEL_SIZE = os.getenv("WHISPER_MODEL", "medium")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Global model cache ---
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
        _diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    return _diarize_model


def _free_gpu():
    """Release cached GPU memory between pipeline stages (no-op on CPU)."""
    if DEVICE == "cuda":
        gc.collect()
        torch.cuda.empty_cache()


def preload_models():
    """Preload models at worker startup so first request doesn't timeout."""
    get_model()
    # Diarization model only preloaded if HF_TOKEN is set
    if HF_TOKEN:
        get_diarize_model()


# ---------------------------------------------------------------------------
# SRT / TXT export helpers
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Convert float seconds to SRT timestamp ``HH:MM:SS,mmm``."""
    if seconds is None:
        return "00:00:00,000"
    assert seconds >= 0, "non-negative timestamp expected"
    ms = round(seconds * 1000.0)
    h = ms // 3_600_000
    ms -= h * 3_600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1_000
    ms -= s * 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_subtitle_blocks(segment_words, speaker, segment_start, segment_end,
                           max_line_width, max_lines, max_duration):
    """Split word-level timestamps into readable subtitle blocks."""
    blocks = []
    current_words = []
    current_lines = 1
    current_line_len = 0
    block_start = segment_start
    block_end = segment_start

    for w in segment_words:
        word_text = w.get("word", "").strip()
        if not word_text:
            continue

        w_start = w.get("start")
        w_end = w.get("end")
        if w_start is None:
            w_start = block_end
        if w_end is None:
            w_end = w_start + 0.5
        if block_start is None:
            block_start = w_start

        duration = (w_end - block_start) if (w_end is not None and block_start is not None) else 0
        needs_wrap = current_line_len > 0 and (current_line_len + 1 + len(word_text) > max_line_width)

        if needs_wrap:
            if current_lines >= max_lines or duration > max_duration:
                blocks.append({
                    "start": block_start, "end": block_end,
                    "text": " ".join(current_words).replace(" \n ", "\n").strip(),
                    "speaker": speaker,
                })
                current_words = []
                current_lines = 1
                current_line_len = 0
                block_start = w_start
                block_end = w_end
            else:
                current_lines += 1
                current_line_len = 0
                current_words.append("\n")
        elif current_words and duration > max_duration:
            blocks.append({
                "start": block_start, "end": block_end,
                "text": " ".join(current_words).replace(" \n ", "\n").strip(),
                "speaker": speaker,
            })
            current_words = []
            current_lines = 1
            current_line_len = 0
            block_start = w_start
            block_end = w_end

        current_words.append(word_text)
        current_line_len += len(word_text) + (1 if current_line_len > 0 else 0)
        block_end = w_end

    if current_words:
        blocks.append({
            "start": block_start,
            "end": block_end if block_end else segment_end,
            "text": " ".join(current_words).replace(" \n ", "\n").strip(),
            "speaker": speaker,
        })

    return blocks


def build_srt(result, max_line_width=42, max_lines=2, max_duration=8.0, speaker_format="inline"):
    """Build SRT string from transcription result."""
    entries = []
    idx = 1

    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "UNKNOWN")
        words = seg.get("words", [])
        text = seg["text"].strip()

        if not words:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            if speaker_format == "inline":
                srt_text = f"[{speaker}]: {text}"
            elif speaker_format == "newline":
                srt_text = f"[{speaker}]:\n{text}"
            else:
                srt_text = text
            entries.append(f"{idx}\n{start} --> {end}\n{srt_text}\n")
            idx += 1
            continue

        for block in _build_subtitle_blocks(words, speaker, seg["start"], seg["end"],
                                            max_line_width, max_lines, max_duration):
            start = format_timestamp(block["start"])
            end = format_timestamp(block["end"])
            block_text = block["text"]
            if speaker_format == "inline":
                srt_text = f"[{speaker}]: {block_text}"
            elif speaker_format == "newline":
                srt_text = f"[{speaker}]:\n{block_text}"
            else:
                srt_text = block_text
            entries.append(f"{idx}\n{start} --> {end}\n{srt_text}\n")
            idx += 1

    return "\n".join(entries)


def build_txt(result):
    """Build plain-text transcript with speaker labels."""
    lines = []
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "UNKNOWN")
        lines.append(f"[{speaker}]: {seg['text'].strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core transcription pipeline
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path, diarize=True, min_speakers=None, max_speakers=None,
                     language=None, batch_size=8):
    """Run full WhisperX pipeline on an audio file.

    Returns dict with keys: ``language``, ``segments``.
    """
    model = get_model()
    audio = whisperx.load_audio(audio_path)

    # 1. Transcribe
    result = model.transcribe(audio, batch_size=batch_size, language=language or None)
    language_detected = result["language"]

    # 2. Word-level alignment
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language_detected, device=DEVICE
    )
    result = whisperx.align(
        result["segments"], align_model, align_metadata, audio, DEVICE,
        return_char_alignments=False,
    )
    del align_model
    _free_gpu()

    # 3. Speaker diarization
    if diarize:
        diarize_model = get_diarize_model()
        diarize_segments = diarize_model(
            audio_path, min_speakers=min_speakers, max_speakers=max_speakers
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)
        _free_gpu()

    return {"language": language_detected, "segments": result["segments"]}
