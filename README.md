# WhisperX RunPod Serverless

GPU-accelerated speech transcription with speaker diarization, deployed on [RunPod Serverless](https://www.runpod.io/).

## Features

- Audio transcription via [WhisperX](https://github.com/m-bain/whisperX)
- Speaker diarization via [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- Word-level timestamp alignment
- Output formats: JSON (segments), SRT (subtitles), TXT (plain text)
- GPU auto-detection (CUDA/cpu), models preloaded at worker start

## RunPod Deployment

### Prerequisites

1. Accept model terms on Hugging Face:
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Create [Hugging Face token](https://huggingface.co/settings/tokens)

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | For diarization | — | HuggingFace access token |
| `WHISPER_MODEL` | No | `medium` | Model size: `large-v3-turbo`, `large-v3`, `medium`, etc. |
| `COMPUTE_TYPE` | No | `float16` (GPU) / `int8` (CPU) | `float16`, `float32`, `int8` |

### Recommended RunPod Config

- **GPU:** 16 GB VRAM
- **Model:** `large-v3-turbo` (11 GB VRAM, 8x faster than `large-v3`)
- **Max Workers:** 1 (scale to need)
- **Min Workers:** 0 (no idle cost)
- **Network Volume:** Mount to `/runpod-volume` for model cache persistence

### Build & Deploy

1. Push to GitHub
2. In RunPod, create serverless endpoint from GitHub repo
3. Select `runpod-serverless` branch under Repository Configuration
4. Set environment variables under Settings
5. Deploy

## API

Send job to RunPod endpoint:

```json
{
  "input": {
    "audio_url": "https://example.com/audio.mp3",
    "diarize": true,
    "language": "en",
    "response_format": "srt",
    "max_line_width": 42,
    "max_lines": 2,
    "speaker_format": "inline"
  }
}
```

### Input Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_url` | string | **required** | URL to audio file |
| `diarize` | bool | `true` | Enable speaker diarization |
| `language` | string | auto | Force language code (e.g. `"en"`) |
| `response_format` | string | `null` | `"srt"`, `"txt"`, or `null` (JSON segments) |
| `batch_size` | int | `8` | Transcription batch size |
| `max_line_width` | int | `42` | Chars per SRT line |
| `max_lines` | int | `2` | Max SRT lines per block |
| `max_duration` | float | `8.0` | Max SRT block duration (seconds) |
| `speaker_format` | string | `"inline"` | `"inline"`, `"newline"`, `"none"` |
| `min_speakers` | int | `null` | Min speaker count |
| `max_speakers` | int | `null` | Max speaker count |

### Response

**JSON** (default):
```json
{
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello world",
      "speaker": "SPEAKER_00",
      "words": [{"word": "Hello", "start": 0.0, "end": 0.5}, ...]
    }
  ]
}
```

**SRT**: Returns `{"srt": "...", "language": "en"}`

**TXT**: Returns `{"txt": "...", "language": "en"}`

## Local Dev

FastAPI server included for local testing (not used in RunPod):

```bash
cp .env.example .env  # edit HF_TOKEN
uvicorn app:app --host 0.0.0.0 --port 5000
```

## Image

- Base: `runpod/pytorch:1.0.7-cu1290-torch271-ubuntu2204`
- CUDA 12.9, PyTorch 2.7.1
