# audio-transcription

A web-based audio transcription service with a choice of transcription backends,
plus two stand-alone CLI scripts.

## Web application

Run `python app.py` and open <http://localhost:5000>.  Upload an audio file and
choose one of three transcription models:

| Model | Backend | Speaker identification | Hardware required |
|-------|---------|----------------------|-------------------|
| **Precise** | WhisperX `large-v3` (local) | ✅ Yes — via [pyannote](https://github.com/pyannote/pyannote-audio) diarization | GPU with ≥ 10 GB VRAM (e.g. 12 GB) |
| **Fast** | WhisperX `base` (local) | ✅ Yes — via pyannote diarization | GPU with ≥ 1 GB VRAM, or CPU |
| **Mistral Voxtral** | [Mistral AI cloud API](https://mistral.ai) | ❌ No — transcript only, no speaker labels | None (cloud API) |

### Speaker identification notes

- **Precise / Fast** run the full [WhisperX](https://github.com/m-bain/whisperX)
  pipeline: transcription → timestamp alignment → speaker diarization.  Each
  passage of speech is labelled with a speaker ID (e.g. `[SPEAKER_00]`).
  Requires a Hugging Face token and accepted user conditions for the pyannote
  models.
- **Mistral Voxtral** sends the audio to Mistral's cloud transcription API
  (`voxtral-mini-2507`) and returns the transcript text only.  No speaker
  identification is performed.

### Minimum hardware requirements

| Use case | Requirement |
|----------|-------------|
| Mistral Voxtral only | Any machine with internet access — no GPU needed |
| Fast (WhisperX `base`) | CPU (slow) or any CUDA GPU |
| Precise (WhisperX `large-v3`) | CUDA GPU with **≥ 10 GB VRAM** (tested on 12 GB) |

### API keys

**Whisper / WhisperX models** require a [Hugging Face token](https://huggingface.co/settings/tokens)
stored as `HF_TOKEN` in your `.env` file.  You must also accept the user
conditions for the pyannote diarization models on Hugging Face.

**Mistral Voxtral** requires a Mistral AI API key stored as `MISTRAL_API_KEY` in
your `.env` file.  Create or retrieve your key at
<https://console.mistral.ai/api-keys/>.

```
HF_TOKEN=hf_your_token_here
MISTRAL_API_KEY=your_mistral_key_here
```

Only the key(s) for the backend(s) you intend to use are required.

---

## CLI scripts

**`transcribe-only.py`** — Simple transcription using OpenAI Whisper (`large` model). No speaker labels.

```bash
python transcribe-only.py audio.m4a
```

**`transcribe.py`** — Full pipeline using [WhisperX](https://github.com/m-bain/whisperX): transcription + timestamp alignment + speaker diarization. Requires a Hugging Face token for diarization.

```bash
python transcribe.py audio.m4a
```

Both scripts write output to a `.txt` file alongside the input audio.

## Setup

```bash
pip install -r requirements.txt
```

For `transcribe.py`, create a `.env` file with your [Hugging Face token](https://huggingface.co/settings/tokens):

```
HF_TOKEN=your_token_here
```

You must also accept the user conditions for the pyannote diarization models on Hugging Face.
