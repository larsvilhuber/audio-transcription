# audio-transcription

Two scripts for transcribing audio files using [Whisper](https://github.com/openai/whisper).

## Scripts

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
