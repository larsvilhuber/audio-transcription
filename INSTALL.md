# Installation Guide

## Prerequisites

- Python 3.10+
- `ffmpeg` installed and on PATH
- CUDA-capable GPU recommended (falls back to CPU)
- nginx installed
- A Hugging Face token with access to the pyannote diarization models

---

## 1. Clone and set up the repo

```bash
git clone <repo-url> /home/transcription
cd /home/transcription
```

Create a `.env` file with your Hugging Face token:

```bash
echo "HF_TOKEN=hf_your_token_here" > /home/transcription/.env
chmod 600 /home/transcription/.env
```

## 2. Set up Python virtual environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 3. Test the Flask app directly

```bash
.venv/bin/python app.py
# Should start on http://127.0.0.1:5000
```

Press Ctrl-C when satisfied.

## 4. Install nginx site config

```bash
sudo cp audio-transcription.nginx /etc/nginx/sites-available/audio-transcription
sudo ln -s /etc/nginx/sites-available/audio-transcription \
           /etc/nginx/sites-enabled/audio-transcription
sudo nginx -t
sudo systemctl reload nginx
```

## 5. Install and start the systemd service

```bash
sudo cp audio-transcription.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable audio-transcription
sudo systemctl start audio-transcription
```

---

## Verification

```bash
# Service running
sudo systemctl status audio-transcription

# Port listening
ss -tlnp | grep 5000

# Nginx config valid
sudo nginx -t

# Direct Flask check
curl -s http://127.0.0.1:5000/        # returns HTML

# Through nginx
curl -s http://cv-ai/audio/            # returns HTML

# Upload test
curl -s -X POST http://cv-ai/audio/upload -F "audio=@test.mp3"
# returns: {"job_id": "..."}

# Poll status
curl -s http://cv-ai/audio/status/<job_id>

# Download results (after status == "done")
curl -o result.txt  http://cv-ai/audio/download/<job_id>/txt
curl -o result.docx http://cv-ai/audio/download/<job_id>/docx

# Logs
sudo journalctl -u audio-transcription -f
```
