# Installation Guide

## Prerequisites

- Python 3.10+
- `ffmpeg` installed and on PATH
- nginx installed
- For **Precise** model (WhisperX `large-v3`): CUDA GPU with **≥ 10 GB VRAM** (tested on 12 GB)
- For **Fast** model (WhisperX `base`): any CUDA GPU, or CPU (slower)
- For **Mistral Voxtral** model: internet access only — no local GPU required
- A [Hugging Face token](https://huggingface.co/settings/tokens) with access to the pyannote diarization models (required for Precise / Fast)
- A [Mistral AI API key](https://console.mistral.ai/api-keys/) (required for Mistral Voxtral)

---

## 1. Clone the repo

```bash
sudo mkdir -p /home/transcription
git clone <repo-url> /home/transcription/app
```

## 2. Create the environment file

The `.env` lives **outside** the app directory so it is never overwritten by a `git pull`:

```bash
cat > /home/transcription/.env << 'EOF'
# Required for Precise and Fast (WhisperX) models.
# Get your token at https://huggingface.co/settings/tokens
# You must also accept the pyannote model conditions on Hugging Face.
HF_TOKEN=hf_your_token_here

# Required for the Mistral Voxtral model.
# Get your API key at https://console.mistral.ai/api-keys/
MISTRAL_API_KEY=your_mistral_key_here

APP_BASE_PATH=/audio/
EOF
chmod 600 /home/transcription/.env
```

Only include the key(s) for the backend(s) you intend to use.

## 3. Set up the Python virtual environment

```bash
cd /home/transcription/app
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 4. Test the Flask app directly

```bash
cd /home/transcription/app
APP_BASE_PATH=/audio/ .venv/bin/python app.py
# Should start on http://0.0.0.0:5000
```

Press Ctrl-C when satisfied.

## 5. Configure nginx

This app uses a **location snippet** that is included inside the default nginx
server block. This requires a one-time modification to the default server, plus
installing the snippet file.

### 5a. Prepare the snippet directory

```bash
sudo mkdir -p /etc/nginx/location-snippets
```

### 5b. One-time patch of the default nginx server

Add the following line inside the `server { ... }` block in
`/etc/nginx/sites-available/default`, right after the `server_name _;` line:

```nginx
include /etc/nginx/location-snippets/*.location;
```

### 5c. Install and enable the location snippet

```bash
sudo cp audio-transcription.location \
        /etc/nginx/sites-available/audio-transcription.location
sudo ln -s /etc/nginx/sites-available/audio-transcription.location \
           /etc/nginx/location-snippets/audio-transcription.location
sudo nginx -t
sudo systemctl reload nginx
```

To **disable** the audio site without removing anything:

```bash
sudo rm /etc/nginx/location-snippets/audio-transcription.location
sudo systemctl reload nginx
```

To **re-enable**:

```bash
sudo ln -s /etc/nginx/sites-available/audio-transcription.location \
           /etc/nginx/location-snippets/audio-transcription.location
sudo systemctl reload nginx
```

## 6. Install and start the systemd service

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

# Upload test (model: "fast", "precise", or "mistral")
curl -s -X POST http://cv-ai/audio/upload \
     -F "audio=@test.mp3" \
     -F "model=precise"
# returns: {"job_id": "..."}

# Poll status (returns language once detected)
curl -s http://cv-ai/audio/status/<job_id>

# Cancel a running job
curl -s -X POST http://cv-ai/audio/cancel/<job_id>

# Download results (after status == "done")
curl -o result.txt  http://cv-ai/audio/download/<job_id>/txt
curl -o result.docx http://cv-ai/audio/download/<job_id>/docx

# Logs
sudo journalctl -u audio-transcription -f
```

---

## Updating

```bash
cd /home/transcription/app
git pull
sudo systemctl restart audio-transcription
```
