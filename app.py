import os
import uuid
import threading
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_file
import io

from transcriber import convert_to_wav, run_transcription

UPLOAD_DIR = Path("/tmp/audio-transcription")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".m4a", ".mp4", ".wav", ".ogg", ".flac", ".webm", ".aac"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "audio" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["audio"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400
    if not _allowed(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    job_id = str(uuid.uuid4())
    suffix = Path(f.filename).suffix.lower()
    raw_path = UPLOAD_DIR / f"{job_id}{suffix}"
    f.save(str(raw_path))

    # Convert to WAV if needed
    if suffix == ".wav":
        wav_path = str(raw_path)
    else:
        wav_path = str(UPLOAD_DIR / f"{job_id}.wav")
        try:
            convert_to_wav(str(raw_path), wav_path)
        except RuntimeError as exc:
            raw_path.unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 500
        finally:
            if suffix != ".wav":
                raw_path.unlink(missing_ok=True)

    job = {
        "status": "running",
        "progress": "Queued",
        "filename": f.filename,
        "txt": None,
        "docx": None,
        "error": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    t = threading.Thread(
        target=run_transcription,
        args=(job, wav_path, f.filename),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    resp = {"status": job["status"], "progress": job["progress"]}
    if job["status"] == "error":
        resp["error"] = job["error"]
    return jsonify(resp)


@app.route("/download/<job_id>/txt")
def download_txt(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    stem = Path(job["filename"]).stem
    return send_file(
        io.BytesIO(job["txt"].encode("utf-8")),
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"{stem}.txt",
    )


@app.route("/download/<job_id>/docx")
def download_docx(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    stem = Path(job["filename"]).stem
    return send_file(
        io.BytesIO(job["docx"]),
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        as_attachment=True,
        download_name=f"{stem}.docx",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
