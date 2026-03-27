import os
import uuid
import threading
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_file
import io

from transcriber import convert_to_wav, run_transcription, MODEL_FAST, MODEL_PRECISE, MODEL_MISTRAL

UPLOAD_DIR = Path("/tmp/audio-transcription")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".m4a", ".mp4", ".wav", ".ogg", ".flac", ".webm", ".aac"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


BASE_PATH = os.environ.get("APP_BASE_PATH", "/")


@app.route("/")
def index():
    return render_template("index.html", base_path=BASE_PATH)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "File too large — check server upload limits"}), 413


@app.route("/upload", methods=["POST"])
def upload():
    if "audio" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["audio"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400
    if not _allowed(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    model_choice = request.form.get("model", "precise")
    if model_choice == "fast":
        model_name = MODEL_FAST
    elif model_choice == "mistral":
        model_name = MODEL_MISTRAL
    else:
        model_name = MODEL_PRECISE

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
        "language": None,
        "cancelled": False,
        "txt": None,
        "docx": None,
        "error": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    t = threading.Thread(
        target=run_transcription,
        args=(job, wav_path, f.filename, model_name),
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
    resp = {
        "status": job["status"],
        "progress": job["progress"],
        "language": job.get("language"),
    }
    if job["status"] == "done":
        resp["stats"] = job.get("stats")
    if job["status"] == "error":
        resp["error"] = job["error"]
    return jsonify(resp)


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] == "running":
        job["cancelled"] = True
        job["progress"] = "Cancelling..."
    return jsonify({"ok": True})


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
    app.run(host="0.0.0.0", port=5000, debug=False)
