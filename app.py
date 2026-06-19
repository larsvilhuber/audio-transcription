import os
import sys
import time
import uuid
import shutil
import threading
import multiprocessing
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_file

import storage
from transcriber import convert_to_wav, transcription_proc, MODEL_FAST, MODEL_PRECISE

UPLOAD_DIR = Path("/tmp/audio-transcription")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".m4a", ".mp4", ".wav", ".ogg", ".flac", ".webm", ".aac"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

# Per-job subprocess handles, so cancel can kill immediately.
_PROCESSES: dict[str, multiprocessing.Process] = {}
_PROCESSES_LOCK = threading.Lock()

# Use spawn so worker processes start with a clean CUDA context — fork after
# torch.cuda.is_available() leaves a broken CUDA state in the child.
_mp_ctx = multiprocessing.get_context("spawn")

# Manager server provides a shared dict the subprocess can write to and
# Flask can read from. Created in main() — NOT at import — because `spawn`
# re-imports this module in every worker; creating it here would spawn a
# redundant manager process in each worker.
_mp_manager = None

BASE_PATH = os.environ.get("APP_BASE_PATH", "/")

IDLE_GPU_TIMEOUT = float(os.environ.get("IDLE_GPU_TIMEOUT_HOURS", "1.0")) * 3600

RESULT_RETENTION = float(os.environ.get("RESULT_RETENTION_HOURS", "24.0")) * 3600

_last_gpu_activity: float | None = None
_gpu_activity_lock = threading.Lock()


def _update_gpu_activity():
    global _last_gpu_activity
    with _gpu_activity_lock:
        _last_gpu_activity = time.monotonic()


def _purge_expired():
    """Delete on-disk jobs past the retention window. Never purges a running
    job (its dir is protected even if old — a long transcription can outlive
    the window)."""
    with JOBS_LOCK:
        protected = {jid for jid, j in JOBS.items() if j.get("status") == "running"}
    removed = storage.purge_expired(RESULT_RETENTION, protected)
    if removed:
        with JOBS_LOCK:
            for jid in removed:
                JOBS.pop(jid, None)


def _idle_monitor():
    while True:
        time.sleep(60)
        _purge_expired()
        with _gpu_activity_lock:
            last = _last_gpu_activity
        if last is None:
            continue
        if time.monotonic() - last < IDLE_GPU_TIMEOUT:
            continue
        with JOBS_LOCK:
            any_running = any(j["status"] == "running" for j in JOBS.values())
        if any_running:
            continue
        print("GPU idle timeout reached — restarting to free VRAM", flush=True)
        _mp_manager.shutdown()
        os.execv(sys.executable, [sys.executable] + sys.argv)


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _start_job(job_id, filename, model_choice, language, wav_path, created_at):
    """Write the initial meta, spawn the worker subprocess, and start the
    thread that reaps it. Shared by /upload and /retranscribe."""
    model_name = MODEL_FAST if model_choice == "fast" else MODEL_PRECISE

    storage.write_meta(job_id, {
        "job_id": job_id,
        "filename": filename,
        "model": model_choice,
        "status": "running",
        "language": None,
        "created_at": created_at,
        "stats": None,
        "error": None,
    })

    job = _mp_manager.dict({
        "status": "running",
        "progress": "Queued",
        "filename": filename,
        "language": None,
        "cancelled": False,
        "txt": None,
        "docx": None,
        "error": None,
        "created_at": created_at,
    })
    with JOBS_LOCK:
        JOBS[job_id] = job

    p = _mp_ctx.Process(
        target=transcription_proc,
        args=(job, wav_path, filename, model_name, language, job_id),
        daemon=True,
    )
    with _PROCESSES_LOCK:
        _PROCESSES[job_id] = p
    p.start()

    def _monitor():
        p.join()
        _update_gpu_activity()
        with _PROCESSES_LOCK:
            _PROCESSES.pop(job_id, None)
        # If the worker died without finalizing (crash/OOM), record an error so
        # the dir doesn't linger as a stuck "running" job.
        try:
            if job.get("status") == "running":
                job["status"] = "error"
                job["error"] = "Worker exited unexpectedly"
                storage.finalize_meta(job_id, "error", error="Worker exited unexpectedly")
        except Exception:
            pass
        # Clean up the disposable WAV if the worker was killed before its
        # finally block ran.
        try:
            os.remove(wav_path)
        except OSError:
            pass

    threading.Thread(target=_monitor, daemon=True).start()


def _load_jobs_from_disk():
    """Rebuild the in-memory job index from disk on startup. Completed jobs are
    restored as plain dicts; interrupted (non-done) jobs can never resume, so
    their dirs are removed."""
    for meta in storage.load_all():
        jid = meta.get("job_id")
        if not jid:
            continue
        if meta.get("status") == "done":
            restored = dict(meta)
            restored.setdefault("progress", "Complete")
            JOBS[jid] = restored
        else:
            storage.delete_job(jid)


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

    language = request.form.get("language") or None
    if language and (len(language) > 8 or not language.isalpha()):
        language = None

    job_id = str(uuid.uuid4())
    suffix = Path(f.filename).suffix.lower()

    # Keep the original audio so the job can be re-run with a different model.
    source_path = storage.job_dir(job_id) / f"source{suffix}"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    f.save(str(source_path))

    # Always normalize to a disposable 16 kHz mono WAV for the pipeline; the
    # original stays in the job dir.
    wav_path = str(UPLOAD_DIR / f"{job_id}.wav")
    try:
        convert_to_wav(str(source_path), wav_path)
    except RuntimeError as exc:
        storage.delete_job(job_id)
        return jsonify({"error": str(exc)}), 500

    _start_job(job_id, f.filename, model_choice, language, wav_path, time.time())
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
    with _PROCESSES_LOCK:
        p = _PROCESSES.get(job_id)
    if p and p.is_alive():
        p.terminate()
        p.join(timeout=3)
        if p.is_alive():
            p.kill()
    if job["status"] == "running":
        job["status"] = "cancelled"
        job["progress"] = "Cancelled"
    # A cancelled job has no output worth keeping; drop its dir so it never
    # shows up in the recent list or survives a restart.
    storage.delete_job(job_id)
    return jsonify({"ok": True})


def _download(job_id, kind):
    meta = storage.read_meta(job_id)
    if meta is None or meta.get("status") != "done":
        return jsonify({"error": "Not ready"}), 404
    stem = Path(meta.get("filename") or job_id).stem
    if kind == "txt":
        path, mime, name = storage.txt_path(job_id), "text/plain", f"{stem}.txt"
    else:
        path = storage.docx_path(job_id)
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        name = f"{stem}.docx"
    if not path.exists():
        return jsonify({"error": "Not ready"}), 404
    return send_file(str(path), mimetype=mime, as_attachment=True, download_name=name)


@app.route("/download/<job_id>/txt")
def download_txt(job_id):
    return _download(job_id, "txt")


@app.route("/download/<job_id>/docx")
def download_docx(job_id):
    return _download(job_id, "docx")


@app.route("/recent")
def recent():
    return jsonify(storage.list_done())


@app.route("/retranscribe/<job_id>", methods=["POST"])
def retranscribe(job_id):
    src = storage.find_source(job_id)
    old = storage.read_meta(job_id)
    if src is None or old is None:
        return jsonify({"error": "Original audio no longer available"}), 404

    data = request.get_json(silent=True) or {}
    model_choice = data.get("model") or old.get("model") or "precise"
    language = data.get("language") or None
    if language and (len(language) > 8 or not language.isalpha()):
        language = None

    filename = old.get("filename") or "audio"
    new_id = str(uuid.uuid4())
    new_src = storage.job_dir(new_id) / f"source{src.suffix}"
    new_src.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(new_src))

    wav_path = str(UPLOAD_DIR / f"{new_id}.wav")
    try:
        convert_to_wav(str(new_src), wav_path)
    except RuntimeError as exc:
        storage.delete_job(new_id)
        return jsonify({"error": str(exc)}), 500

    _start_job(new_id, filename, model_choice, language, wav_path, time.time())
    return jsonify({"job_id": new_id})


def main():
    # All startup side effects live here so they run ONLY in the real main
    # process. `spawn` re-imports this module in every worker to rebuild
    # __main__; doing these at import time would make each worker recreate the
    # Manager, start an idle monitor, and — worst — re-run the disk loader,
    # which deletes the very job the worker is running (its meta is still
    # "running"), wiping the source audio and initial meta.
    global _mp_manager
    _mp_manager = multiprocessing.Manager()
    _load_jobs_from_disk()
    threading.Thread(target=_idle_monitor, daemon=True, name="idle-gpu-monitor").start()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
