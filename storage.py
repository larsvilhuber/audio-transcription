"""Disk-backed persistence for transcription jobs.

Layout: data/jobs/<job_id>/{meta.json, transcript.txt, transcript.docx, source<ext>}

This module must NOT import app.py: the transcription worker runs in a spawned
subprocess that imports transcriber.py (and, through it, this module). Importing
app.py there would recreate the Manager and idle-monitor thread in the worker.
"""
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "jobs"


def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def job_dir(job_id):
    return DATA_DIR / job_id


def _meta_path(job_id):
    return job_dir(job_id) / "meta.json"


def write_meta(job_id, meta):
    """Atomically write meta.json (temp file + os.replace)."""
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(d), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        os.replace(tmp, str(_meta_path(job_id)))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_meta(job_id):
    try:
        with open(_meta_path(job_id), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def finalize_meta(job_id, status, language=None, stats=None, error=None):
    """Merge a final status into the existing meta (preserving created_at etc.)."""
    meta = read_meta(job_id) or {"job_id": job_id}
    meta["status"] = status
    if language is not None:
        meta["language"] = language
    if stats is not None:
        meta["stats"] = stats
    if error is not None:
        meta["error"] = error
    write_meta(job_id, meta)
    return meta


def write_outputs(job_id, txt, docx):
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "transcript.txt").write_text(txt, encoding="utf-8")
    (d / "transcript.docx").write_bytes(docx)


def txt_path(job_id):
    return job_dir(job_id) / "transcript.txt"


def docx_path(job_id):
    return job_dir(job_id) / "transcript.docx"


def find_source(job_id):
    """Return the stored original audio Path for a job, or None."""
    d = job_dir(job_id)
    if not d.is_dir():
        return None
    for p in d.iterdir():
        if p.stem == "source":
            return p
    return None


def delete_job(job_id):
    shutil.rmtree(job_dir(job_id), ignore_errors=True)


def load_all():
    """Return every job's meta dict found on disk."""
    _ensure()
    out = []
    for d in DATA_DIR.iterdir():
        if not d.is_dir():
            continue
        m = read_meta(d.name)
        if m:
            out.append(m)
    return out


def list_done():
    """Recent completed jobs, newest first — payload for the UI."""
    metas = [m for m in load_all() if m.get("status") == "done"]
    metas.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    return [
        {
            "job_id": m["job_id"],
            "filename": m.get("filename"),
            "created_at": m.get("created_at"),
            "language": m.get("language"),
            "model": m.get("model"),
            "stats": m.get("stats"),
        }
        for m in metas
    ]


def purge_expired(retention_s, protected_ids):
    """Delete job dirs older than retention_s. Never touches protected_ids
    (running jobs). Returns the list of removed job_ids."""
    now = time.time()
    removed = []
    for m in load_all():
        jid = m.get("job_id")
        if not jid or jid in protected_ids:
            continue
        if now - m.get("created_at", 0) > retention_s:
            delete_job(jid)
            removed.append(jid)
    return removed
