# Plan: Disk-backed persistent transcription results

**Status:** Planning only — not yet implemented. Hand-off for a separate session.
**Goal:** Survive page refresh *and* the hourly GPU-idle self-restart by persisting
results to disk. Show a "Recent transcriptions" list, purge entries after 24h, and
keep the original audio so a stored job can be re-run with a different model/language.

## Why disk (not just in-memory or localStorage)

- `currentJobId` is browser-only JS state (`index.html:321`) → lost on refresh.
- `JOBS` is an in-memory dict (`app.py:21`) → wiped by the GPU-idle `os.execv`
  restart (`app.py:43-57`), which fires ~hourly. So in-memory alone can't reach 24h.
- `job["txt"]`/`job["docx"]` live only in RAM (`app.py:117-118`, set in
  `transcriber.py:163-164`) → must be written to disk.
- Downloads already come from the server, so disk is the natural source of truth.

## Decision on keeping audio

KEEP the original upload. The app has a model toggle (precise/fast) and a
retranscribe flow, but retranscribe currently re-uploads the client-side
`selectedFile` and dies on refresh. Storing the source server-side lets a
restored/recent job be re-run with a different model or language without the
original file. (~1 MB/min for typical mp3/m4a; 604 GB free — space is not a concern.)

## Storage layout

```
/home/transcription/app/data/jobs/<job_id>/
  meta.json          # {job_id, filename, created_at, status, language, model, stats, error}
  transcript.txt
  transcript.docx
  source<ext>        # original uploaded audio (e.g. source.m4a)
```

- `created_at`: epoch seconds (used for 24h purge and recent-list sort).
- Write `meta.json` atomically (temp file + `os.replace`).
- Only `status == "done"` jobs appear in the recent list.

## Backend changes (`app.py`)

1. **Dirs/config:** `DATA_DIR = Path(__file__).parent / "data" / "jobs"`, `mkdir`.
   New env `RESULT_RETENTION_HOURS` (default `24`).
2. **Job schema:** add `created_at` (set at upload) and `model` fields.
3. **Persistence helpers:**
   - `_job_dir(job_id)`
   - `_persist_meta(job_id, job)` — atomic write of meta.json
   - `_persist_outputs(job_id, job)` — write transcript.txt / transcript.docx
   - `_save_source(job_id, raw_path, suffix)` — move original upload into job dir
   - `_load_jobs_from_disk()` — on startup, scan `DATA_DIR`, rebuild `JOBS` from
     each meta.json (do NOT load txt/docx into RAM; downloads read from disk).
     Skip dirs with missing/corrupt meta.json. Drop non-`done` leftovers.
4. **Upload flow:** save the original file into the job dir (keep it); convert a
   *temp* WAV in `/tmp` for processing (the temp WAV stays disposable — current
   `transcriber.py` deletion behavior is fine for it). After the worker thread
   finishes, if `done`: call `_persist_outputs` + `_persist_meta`; may drop
   txt/docx from the in-memory dict to save RAM.
5. **Downloads:** change `/download/<job_id>/txt|docx` to read files from the job
   dir (so they work after a restart when RAM has no txt/docx).
6. **New `GET /recent`:** list `done` jobs sorted by `created_at` desc →
   `[{job_id, filename, created_at, language, model, stats}]`.
7. **New `POST /retranscribe/<job_id>`:** body `{model?, language?}`. Reconvert the
   stored `source<ext>` to a temp WAV, create a NEW job_id from it, run the
   pipeline. This is the capability that justifies keeping audio.
8. **Purge:** extend `_idle_monitor` (already ticks every 60s, `app.py:43`) — or a
   dedicated sweeper — to `shutil.rmtree` job dirs older than
   `RESULT_RETENTION_HOURS` and drop them from `JOBS`. MUST skip `status ==
   "running"` jobs (mirror the existing idle-monitor guard) so a long
   transcription is never deleted mid-run.

## Frontend changes (`templates/index.html`)

1. On load, `GET /recent` and render a "Recent transcriptions" list: filename,
   relative time, language badge, model badge, plus Download TXT / DOCX links.
2. Clicking a recent item restores the session: set `currentJobId`, point the
   download buttons at it, show its stats (via `/status` or the recent payload).
3. Add a retranscribe affordance on a restored job (model toggle + language select)
   that POSTs to `/retranscribe/<job_id>` — no client-side file needed anymore.
4. Server is the source of truth; no localStorage required.

## Edge cases / notes

- Atomic meta.json writes (temp + rename) to avoid torn reads during the 60s sweep.
- Startup loader must tolerate partial/corrupt job dirs.
- Purge guard: never delete a `running` job regardless of age.
- After this, the GPU-idle `os.execv` restart is safe — `_load_jobs_from_disk()`
  repopulates the recent list.
- `.gitignore`: add `data/` (audio + transcripts must not be committed).

## Touch list

- `app.py` — dirs/config, schema, persistence helpers, upload/download wiring,
  `/recent`, `/retranscribe`, purge in idle monitor, startup load.
- `transcriber.py` — minimal; ensure it only deletes the disposable temp WAV, not
  the persisted source (current behavior already deletes only the path passed in).
- `templates/index.html` — recent list UI, restore + server-side retranscribe.
- `.gitignore` — add `data/`.
- `memory/project_structure.md` — update endpoints, job schema, storage section.
```
