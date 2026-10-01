# Versioning

The app version lives in one place: `version` in `pyproject.toml` (also mirrored in the
`audio-transcription` entry of `uv.lock`). `app.py` reads it at startup and the page footer shows it.

Follow semantic versioning (MAJOR.MINOR.PATCH). Bump the version in the same commit as the change:

- **PATCH** (1.2.0 -> 1.2.1): bug fixes, style tweaks, docs, dependency bumps with no behavior change.
- **MINOR** (1.2.x -> 1.3.0): new backwards-compatible features (new endpoint, new accepted file type, new UI feature). Reset PATCH to 0.
- **MAJOR** (1.x.y -> 2.0.0): breaking changes (removed/renamed API endpoints, changed job/storage schema that invalidates existing data, changed env vars or deployment requirements).
  **Never bump MAJOR without asking the user first.** If a change looks breaking, stop and check with them.

When bumping, update both `pyproject.toml` and the `uv.lock` entry (or run `uv lock`).
Deploying a new version = restart the service.
