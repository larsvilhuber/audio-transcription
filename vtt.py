"""Parse WebVTT transcripts (e.g. Zoom) into speaker-merged segments.

Stdlib only: this is imported by the Flask process and must stay lightweight.
"""
import html
import re

_VOICE_RE = re.compile(r"<v(?:\.[^\s>]*)?\s+([^>]+)>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]*>")
_LABEL_RE = re.compile(r"^([^:\n]{1,60}):\s+(.*)", re.DOTALL)


def _clean(text):
    text = html.unescape(_TAG_RE.sub("", text))
    return " ".join(text.split())


def parse_vtt(text: str) -> list[dict]:
    """Return [{"speaker", "text"}], one per consecutive same-speaker run."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")

    # Collect (voice_speaker, raw_text) for each cue.
    cues = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip("\n").split("\n")
        if not lines or not lines[0].strip():
            continue
        first = lines[0].strip()
        if first.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        timing = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if timing is None:
            continue
        raw = "\n".join(lines[timing + 1:])
        voice = _VOICE_RE.search(raw)
        cues.append((html.unescape(voice.group(1)).strip() if voice else None, raw))

    # Zoom-style "Name: text" labels only count if most cues use them, so
    # ordinary sentences containing colons aren't mangled.
    plain = [_clean(raw) for _, raw in cues]
    labelled = sum(1 for (v, _), p in zip(cues, plain) if v is None and _LABEL_RE.match(p))
    use_labels = labelled > len(cues) / 2

    segments = []
    speaker = None
    for (voice, _raw), body in zip(cues, plain):
        if voice:
            speaker = voice
        elif use_labels:
            m = _LABEL_RE.match(body)
            if m:
                speaker, body = m.group(1).strip(), m.group(2)
        body = body.strip()
        if not body:
            continue
        who = speaker or "UNKNOWN"
        if segments and segments[-1]["speaker"] == who:
            segments[-1]["text"] += " " + body
        else:
            segments.append({"speaker": who, "text": body})
    return segments
