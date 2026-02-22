import os
import io
import subprocess
import threading
import torch
import whisperx
from docx import Document
from dotenv import load_dotenv

load_dotenv()

# Global model cache — one entry per model name, loaded on first use
_models: dict = {}
_model_lock = threading.Lock()

_device = "cuda" if torch.cuda.is_available() else "cpu"
_compute_type = "float16" if _device == "cuda" else "int8"

MODEL_FAST    = "base"
MODEL_PRECISE = "large-v3"


def _get_model(model_name: str):
    with _model_lock:
        if model_name not in _models:
            _models[model_name] = whisperx.load_model(
                model_name, _device, compute_type=_compute_type
            )
    return _models[model_name]


def convert_to_wav(input_path, output_path):
    """Convert audio file to 16 kHz mono WAV using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def segments_to_text(segments):
    """Convert diarized segments to speaker-labeled plain text."""
    lines = []
    current_speaker = None
    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment["text"].strip()
        if speaker != current_speaker:
            if current_speaker is not None:
                lines.append("")
            lines.append(f"[{speaker}]")
            current_speaker = speaker
        lines.append(text)
    return "\n".join(lines)


def segments_to_docx(segments, filename):
    """Convert diarized segments to DOCX bytes with bold speaker headings."""
    doc = Document()
    doc.add_heading(filename, level=1)
    current_speaker = None
    current_para = None
    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment["text"].strip()
        if speaker != current_speaker:
            para = doc.add_paragraph()
            run = para.add_run(f"[{speaker}]")
            run.bold = True
            current_speaker = speaker
            current_para = doc.add_paragraph(text)
        else:
            current_para.add_run(" " + text)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


def run_transcription(job, wav_path, filename, model_name: str = MODEL_PRECISE):
    """Run full WhisperX pipeline and update job dict in-place."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        job["status"] = "error"
        job["error"] = "HF_TOKEN not set"
        return

    def cancelled():
        return job.get("cancelled", False)

    try:
        job["progress"] = "Loading model..."
        model = _get_model(model_name)

        if cancelled():
            job["status"] = "cancelled"
            return

        job["progress"] = "Transcribing audio..."
        audio = whisperx.load_audio(wav_path)
        result = model.transcribe(audio, batch_size=16)
        language = result["language"]
        job["language"] = language

        if cancelled():
            job["status"] = "cancelled"
            return

        job["progress"] = f"Aligning timestamps..."
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=_device
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, _device,
            return_char_alignments=False,
        )

        if cancelled():
            job["status"] = "cancelled"
            return

        job["progress"] = "Diarizing speakers..."
        diarize_model = whisperx.diarize.DiarizationPipeline(
            token=hf_token, device=_device
        )
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        if cancelled():
            job["status"] = "cancelled"
            return

        job["progress"] = "Generating output files..."
        segments = result["segments"]
        job["txt"] = segments_to_text(segments)
        job["docx"] = segments_to_docx(segments, filename)
        job["status"] = "done"
        job["progress"] = "Complete"

    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
    finally:
        # Clean up temp WAV
        try:
            os.remove(wav_path)
        except OSError:
            pass
