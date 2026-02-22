import sys
import os
import torch
import whisperx
from dotenv import load_dotenv

load_dotenv()

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <path_to_audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not set in environment or .env file")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Using device: {device}")

    print("Loading model...")
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

    print("Transcribing...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=4)
    language = result["language"]
    print(f"Detected language: {language}")

    print("Aligning timestamps...")
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)

    print("Diarizing speakers...")
    diarize_model = whisperx.diarize.DiarizationPipeline(token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    print("Writing output...")
    output_path = audio_path.rsplit(".", 1)[0] + ".txt"
    with open(output_path, "w", encoding="utf-8") as f:
        current_speaker = None
        for segment in result["segments"]:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment["text"].strip()
            if speaker != current_speaker:
                if current_speaker is not None:
                    f.write("\n")
                f.write(f"[{speaker}]\n")
                current_speaker = speaker
            f.write(f"{text}\n")

    print(f"Transcription saved to: {output_path}")

if __name__ == "__main__":
    main()
