import whisper
import torch
import sys

def load_model(model_name="medium", use_gpu=True):
    # Load the model, optionally on GPU
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    print(f"Using device: {device}")
    
    # Instantiate a WhisperModel
    model = whisper.load_model(model_name)
    model.to(device)
    return model

def transcribe_audio(audio_path, model):
    # Load the audio file and transcribe it
    result = model.transcribe(audio_path)
    return result["text"]

def main():
    # Check if an audio path is provided as a command line argument
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <path_to_audio_file>")
        sys.exit(1)

    # Use the first command-line argument as the audio file path
    audio_path = sys.argv[1]
    
    # Choose a model size: tiny, base, small, medium, large
    model_name = "large"

    # Load the model with GPU support if available
    model = load_model(model_name=model_name, use_gpu=True)

    # Transcribe the audio file
    transcription = transcribe_audio(audio_path, model)
    
    output_path = audio_path.rsplit(".", 1)[0] + ".txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"Transcription saved to: {output_path}")

if __name__ == "__main__":
    main()
