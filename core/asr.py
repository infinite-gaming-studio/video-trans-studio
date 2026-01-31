import gc
import torch
from faster_whisper import WhisperModel
from config import Config

class ASRProcessor:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            print(f"⏳ Loading Whisper Model ({Config.WHISPER_MODEL_SIZE})...")
            self.model = WhisperModel(
                Config.WHISPER_MODEL_SIZE, 
                device=Config.DEVICE, 
                compute_type=Config.WHISPER_COMPUTE_TYPE
            )
            print("✅ Whisper Model Loaded.")

    def transcribe(self, audio_path):
        self.load_model()
        print(f"🎙️ Transcribing: {audio_path}...")
        
        segments, info = self.model.transcribe(
            audio_path, 
            beam_size=5, 
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Convert generator to list to ensure processing is done
        result_segments = []
        for segment in segments:
            result_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            
        print(f"✅ Transcription complete. Detected language: {info.language}")
        return result_segments

    def unload(self):
        """Free up VRAM for the next step."""
        if self.model:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            print("🗑️ Whisper Model Unloaded.")
