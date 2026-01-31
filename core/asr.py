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
        
        # 优化参数：增加 word_timestamps 和更精细的 vad 控制
        segments, info = self.model.transcribe(
            audio_path, 
            beam_size=5, 
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=True,  # 开启词级时间戳，方便后续精细化处理
            initial_prompt="以下是普通话，请加标点符号。", # 强制要求带标点，有助于断句
        )
        
        result_segments = []
        for segment in segments:
            # 如果单句太长（比如超过 10 秒），在这里可以做进一步的逻辑分割
            # 目前先进行基础清理
            text = segment.text.strip()
            if not text:
                continue
                
            result_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": text
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
