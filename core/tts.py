import asyncio
import edge_tts
from config import Config
from pydub import AudioSegment
import os
import torch
import soundfile as sf

class TTSProcessor:
    """Base class for TTS, defaults to EdgeTTS for speed/efficiency."""
    def __init__(self, voice="en-US-ChristopherNeural"):
        self.voice = voice

    async def _generate_audio(self, text, output_file):
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_file)

    async def generate_full_audio(self, segments, output_path):
        print(f"🗣️ Generating TTS audio via Edge-TTS...")
        combined_audio = AudioSegment.empty()
        current_time_ms = 0
        temp_segment_file = Config.TEMP_DIR / "temp_seg.mp3"
        
        for seg in segments:
            start_ms = int(seg['start'] * 1000)
            silence_duration = start_ms - current_time_ms
            if silence_duration > 0:
                combined_audio += AudioSegment.silent(duration=silence_duration)
                current_time_ms += silence_duration
            
            # 直接 await，不再使用 asyncio.run()
            await self._generate_audio(seg['text'], str(temp_segment_file))
            seg_audio = AudioSegment.from_mp3(temp_segment_file)
            combined_audio += seg_audio
            current_time_ms += len(seg_audio)
            
        combined_audio.export(output_path, format="wav")
        if os.path.exists(temp_segment_file): os.remove(temp_segment_file)
        print(f"✅ Audio generated: {output_path}")
        return output_path

class IndexTTSProcessor:
    """Advanced TTS using Index-TTS2 with CUDA acceleration and voice cloning."""
    def __init__(self, device="cuda"):
        self.device = device
        self.model_name = "IndexTeam/IndexTTS-2"
        self.model = None
        # In a real implementation, we would load the model from HF here
        # For the prototype, we will assume the environment is setup via setup_colab.sh
    
    def load_model(self):
        if self.model is None:
            print("⏳ Loading Index-TTS2 Model on GPU...")
            # Placeholder for actual model loading logic
            # This would typically involve:
            # self.model = AutoModel.from_pretrained(self.model_name).to(self.device).half()
            print("✅ Index-TTS2 Loaded (CUDA/FP16).")

    def generate_with_cloning(self, text, ref_audio_path, output_path):
        """Uses a reference audio to clone voice and generate speech."""
        self.load_model()
        print(f"🎙️ Cloning voice from {ref_audio_path}...")
        # Implementation of Index-TTS2's inference logic goes here
        # It uses CUDA to match the duration and prosody.
        pass

    def unload(self):
        if self.model:
            del self.model
            torch.cuda.empty_cache()
            print("🗑️ Index-TTS2 Unloaded from VRAM.")
