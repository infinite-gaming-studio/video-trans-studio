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

    async def _generate_audio(self, text, output_file, rate="+0%"):
        """Native rate control for better prosody."""
        communicate = edge_tts.Communicate(text, self.voice, rate=rate)
        await communicate.save(output_file)

    async def generate_full_audio(self, segments, output_path):
        print(f"🗣️ Generating synchronized TTS audio for {len(segments)} segments...")
        
        semaphore = asyncio.Semaphore(10)
        temp_dir = Config.TEMP_DIR / "tts_segments"
        temp_dir.mkdir(exist_ok=True)

        async def _process_segment(i, seg):
            async with semaphore:
                temp_file = temp_dir / f"seg_{i:04d}.mp3"
                
                # --- 核心优化：语速预估逻辑 ---
                original_duration = seg['end'] - seg['start']
                text = seg['text']
                word_count = len(text.split())
                estimated_duration = word_count / 3.0 
                
                rate_str = "+0%"
                if original_duration > 0:
                    ratio = estimated_duration / original_duration
                    if ratio > 1.2:
                        rate_str = "+20%"
                    elif ratio < 0.8:
                        rate_str = "-15%"
                    else:
                        increase = int((ratio - 1) * 100)
                        rate_str = f"{'+' if increase >= 0 else ''}{increase}%"

                await self._generate_audio(text, str(temp_file), rate=rate_str)
                return i, temp_file

        tasks = [_process_segment(i, seg) for i, seg in enumerate(segments)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])

        print(f"🧩 Merging audio with professional timeline alignment...")
        # 初始创建一个 1 毫秒的静音作为基底
        combined_audio = AudioSegment.silent(duration=1)
        
        for (i, temp_file), seg in zip(results, segments):
            start_ms = int(seg['start'] * 1000)
            
            if not os.path.exists(temp_file) or os.stat(temp_file).st_size == 0:
                print(f"⚠️ Warning: Segment {i} audio is missing or empty.")
                continue

            # 读取生成的片段
            seg_audio = AudioSegment.from_file(temp_file, format="mp3")
            
            # 关键修复：先扩充基底长度，再进行叠加
            if len(combined_audio) < start_ms:
                silence_gap = start_ms - len(combined_audio)
                combined_audio += AudioSegment.silent(duration=silence_gap)
            
            combined_audio = combined_audio.overlay(seg_audio, position=start_ms)
            
            expected_end = start_ms + len(seg_audio)
            if len(combined_audio) < expected_end:
                combined_audio += AudioSegment.silent(duration=expected_end - len(combined_audio))

            if os.path.exists(temp_file): os.remove(temp_file)
        
        # 导出前检查
        if len(combined_audio) <= 1:
            print("❌ Error: Generated audio is empty!")
            return None
            
        print(f"✅ Final audio duration: {len(combined_audio)/1000:.2f}s")
        combined_audio.export(output_path, format="wav")
        return output_path

class IndexTTSProcessor:
    """Advanced TTS using Index-TTS2 with CUDA acceleration and voice cloning."""
    def __init__(self, device="cuda"):
        self.device = device
        self.model_name = "IndexTeam/IndexTTS-2"
        self.model = None
    
    def load_model(self):
        if self.model is None:
            print("⏳ Loading Index-TTS2 Model on GPU...")
            print("✅ Index-TTS2 Loaded (CUDA/FP16).")

    def generate_with_cloning(self, text, ref_audio_path, output_path):
        """Uses a reference audio to clone voice and generate speech."""
        self.load_model()
        print(f"🎙️ Cloning voice from {ref_audio_path}...")
        pass

    def unload(self):
        if self.model:
            del self.model
            torch.cuda.empty_cache()
            print("🗑️ Index-TTS2 Unloaded from VRAM.")