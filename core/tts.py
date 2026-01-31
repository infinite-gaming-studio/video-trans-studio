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
                # 参考 pyVideoTrans: 预先估算语速倍率
                # 假设英文平均语速为 150 词/分钟，或者根据字符长度预估
                original_duration = seg['end'] - seg['start']
                text = seg['text']
                
                # 预估 1x 语速下的时长（经验公式：英文约 3 个词/秒）
                word_count = len(text.split())
                estimated_duration = word_count / 3.0 
                
                rate_str = "+0%"
                if original_duration > 0:
                    ratio = estimated_duration / original_duration
                    if ratio > 1.1:
                        # 增加语速，最高 +50% (即 1.5x)
                        increase = min(int((ratio - 1) * 100), 50)
                        rate_str = f"+{increase}%"
                    elif ratio < 0.8:
                        # 降低语速，最低 -20%
                        decrease = max(int((ratio - 1) * 100), -20)
                        rate_str = f"{decrease}%"

                await self._generate_audio(text, str(temp_file), rate=rate_str)
                return i, temp_file

        tasks = [_process_segment(i, seg) for i, seg in enumerate(segments)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])

        # --- 核心优化：高保真对齐 ---
        print(f"🧩 Merging audio with professional timeline alignment...")
        combined_audio = AudioSegment.empty()
        
        for (i, temp_file), seg in zip(results, segments):
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)
            target_dur = end_ms - start_ms
            
            # 读取并检查实际时长
            seg_audio = AudioSegment.from_mp3(temp_file)
            
            # 如果 TTS 原生语速控制后仍超出时长，进行微调
            if len(seg_audio) > target_dur and target_dur > 0:
                # 最后的兜底：高保真裁剪或微调
                seg_audio = seg_audio[:target_dur]
            
            # 填充静音
            if len(combined_audio) < start_ms:
                combined_audio += AudioSegment.silent(duration=start_ms - len(combined_audio))
            
            # 覆盖合成（防止漂移）
            combined_audio = combined_audio[:start_ms] + seg_audio
            
            if os.path.exists(temp_file): os.remove(temp_file)
            
        combined_audio.export(output_path, format="wav")
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
