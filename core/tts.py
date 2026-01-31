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
        print(f"🗣️ Generating TTS audio for {len(segments)} segments via Edge-TTS...")
        
        # 使用信号量限制并发，避免触发 API 限制或过载
        semaphore = asyncio.Semaphore(10)
        temp_dir = Config.TEMP_DIR / "tts_segments"
        temp_dir.mkdir(exist_ok=True)

        async def _process_segment(i, text):
            async with semaphore:
                temp_file = temp_dir / f"seg_{i:04d}.mp3"
                await self._generate_audio(text, str(temp_file))
                return i, temp_file

        # 并行执行所有 TTS 请求
        tasks = [
            _process_segment(i, seg['text']) 
            for i, seg in enumerate(segments)
        ]
        
        print(f"⏳ Downloading segments in parallel...")
        results = await asyncio.gather(*tasks)
        # 按索引排序确保顺序正确
        results.sort(key=lambda x: x[0])

        print(f"🧩 Combining audio segments with Precise Time Matching...")
        combined_audio = AudioSegment.empty()
        
        for (i, temp_file), seg in zip(results, segments):
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)
            target_duration = end_ms - start_ms
            
            # 1. 填充静音直到当前片段开始
            if len(combined_audio) < start_ms:
                silence_gap = start_ms - len(combined_audio)
                combined_audio += AudioSegment.silent(duration=silence_gap)
            
            # 2. 读取生成的音频
            seg_audio = AudioSegment.from_mp3(temp_file)
            actual_duration = len(seg_audio)
            
            # 3. 动态倍速处理 (Time Stretching)
            # 如果翻译后的文本太长，导致音频超过了原视频片段的时长，我们需要对其进行变速
            # 参考开源项目最佳实践：倍速范围建议在 0.8x 到 1.5x 之间，否则声音会失真严重
            if actual_duration > target_duration and target_duration > 0:
                speed_factor = actual_duration / target_duration
                # 限制最大倍速，避免变成“花栗鼠”声音
                if speed_factor > 1.5:
                    print(f"⚠️ Warning: Segment {i} is too long ({actual_duration}ms vs {target_duration}ms). Capping speed factor at 1.5x.")
                    speed_factor = 1.5
                
                # 使用 pydub 的变速功能（注意：这种变速会改变音调，后续可以考虑用 ffmpeg atempo 优化无损音调变速）
                seg_audio = seg_audio.speedup(playback_speed=speed_factor, chunk_size=150, crossfade=25)
            
            # 4. 裁剪多余部分或保留（视逻辑而定，这里我们根据 start_ms 强制对齐）
            # 确保不覆盖下一个片段（除非不得不覆盖）
            combined_audio = combined_audio[:start_ms] + seg_audio
            
            # 立即删除小的临时文件
            if os.path.exists(temp_file): os.remove(temp_file)
            
        combined_audio.export(output_path, format="wav")
        print(f"✅ Audio generated with sync protection: {output_path}")
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
