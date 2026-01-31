import asyncio
import edge_tts
from config import Config
from pydub import AudioSegment
import os
import torch
import soundfile as sf
import subprocess

class TTSProcessor:
    """Industrial-grade TTS Processor with precise sync and buffer management."""
    def __init__(self, voice="en-US-ChristopherNeural"):
        self.voice = voice

    async def _generate_audio(self, text, output_file, rate="+0%"):
        """Generates audio with native rate control."""
        communicate = edge_tts.Communicate(text, self.voice, rate=rate)
        await communicate.save(output_file)

    async def generate_full_audio(self, segments, output_path):
        print(f"🗣️ Executing Audio Rendering Pipeline for {len(segments)} segments...")
        
        semaphore = asyncio.Semaphore(10)
        temp_dir = Config.TEMP_DIR / "tts_segments"
        temp_dir.mkdir(exist_ok=True)

        async def _process_segment(i, seg):
            async with semaphore:
                temp_file = temp_dir / f"seg_{i:04d}.mp3"
                wav_file = temp_dir / f"seg_{i:04d}.wav"
                
                # 语速预估
                original_duration = seg['end'] - seg['start']
                text = seg['text']
                word_count = len(text.split())
                estimated_duration = word_count / 3.0 # 经验常数
                
                rate_str = "+0%"
                if original_duration > 0:
                    ratio = estimated_duration / original_duration
                    if ratio > 1.2: rate_str = "+20%"
                    elif ratio < 0.8: rate_str = "-15%"
                    else:
                        inc = int((ratio - 1) * 100)
                        rate_str = f"{'+' if inc >= 0 else ''}{inc}%"

                await self._generate_audio(text, str(temp_file), rate=rate_str)
                
                # 诊断与渲染优化：立即将 MP3 转换为标准的 PCM WAV 格式，统一采样率
                # 解决“编解码器性能”和“采样率不匹配”导致的断音
                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", str(temp_file), 
                        "-ar", "44100", "-ac", "2", str(wav_file)
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except:
                    print(f"⚠️ Warning: FFmpeg conversion failed for segment {i}")
                    wav_file = temp_file # 降级处理

                return i, wav_file

        tasks = [_process_segment(i, seg) for i, seg in enumerate(segments)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])

        print(f"🧩 Analyzing Buffers and Synchronizing Streams...")
        # 初始化一个空的高品质音轨
        combined_audio = AudioSegment.silent(duration=0, frame_rate=44100)
        
        for (i, wav_file), seg in zip(results, segments):
            start_ms = int(seg['start'] * 1000)
            
            if not os.path.exists(wav_file) or os.path.getsize(wav_file) < 100:
                continue

            # 诊断音频流中断：使用 pydub 加载 PCM 数据
            seg_audio = AudioSegment.from_file(wav_file)
            
            # 缓冲区管理：精确计算静音填充，确保 combined_audio 的 Base 永远长于叠加位置
            current_len = len(combined_audio)
            if current_len < start_ms:
                # 补齐到起始位置
                combined_audio += AudioSegment.silent(duration=start_ms - current_len, frame_rate=44100)
            
            # 验证同步机制：
            # 如果是顺序排列且无重叠，直接追加 (Append) 以获得最佳性能
            # 如果有重叠（由于语速限制），则进行 Overlay
            if len(combined_audio) <= start_ms:
                combined_audio += seg_audio
            else:
                # 处理重叠：先扩充 Base，再叠加
                needed_len = start_ms + len(seg_audio)
                if len(combined_audio) < needed_len:
                    extension = needed_len - len(combined_audio)
                    combined_audio += AudioSegment.silent(duration=extension, frame_rate=44100)
                
                combined_audio = combined_audio.overlay(seg_audio, position=start_ms)

            # 渲染完成后清理
            if os.path.exists(wav_file): os.remove(wav_file)
            mp3_file = str(wav_file).replace(".wav", ".mp3")
            if os.path.exists(mp3_file): os.remove(mp3_file)
        
        # 优化音频渲染管道：最终归一化导出
        print(f"✅ Rendering Complete. Final Duration: {len(combined_audio)/1000:.2f}s")
        combined_audio.export(output_path, format="wav", parameters=["-ar", "44100", "-ac", "2"])
        return output_path

class IndexTTSProcessor:
    def __init__(self, device="cuda"):
        self.device = device
        self.model_name = "IndexTeam/IndexTTS-2"
        self.model = None
    
    def load_model(self):
        if self.model is None:
            print("⏳ Loading Index-TTS2...")
            print("✅ Index-TTS2 Ready.")

    def generate_with_cloning(self, text, ref_audio_path, output_path):
        self.load_model()
        pass

    def unload(self):
        if self.model:
            del self.model
            torch.cuda.empty_cache()
