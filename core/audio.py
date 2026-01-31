import os
import subprocess
from config import Config

class AudioProcessor:
    @staticmethod
    def extract_audio(video_path, output_audio_path=None):
        """Extracts audio track from video using high-fidelity settings."""
        if output_audio_path is None:
            output_audio_path = str(Config.TEMP_DIR / "original_audio.wav")
            
        print(f"🎬 Extracting audio from {video_path}...")
        # 工业级方案：使用 ffmpeg 提取 pcm_s16le 格式，确保后期 ASR 处理最精准
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(output_audio_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Audio extracted to: {output_audio_path}")
        return output_audio_path

    @staticmethod
    def combine_video_audio(video_path, audio_path, output_path):
        """
        Combines video and audio using FFmpeg Stream Copy.
        This is much faster and preserves quality better than moviepy.
        """
        print(f"🎥 Merging audio and video using Stream Copy...")
        
        # 工业级指令：
        # -map 0:v:0 获取原始视频流
        # -map 1:a:0 获取新的配音流
        # -c:v copy 视频流直接拷贝，不重编码
        # -c:a aac -b:a 192k 音频转为高品质 AAC
        # -shortest 确保时长对齐
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy", 
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ Final video saved at: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg merge failed: {e}")
            return None