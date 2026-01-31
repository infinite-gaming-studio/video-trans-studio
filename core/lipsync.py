import os
import subprocess
import requests
from tqdm import tqdm
from config import Config

class LipSyncProcessor:
    def __init__(self):
        self.repo_url = "https://github.com/Rudrabha/Wav2Lip.git"
        self.repo_path = Config.BASE_DIR / "Wav2Lip"
        self.model_path = Config.CHECKPOINTS_DIR / "wav2lip_gan.pth"

    def setup(self):
        """Clones Wav2Lip repo and downloads weights if missing."""
        if not self.repo_path.exists():
            print("📥 Cloning Wav2Lip repository...")
            subprocess.run(["git", "clone", self.repo_url], check=True)
            # Add __init__.py to make it a package if needed
            (self.repo_path / "__init__.py").touch()
            
            # Wav2Lip needs a fix for newer librosa versions
            # We'll handle that in Colab notebook or here if needed.
            
        if not self.model_path.exists():
            print("📥 Downloading Wav2Lip-GAN weights...")
            self._download_file(Config.WAV2LIP_GAN_WEIGHTS_URL, self.model_path)

    def _download_file(self, url, dest):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(str(dest))) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))

    async def sync(self, video_path, audio_path, output_path):
        """Executes the lip-sync process using Wav2Lip."""
        self.setup()
        
        # 1. 健壮性检查：确保视频中包含人脸
        if not self._has_face(video_path):
            print(f"⚠️ Warning: No face detected in {video_path}. Skipping LipSync.")
            # 如果没有检测到人脸，直接复制原视频或仅替换音频（视需求而定）
            # 这里简单起见，我们返回 None，主流程可以决定是跳过还是降级处理
            # 为了保证流程不中断，我们可以尝试用 ffmpeg 直接合并音频
            print("🔄 Fallback: Merging audio without lip-sync...")
            self._merge_audio_only(video_path, audio_path, output_path)
            return output_path

        print("👄 Starting LipSync (Wav2Lip) - This might take a while on T4...")
        
        cmd = [
            "python", "Wav2Lip/inference.py",
            "--checkpoint_path", str(self.model_path),
            "--face", str(video_path),
            "--audio", str(audio_path),
            "--outfile", str(output_path),
            "--pads", "0", "20", "0", "0"
        ]
        
        try:
            # 使用 asyncio 执行子进程，使其非阻塞
            import asyncio
            print(f"🐛 Debug executing command: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(Config.BASE_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 等待完成并获取输出
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                print(f"✅ LipSync complete: {output_path}")
                return output_path
            else:
                print(f"❌ LipSync failed with return code {process.returncode}")
                print(f"🔍 Error Output:\n{stderr.decode()}")
                print(f"📜 Standard Output:\n{stdout.decode()}")
                return None
        except Exception as e:
            print(f"❌ LipSync failed: {e}")
            return None

    def _has_face(self, video_path):
        """Checks if at least one face exists in the video using OpenCV."""
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        has_face = False
        # Check first 30 frames (1 second approx) to save time
        frame_count = 0
        while frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                has_face = True
                break
            frame_count += 1
        
        cap.release()
        return has_face

    def _merge_audio_only(self, video_path, audio_path, output_path):
        """Merges audio with video using ffmpeg, without lip-sync."""
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
