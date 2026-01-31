import os
import subprocess
import requests
import asyncio
import sys
from pathlib import Path
from tqdm import tqdm
from config import Config

class LipSyncProcessor:
    """
    Modern LipSync Processor exclusively using MuseTalk.
    Clean, robust, and optimized for modern AI environments.
    """
    def __init__(self):
        self.repo_path = Config.BASE_DIR / "MuseTalk"
        self.ckpt_dir = self.repo_path / "models"
        
    def setup(self):
        """Initializes MuseTalk and downloads weights via official hub if missing."""
        if not self.repo_path.exists():
            print("📥 MuseTalk repository not found. Cloning...")
            subprocess.run(["git", "clone", Config.MUSETALK_REPO_URL], check=True)
            (self.repo_path / "__init__.py").touch()

        self._download_models()

    def _download_models(self):
        """Downloads MuseTalk checkpoints using huggingface_hub for robustness."""
        print("📥 Checking MuseTalk models...")
        from huggingface_hub import snapshot_download
        
        try:
            # We download directly into the repo's model structure
            snapshot_download(
                repo_id="TMElyralab/MuseTalk",
                local_dir=self.repo_path,
                local_dir_use_symlinks=False,
                allow_patterns=["musetalk/*", "dwpose/*", "models/face-parse-bisent/*", "sd-vae-ft-mse/*", "whisper/*"]
            )
            print("✅ MuseTalk models ready.")
        except Exception as e:
            print(f"❌ Error downloading models: {e}")
            raise

    async def sync(self, video_path, audio_path, output_path):
        """Executes the lip-sync process using MuseTalk."""
        self.setup()
        
        if not self._has_face(video_path):
            print(f"⚠️ No face detected. Falling back to simple merge.")
            return self._merge_audio_only(video_path, audio_path, output_path)

        print("👄 Starting MuseTalk High-Fidelity LipSync...")
        
        # Prepare Environment
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.repo_path}:{env.get('PYTHONPATH', '')}"

        # MuseTalk Inference
        cmd = [
            sys.executable, str(self.repo_path / "scripts/inference.py"),
            "--inference_config", str(self.repo_path / "configs/inference/test_config.yaml"),
            "--video_path", str(video_path),
            "--audio_path", str(audio_path),
            "--output_path", str(output_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                print(f"✅ LipSync Complete: {output_path}")
                return output_path
            else:
                print(f"❌ MuseTalk failed. Stderr:\n{stderr.decode()}")
                return self._merge_audio_only(video_path, audio_path, output_path)
        except Exception as e:
            print(f"❌ MuseTalk Runtime Error: {e}")
            return self._merge_audio_only(video_path, audio_path, output_path)

    def _has_face(self, video_path):
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        has_face = False
        for _ in range(30): # Check 1st second
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                has_face = True
                break
        cap.release()
        return has_face

    def _merge_audio_only(self, video_path, audio_path, output_path):
        print("🔄 FFmpeg Fallback: Syncing audio without face animation...")
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
            "-shortest", str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path