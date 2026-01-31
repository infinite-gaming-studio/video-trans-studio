import os
import subprocess
import requests
import numpy as np
import asyncio
from pathlib import Path
from tqdm import tqdm
from config import Config

# Monkey patch for NumPy 2.0+ compatibility
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

class LipSyncProcessor:
    """
    Modern LipSync Processor using MuseTalk for high-fidelity results.
    Includes automated setup, patching, and robust execution.
    """
    def __init__(self):
        self.repo_url = Config.MUSETALK_REPO_URL
        self.repo_path = Config.BASE_DIR / "MuseTalk"
        self.ckpt_dir = self.repo_path / "models"
        
    def setup(self):
        """Initializes MuseTalk repository and downloads checkpoints."""
        if not self.repo_path.exists():
            print("📥 Cloning MuseTalk repository...")
            subprocess.run(["git", "clone", self.repo_url], check=True)
            self._apply_patches()

        # Ensure directory structure for models
        model_dirs = ["musetalk", "dwpose", "face-parse-bisent", "sd-vae-ft-mse", "whisper"]
        for d in model_dirs:
            (self.ckpt_dir / d).mkdir(parents=True, exist_ok=True)

        self._download_models()

    def _apply_patches(self):
        """Fixes compatibility issues in MuseTalk for modern environments."""
        print("🩹 Applying compatibility patches to MuseTalk...")
        (self.repo_path / "__init__.py").touch()
        
        req_file = self.repo_path / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r') as f:
                lines = f.readlines()
            filtered = [l for l in lines if not any(x in l for x in ["torch", "numpy", "opencv"])]
            with open(req_file, 'w') as f:
                f.writelines(filtered)

    def _download_models(self):
        """Downloads MuseTalk checkpoints if missing."""
        mapping = {
            "musetalk/musetalk.pth": Config.MUSETALK_CHECKPOINTS["musetalk"],
            "dwpose/dw-ll_ucoco_384.pth": Config.MUSETALK_CHECKPOINTS["dwpose"],
            "face-parse-bisent/79999_iter.pth": Config.MUSETALK_CHECKPOINTS["face_detection"],
            "sd-vae-ft-mse/diffusion_pytorch_model.bin": Config.MUSETALK_CHECKPOINTS["sd_vae"],
            "whisper/tiny.pt": Config.MUSETALK_CHECKPOINTS["whisper"]
        }
        
        for rel_path, url in mapping.items():
            dest = self.ckpt_dir / rel_path
            if not dest.exists():
                print(f"📥 Downloading weights: {rel_path}")
                self._download_file(url, dest)

    def _download_file(self, url, dest):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=dest.name) as pbar:
            for data in response.iter_content(chunk_size=1024 * 1024):
                f.write(data)
                pbar.update(len(data))

    async def sync(self, video_path, audio_path, output_path):
        """Executes the lip-sync process using MuseTalk."""
        self.setup()
        
        if not self._has_face(video_path):
            print(f"⚠️ No face detected in {video_path}. Using fallback merge.")
            return self._merge_audio_only(video_path, audio_path, output_path)

        print("👄 Starting MuseTalk LipSync - High Fidelity Mode...")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.repo_path}:{env.get('PYTHONPATH', '')}"

        cmd = [
            "python", str(self.repo_path / "scripts/inference.py"),
            "--inference_config", str(self.repo_path / "configs/inference/test_config.yaml"),
            "--video_path", str(video_path),
            "--audio_path", str(audio_path),
            "--output_path", str(output_path)
        ]
        
        try:
            print(f"🚀 Executing MuseTalk: {' '.join(cmd)}")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                print(f"✅ MuseTalk complete: {output_path}")
                return output_path
            else:
                print(f"❌ MuseTalk failed (code {process.returncode}). Check logs.")
                return self._merge_audio_only(video_path, audio_path, output_path)
        except Exception as e:
            print(f"❌ Error during MuseTalk execution: {e}")
            return self._merge_audio_only(video_path, audio_path, output_path)

    def _has_face(self, video_path):
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        has_face = False
        for _ in range(30):
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
        print("🔄 Falling back to simple FFmpeg audio replacement...")
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
            "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
