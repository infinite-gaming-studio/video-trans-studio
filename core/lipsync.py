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
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(Config.BASE_DIR)
            )
            await process.wait()
            
            if process.returncode == 0:
                print(f"✅ LipSync complete: {output_path}")
                return output_path
            else:
                print(f"❌ LipSync failed with return code {process.returncode}")
                return None
        except Exception as e:
            print(f"❌ LipSync failed: {e}")
            return None
