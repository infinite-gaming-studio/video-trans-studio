import os
import torch
from pathlib import Path

class Config:
    # Base Paths
    BASE_DIR = Path(__file__).parent.absolute()
    TEMP_DIR = BASE_DIR / "temp"
    OUTPUT_DIR = BASE_DIR / "output"
    CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
    
    # Create dirs if not exist
    TEMP_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)

    # Hardware Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    
    # Model Configurations
    WHISPER_MODEL_SIZE = "large-v3"
    WHISPER_COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
    
    # LivePortrait Configuration (Next-Gen Face Reenactment)
    LIVEPORTRAIT_REPO_URL = "https://github.com/KwaiVGI/LivePortrait.git"
    LIVEPORTRAIT_CHECKPOINTS = {
        "base": "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/base_models",
        "landmark": "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/landmark.pth"
    }

    # Index-TTS2 Configuration (Modern Voice Cloning)
    INDEXTTS_REPO_URL = "https://github.com/index-tts/index-tts.git"
    INDEXTTS_MODEL_DIR = CHECKPOINTS_DIR / "IndexTTS-2"
    INDEXTTS_CONFIG_PATH = INDEXTTS_MODEL_DIR / "config.yaml"
    INDEXTTS_MODELS = {
        "config.yaml": "https://huggingface.co/IndexTeam/IndexTTS-2/resolve/main/config.yaml",
        "model.safetensors": "https://huggingface.co/IndexTeam/IndexTTS-2/resolve/main/model.safetensors",
        "vocab.json": "https://huggingface.co/IndexTeam/IndexTTS-2/resolve/main/vocab.json"
    }

    @classmethod
    def print_info(cls):
        print(f"✅ Running on: {cls.DEVICE.upper()}")
        if cls.DEVICE == "cuda":
            print(f"🚀 GPU: {cls.GPU_NAME}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"💾 VRAM: {vram:.2f} GB")
        print(f"📂 Output Dir: {cls.OUTPUT_DIR}")

if __name__ == "__main__":
    Config.print_info()