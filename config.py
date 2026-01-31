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
    WHISPER_MODEL_SIZE = "large-v3" # Fits on T4 easily
    WHISPER_COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
    
    # TTS Configuration
    TTS_VOICE = "en-US-ChristopherNeural" # Default Edge-TTS voice
    
    # Wav2Lip Configuration
    WAV2LIP_BATCH_SIZE = 128 # T4 can handle larger batches
    WAV2LIP_GAN_WEIGHTS_URL = "https://huggingface.co/goutham79/Wav2Lip-GAN/resolve/main/checkpoints/Wav2Lip_GAN.pth"
    
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
