#!/bin/bash
echo "🚀 Starting Video Trans Studio Colab Setup..."

# 1. Update and install system dependencies
echo "📦 Installing system dependencies (ffmpeg)..."
apt-get update -qq && apt-get install -y ffmpeg -qq

# 2. Install Python requirements
echo "🐍 Installing Python libraries..."
# Force reinstall transformers to fix potential corruption
pip install --upgrade --force-reinstall transformers -q
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118 -q
pip install accelerate sentencepiece deep-translator -q
pip install -r requirements.txt -q

# Fix for NLLB model loading in some environments
pip install protobuf==3.20.3 -q

# 3. Setup Wav2Lip and Index-TTS2
if [ ! -d "Wav2Lip" ]; then
    echo "📥 Cloning Wav2Lip repository..."
    git clone https://github.com/Rudrabha/Wav2Lip.git -q
fi

if [ ! -d "index-tts" ]; then
    echo "📥 Cloning Index-TTS2 repository (skipping large files)..."
    # Skip LFS to avoid budget exceeded errors
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/index-tts/index-tts.git -q
fi

# Fixes for Wav2Lip (Python 3.10+ compatibility)
touch Wav2Lip/__init__.py
find Wav2Lip -name "*.py" -exec sed -i 's/from collections import Iterable/from collections.abc import Iterable/g' {} +

# 4. Create directory structure
mkdir -p checkpoints output temp

# 5. Download Model Weights
echo "📥 Downloading AI model weights (Wav2Lip, NLLB, Index-TTS)..."
if [ ! -f "checkpoints/wav2lip_gan.pth" ]; then
    wget "https://huggingface.co/goutham79/Wav2Lip-GAN/resolve/main/checkpoints/Wav2Lip_GAN.pth" -O checkpoints/wav2lip_gan.pth -q
fi

# Pre-download NLLB model
python3 -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')"

# Note: Index-TTS2 weights are large and usually handled by its own script or Git-LFS
# We will ensure the core dependencies are ready.
pip install vocos einops vector_quantize_pytorch -q

echo "✅ Setup Complete! Ready to process videos."
echo "💡 Usage: python main.py your_video.mp4 zh-cn"
