#!/bin/bash
echo "🚀 Starting Video Trans Studio Colab Setup..."

# 1. Update and install system dependencies
echo "📦 Installing system dependencies (ffmpeg)..."
apt-get update -qq && apt-get install -y ffmpeg -qq

# 2. Install Python requirements
echo "🐍 Installing Python libraries..."
pip install -r requirements.txt -q
# Wav2Lip often breaks with newer librosa, forcing 0.8.0 is a known fix
pip install librosa==0.8.0 -q

# 3. Setup Wav2Lip
if [ ! -d "Wav2Lip" ]; then
    echo "📥 Cloning Wav2Lip repository..."
    git clone https://github.com/Rudrabha/Wav2Lip.git -q
fi

# Fixes for Wav2Lip (Python 3.10+ compatibility)
touch Wav2Lip/__init__.py
# The following fix is often needed for 'collections' vs 'collections.abc'
find Wav2Lip -name "*.py" -exec sed -i 's/from collections import Iterable/from collections.abc import Iterable/g' {} +

# 4. Create directory structure
mkdir -p checkpoints output temp

# 5. Download Model Weights
if [ ! -f "checkpoints/wav2lip_gan.pth" ]; then
    echo "📥 Downloading Wav2Lip-GAN weights..."
    wget "https://huggingface.co/goutham79/Wav2Lip-GAN/resolve/main/checkpoints/Wav2Lip_GAN.pth" -O checkpoints/wav2lip_gan.pth -q
fi

# Pre-download NLLB model to cache
echo "📥 Pre-downloading NLLB-200 Translation model..."
python3 -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')"

echo "✅ Setup Complete! Ready to process videos."
echo "💡 Usage: python main.py your_video.mp4 zh-cn"
