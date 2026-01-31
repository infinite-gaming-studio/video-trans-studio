#!/bin/bash
echo "🚀 Starting Video Trans Studio - MuseTalk & Index-TTS2 Edition Setup..."

# 1. Update and install system dependencies
echo "📦 Installing system dependencies (ffmpeg)..."
apt-get update -qq && apt-get install -y ffmpeg -qq

# 2. Deep Clean Python Environment
echo "🧹 Cleaning up existing packages to prevent conflicts..."
pip uninstall -y transformers tokenizers protobuf librosa numpy -q

# 3. Install Modern Python Infrastructure
echo "🐍 Installing modern AI libraries..."
pip install --no-cache-dir torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118 -q
pip install --no-cache-dir -r requirements.txt -q

# 4. Setup Repositories
if [ ! -d "MuseTalk" ]; then
    echo "📥 Cloning MuseTalk repository..."
    git clone https://github.com/TMElyralab/MuseTalk.git -q
    touch MuseTalk/__init__.py
fi

if [ ! -d "index-tts" ]; then
    echo "📥 Cloning Index-TTS2 repository..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/index-tts/index-tts.git -q
fi

# 5. Create directory structure
mkdir -p checkpoints output temp

echo "✅ Environment Setup Complete! No legacy Wav2Lip dependencies remaining."
