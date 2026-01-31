#!/bin/bash
echo "🚀 Starting Video Trans Studio - MuseTalk & Index-TTS2 Edition Setup..."

# 1. Update and install system dependencies
echo "📦 Installing system dependencies (ffmpeg)..."
apt-get update -qq && apt-get install -y ffmpeg -qq

# 2. Deep Clean Python Environment
echo "🧹 Cleaning up existing packages to prevent conflicts..."
pip uninstall -y transformers tokenizers protobuf librosa numpy jax -q

# 3. Install Modern Python Infrastructure
echo "🐍 Installing modern AI libraries (Force Upgrade)..."
pip install --no-cache-dir torch torchaudio torchvision -q
pip install --no-cache-dir "transformers>=4.46.0" "tokenizers>=0.20" "numpy>=2.0.0,<2.1.0" -q
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

# 4.5 Install MuseTalk & Index-TTS2 Specific Dependencies
echo "📦 Installing MuseTalk and Index-TTS2 dependencies..."
pip install --no-cache-dir diffusers face-alignment -q

# Fix MuseTalk requirements: Remove pinned versions that conflict with Python 3.12 or Numpy 2.0
if [ -f "MuseTalk/requirements.txt" ]; then
    sed -i '/tensorflow/d' MuseTalk/requirements.txt
    sed -i '/tensorboard/d' MuseTalk/requirements.txt
    sed -i '/numpy/d' MuseTalk/requirements.txt
    sed -i '/transformers/d' MuseTalk/requirements.txt
    sed -i '/accelerate/d' MuseTalk/requirements.txt
    pip install --no-cache-dir -r MuseTalk/requirements.txt -q
fi

# 5. Create directory structure
mkdir -p checkpoints output temp

echo "✅ Environment Setup Complete! No legacy Wav2Lip dependencies remaining."
