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
if [ ! -d "LivePortrait" ]; then
    echo "📥 Cloning LivePortrait repository..."
    git clone https://github.com/KwaiVGI/LivePortrait.git -q
    touch LivePortrait/__init__.py
fi

if [ ! -d "index-tts" ]; then
    echo "📥 Cloning Index-TTS2 repository..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/index-tts/index-tts.git -q
fi

# 4.5 Install LivePortrait & Index-TTS2 Specific Dependencies
echo "📦 Installing LivePortrait and Index-TTS2 dependencies..."
pip install --no-cache-dir onnxruntime-gpu -q

# 🚨 CRITICAL: Prevent LivePortrait from downgrading our core AI stack
if [ -f "LivePortrait/requirements.txt" ]; then
    echo "🧹 Stripping version constraints from LivePortrait/requirements.txt..."
    sed -i '/transformers/d' LivePortrait/requirements.txt
    sed -i '/numpy/d' LivePortrait/requirements.txt
    sed -i '/accelerate/d' LivePortrait/requirements.txt
    sed -i '/opencv-python/d' LivePortrait/requirements.txt
    pip install --no-cache-dir -r LivePortrait/requirements.txt -q
fi

# 🚨 FINAL CALIBRATION: Ensure core versions are NOT downgraded by dependencies
echo "🛠️ Finalizing environment calibration..."
pip install --no-cache-dir "transformers>=4.46.0" "numpy>=2.0.0,<2.1.0" "accelerate>=0.33.0" -q

# 5. Create directory structure
mkdir -p checkpoints output temp

echo "✅ Environment Setup Complete! No legacy Wav2Lip dependencies remaining."
