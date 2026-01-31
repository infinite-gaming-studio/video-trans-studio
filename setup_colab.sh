#!/bin/bash
echo "🚀 Starting Video Trans Studio - MuseTalk & Index-TTS2 Edition Setup..."

# 1. Update and install system dependencies
echo "📦 Installing system dependencies (ffmpeg)..."
apt-get update -qq && apt-get install -y ffmpeg -qq

# 1. Install High-Performance Python Infrastructure
echo "🚀 Installing 'uv' package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 2. Deep Clean Python Environment
echo "🧹 Cleaning up existing packages and legacy repos..."
# Remove legacy Index-TTS2 folder if it exists
rm -rf index-tts
uv pip uninstall transformers tokenizers protobuf librosa numpy jax -y -q

# 3. Install Core AI Stack (Golden Versions)
echo "🐍 Installing core AI libraries via uv..."
uv pip install --no-cache torch torchaudio torchvision -q
uv pip install --no-cache -r requirements.txt -q

# 4. Clone & Calibrate Sub-Repositories
cd /content/video-trans-studio

# LivePortrait
if [ ! -d "LivePortrait" ]; then
    echo "📥 Cloning LivePortrait..."
    git clone https://github.com/KwaiVGI/LivePortrait.git
fi

# 🚨 CRITICAL: Prevent LivePortrait from downgrading our core AI stack
if [ -f "LivePortrait/requirements.txt" ]; then
    echo "🧹 Stripping version constraints from LivePortrait/requirements.txt..."
    sed -i '/transformers/d' LivePortrait/requirements.txt
    sed -i '/numpy/d' LivePortrait/requirements.txt
    sed -i '/accelerate/d' LivePortrait/requirements.txt
    uv pip install --no-cache -r LivePortrait/requirements.txt -q
fi

# F5-TTS (Stable Voice Cloning)
echo "🎙️  Installing F5-TTS for stable voice cloning..."
uv pip install --no-cache f5-tts -q

echo "✨ Environment calibration complete. Ready for high-fidelity dubbing!"
mkdir -p checkpoints output temp

echo "✅ Environment Setup Complete! No legacy Wav2Lip dependencies remaining."
