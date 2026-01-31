#!/bin/bash
echo "🚀 Starting Video Trans Studio Colab Setup..."

# 1. Update and install system dependencies
echo "📦 Installing system dependencies (ffmpeg)..."
apt-get update -qq && apt-get install -y ffmpeg -qq

# 2. Deep Clean Python Environment
echo "🧹 Cleaning up existing packages to prevent conflicts..."
pip uninstall -y transformers tokenizers protobuf librosa -q

# 3. Install Python requirements in specific order
echo "🐍 Installing Python libraries (this may take a few minutes)..."
# A. Base AI Infrastructure
pip install --no-cache-dir torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118 -q
pip install --no-cache-dir "transformers>=4.40.0" "tokenizers>=0.14" "protobuf==3.20.3" -q
pip install --no-cache-dir accelerate sentencepiece -q

# B. Project Requirements
pip install --no-cache-dir -r requirements.txt -q

# C. Specific version fixes
pip install --no-cache-dir librosa==0.9.1 -q

# 4. Setup Wav2Lip and Index-TTS2
if [ ! -d "Wav2Lip" ]; then
    echo "📥 Cloning Wav2Lip repository..."
    git clone https://github.com/Rudrabha/Wav2Lip.git -q
fi

if [ ! -d "index-tts" ]; then
    echo "📥 Cloning Index-TTS2 repository (skipping large files)..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/index-tts/index-tts.git -q
fi

# Fixes for Wav2Lip (Python 3.10+ compatibility)
touch Wav2Lip/__init__.py
find Wav2Lip -name "*.py" -exec sed -i 's/from collections import Iterable/from collections.abc import Iterable/g' {} +

# Fix for librosa.output.write_wav deprecation (use soundfile instead)
# This is critical because librosa >= 0.8.0 removed output module
find Wav2Lip -name "audio.py" -exec sed -i 's/import librosa/import librosa\nimport soundfile as sf/g' {} +
find Wav2Lip -name "audio.py" -exec sed -i 's/librosa.output.write_wav/sf.write/g' {} +


# 5. Create directory structure
mkdir -p checkpoints output temp

# 6. Pre-download NLLB model to verify installation
echo "📥 Verifying Transformers installation..."
python3 -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; print('✅ Transformers is working correctly!')"

echo "✅ Setup Complete!"