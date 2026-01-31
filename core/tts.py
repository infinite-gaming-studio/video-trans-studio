import asyncio
import os
import torch
import gc
import sys
import subprocess
import requests
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
from config import Config

# Monkey patch for NumPy 2.0+ compatibility
if not hasattr(np, "complex"): np.complex = complex
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"): np.int = int
if not hasattr(np, "bool"): np.bool = bool
if not hasattr(np, "object"): np.object = object

# Monkey patch for transformers compatibility (Index-TTS2 sensitivity)
try:
    import transformers.cache_utils
    if not hasattr(transformers.cache_utils, "EncoderDecoderCache"):
        transformers.cache_utils.EncoderDecoderCache = object
    if not hasattr(transformers.cache_utils, "OffloadedCache"):
        transformers.cache_utils.OffloadedCache = object
except ImportError:
    pass

class TTSProcessor:
    """
    Industrial-grade TTS Processor using Index-TTS2 for Zero-shot Voice Cloning.
    Replaces legacy Edge-TTS for high-fidelity, synchronized output.
    """
    def __init__(self, device="cuda"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.repo_path = Config.BASE_DIR / "index-tts"
        self.model_dir = Config.INDEXTTS_MODEL_DIR
        self.model = None

    def setup(self):
        """Ensures Index-TTS2 repo and models are ready."""
        if not self.repo_path.exists():
            print("📥 Cloning Index-TTS2 repository...")
            subprocess.run(["git", "clone", Config.INDEXTTS_REPO_URL], check=True)
            # Add to path for imports
            if str(self.repo_path) not in sys.path:
                sys.path.append(str(self.repo_path))

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._download_models()

    def _download_models(self):
        """Downloads Index-TTS2 weights if missing."""
        for name, url in Config.INDEXTTS_MODELS.items():
            dest = self.model_dir / name
            if not dest.exists():
                print(f"📥 Downloading Index-TTS2 weight: {name}")
                response = requests.get(url, stream=True)
                total = int(response.headers.get('content-length', 0))
                with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

    def load_model(self):
        """Loads Index-TTS2 into VRAM."""
        if self.model is None:
            self.setup()
            print("⏳ Loading Index-TTS2 into VRAM (FP16 mode)...")
            try:
                # Add repo to sys.path to allow internal imports
                sys.path.append(str(self.repo_path))
                from indextts.infer_v2 import IndexTTS2
                
                self.model = IndexTTS2(
                    cfg_path=str(Config.INDEXTTS_CONFIG_PATH),
                    model_dir=str(self.model_dir),
                    use_fp16=True if self.device == "cuda" else False
                )
                print("✅ Index-TTS2 Model Loaded.")
            except Exception as e:
                print(f"❌ Failed to load Index-TTS2: {e}")
                raise

    async def generate_full_audio(self, segments, original_audio_path, output_path):
        """
        Generates full dubbed audio with voice cloning for each segment.
        - segments: List of translated segments (with start, end, text)
        - original_audio_path: Path to the original full audio wav
        """
        self.load_model()
        print(f"🗣️ Cloning voices and rendering {len(segments)} segments...")

        # Create temp dir for segments
        temp_dir = Config.TEMP_DIR / "indextts_segments"
        temp_dir.mkdir(exist_ok=True)
        
        # Load full original audio for cropping reference samples
        orig_audio = AudioSegment.from_wav(original_audio_path)
        
        combined_audio = AudioSegment.silent(duration=0, frame_rate=44100)
        
        for i, seg in enumerate(segments):
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)
            text = seg['text']
            
            # Extract original segment as voice prompt for cloning
            # We take the original segment audio as the reference speaker prompt
            ref_path = temp_dir / f"ref_{i:04d}.wav"
            ref_seg = orig_audio[start_ms:end_ms]
            # If segment is too short, extend it for better cloning (Index-TTS needs ~3-5s for best results)
            if len(ref_seg) < 3000:
                # Try to take a bit more around it
                pad_start = max(0, start_ms - 1000)
                pad_end = min(len(orig_audio), end_ms + 1000)
                ref_seg = orig_audio[pad_start:pad_end]
            ref_seg.export(str(ref_path), format="wav")

            # Output path for synthesized segment
            seg_out_path = temp_dir / f"syn_{i:04d}.wav"
            
            print(f"🎙️ Rendering Segment {i} (Cloning Original Voice)...")
            # Index-TTS2 inference is synchronous, so we run in executor if needed
            # but usually okay to run sequential for high quality
            self.model.infer(
                spk_audio_prompt=str(ref_path),
                text=text,
                output_path=str(seg_out_path)
            )
            
            # Load and Merge with Sync Protection
            if seg_out_path.exists():
                syn_audio = AudioSegment.from_wav(str(seg_out_path))
                
                # Dynamic Sync (Rate check) - Index-TTS is natural but text might be long
                # If much longer than original, we might need a slight stretch
                target_dur = end_ms - start_ms
                if len(syn_audio) > target_dur * 1.2 and target_dur > 0:
                    speed = min(len(syn_audio) / target_dur, 1.25)
                    syn_audio = syn_audio.speedup(playback_speed=speed, chunk_size=150, crossfade=25)

                # Ensure base is long enough
                if len(combined_audio) < start_ms:
                    combined_audio += AudioSegment.silent(duration=start_ms - len(combined_audio), frame_rate=44100)
                
                # Overlay
                combined_audio = combined_audio.overlay(syn_audio, position=start_ms)
                
                # Cleanup temp segment files
                os.remove(ref_path)
                os.remove(seg_out_path)

        # Export final merged audio
        combined_audio.export(output_path, format="wav", parameters=["-ar", "44100", "-ac", "2"])
        print(f"✅ Voice Cloned Dubbing Complete: {output_path}")
        return output_path

    def unload(self):
        """Releases VRAM."""
        if self.model:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🗑️ Index-TTS2 Unloaded from VRAM.")