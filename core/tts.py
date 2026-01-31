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
        """Lazy load F5-TTS model."""
        if self.model is not None:
            return

        print("⏳ Loading F5-TTS into VRAM...")
        try:
            from f5_tts.api import F5TTS
            self.model = F5TTS(device=self.device)
            print("✅ F5-TTS Model Loaded.")
        except Exception as e:
            print(f"❌ Failed to load F5-TTS: {e}")
            raise

    async def generate_full_audio(self, segments, original_audio_path, output_path, emo_alpha=None):
        """
        Generates full dubbed audio with F5-TTS zero-shot voice cloning.
        - segments: List of translated segments (with start, end, text)
        - original_audio_path: Path to the original full audio wav
        """
        self.load_model()
        print(f"🗣️ Cloning voices and rendering {len(segments)} segments via F5-TTS...")

        # Create temp dir for segments
        temp_dir = Config.TEMP_DIR / "f5tts_segments"
        temp_dir.mkdir(exist_ok=True)
        
        # Load full original audio for cropping reference samples
        orig_audio = AudioSegment.from_wav(original_audio_path)
        
        combined_audio = AudioSegment.silent(duration=0, frame_rate=44100)
        
        for i, seg in enumerate(segments):
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)
            text = seg['text']
            
            # Extract original segment as voice prompt for cloning
            ref_path = temp_dir / f"ref_{i:04d}.wav"
            ref_seg = orig_audio[start_ms:end_ms]
            
            # F5-TTS works best with 5-10s reference. Let's pad if short.
            if len(ref_seg) < 5000:
                pad_start = max(0, start_ms - 2000)
                pad_end = min(len(orig_audio), end_ms + 2000)
                ref_seg = orig_audio[pad_start:pad_end]
            
            ref_seg.export(str(ref_path), format="wav")

            # Output path for synthesized segment
            seg_out_path = temp_dir / f"syn_{i:04d}.wav"
            
            print(f"🎙️ Rendering Segment {i} (F5-TTS Cloning)...")
            
            # F5-TTS Inference
            self.model.infer(
                ref_file=str(ref_path),
                ref_text="", # F5-TTS uses ASR on reference if text is empty, more robust
                gen_text=text,
                output_file=str(seg_out_path)
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