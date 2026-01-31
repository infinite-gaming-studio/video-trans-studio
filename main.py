import sys
import os
import torch
import gc
from config import Config
from core.audio import AudioProcessor
from core.asr import ASRProcessor
from core.translator import Translator
from core.tts import TTSProcessor
from core.lipsync import LipSyncProcessor
from core.utils import ProgressTracker, SubtitleGenerator

def cleanup_vram():
    """Forcefully clear VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

async def run_pipeline(video_path, target_lang="en"):
    """
    Orchestrates the full video translation pipeline.
    video_path: Path to source video
    target_lang: Language code for translation (default: en)
    """
    Config.print_info()
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    # Initialize progress tracker
    tracker = ProgressTracker()
    tracker.start_reporting()

    try:
        # Prepare output directory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        project_output_dir = Config.OUTPUT_DIR / video_name
        project_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output file paths
        final_video_path = str(project_output_dir / f"final_{video_name}_{target_lang}.mp4")
        original_srt_path = str(project_output_dir / f"{video_name}_original.srt")
        translated_srt_path = str(project_output_dir / f"{video_name}_{target_lang}.srt")
        
        # 1. Extract Audio
        tracker.set_step(0, "Audio Extraction (Extracting Wav)")
        audio_path = AudioProcessor.extract_audio(video_path)
        
        # 2. ASR (Whisper)
        tracker.set_step(1, "ASR Transcription (Whisper Large-v3)")
        asr = ASRProcessor()
        segments = asr.transcribe(audio_path)
        SubtitleGenerator.save_srt(segments, original_srt_path)
        asr.unload() 
        cleanup_vram()
        
        # 3. Translate
        tracker.set_step(2, f"Translation (NLLB to {target_lang})")
        translator = Translator(target_lang=target_lang)
        translated_segments = translator.translate_segments(segments)
        SubtitleGenerator.save_srt(translated_segments, translated_srt_path)
        
        # 4. TTS (Index-TTS2 Voice Cloning)
        tracker.set_step(3, "TTS Generation (Index-TTS2 Cloning)")
        tts = TTSProcessor()
        dubbed_audio_path = str(project_output_dir / "dubbed_audio.wav")
        # Pass the original audio path for speaker cloning
        await tts.generate_full_audio(translated_segments, audio_path, dubbed_audio_path)
        tts.unload()
        cleanup_vram()
        
        # 5. LipSync (MuseTalk)
        tracker.set_step(4, "Lip-Syncing (MuseTalk Syncing)")
        lipsync = LipSyncProcessor()
        # MuseTalk process
        await lipsync.sync(video_path, dubbed_audio_path, final_video_path)
        
        tracker.set_step(5, "Pipeline Complete")
        print(f"\n\n🎉 Pipeline Finished Successfully!")
        print(f"📦 Final Result: {final_video_path}")
        print(f"📄 Also check: {dubbed_audio_path}")
        
        return final_video_path
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        tracker.stop()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path> [target_lang]")
        sys.exit(1)
        
    video_input = sys.argv[1]
    lang_input = sys.argv[2] if len(sys.argv) > 2 else "en"
    
    import asyncio
    asyncio.run(run_pipeline(video_input, lang_input))