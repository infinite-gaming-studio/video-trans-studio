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

def cleanup_vram():
    """Forcefully clear VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

from core.utils import ProgressTracker

async def run_pipeline(video_path, target_lang="zh-cn"):
    Config.print_info()
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    # 初始化进度追踪器
    tracker = ProgressTracker()
    tracker.start_reporting()

    try:
        # 获取视频文件名（不含扩展名）并创建输出子目录
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        project_output_dir = Config.OUTPUT_DIR / video_name
        project_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Extract Audio
        tracker.set_step(0, "Extracting Wav")
        audio_path = AudioProcessor.extract_audio(video_path)
        
        # 2. ASR (Whisper)
        tracker.set_step(1, "Whisper Large-v3")
        asr = ASRProcessor()
        segments = asr.transcribe(audio_path)
        asr.unload() 
        cleanup_vram()
        
        # 3. Translate
        tracker.set_step(2, f"NLLB to {target_lang}")
        translator = Translator(target_lang=target_lang)
        translated_segments = translator.translate_segments(segments)
        
        # 4. TTS (Edge-TTS)
        tracker.set_step(3, "Edge-TTS Generating")
        tts = TTSProcessor()
        dubbed_audio_path = str(project_output_dir / "dubbed_audio.wav")
        await tts.generate_full_audio(translated_segments, dubbed_audio_path)
        
        # 5. LipSync (MuseTalk)
        await update_progress(80, "Lip-Syncing (MuseTalk Syncing)")
        lipsync = LipSyncProcessor()
        await lipsync.sync(video_path, dubbed_audio_path, final_video_path)
        
        tracker.set_step(5, "Complete")
        print(f"\n\n🎉 Pipeline Finished Successfully!")
        print(f"📦 Final Result: {final_video_path}")
        print(f"📄 Also check: {dubbed_audio_path}")
        
    finally:
        tracker.stop()

if __name__ == "__main__":
    # Example usage: python main.py input_video.mp4 zh-cn
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path> [target_lang]")
        print("Example: python main.py test.mp4 zh-cn")
        sys.exit(1)
        
    video_input = sys.argv[1]
    lang_input = sys.argv[2] if len(sys.argv) > 2 else "zh-cn"
    
    import asyncio
    asyncio.run(run_pipeline(video_input, lang_input))
