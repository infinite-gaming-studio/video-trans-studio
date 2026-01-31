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

def run_pipeline(video_path, target_lang="zh-cn"):
    Config.print_info()
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    # 1. Extract Audio
    audio_path = AudioProcessor.extract_audio(video_path)
    
    # 2. ASR (Whisper)
    asr = ASRProcessor()
    segments = asr.transcribe(audio_path)
    asr.unload() 
    cleanup_vram()
    
    # 3. Translate
    translator = Translator(target_lang=target_lang)
    translated_segments = translator.translate_segments(segments)
    
    # 4. TTS (Edge-TTS)
    # Note: Edge-TTS uses asyncio, which we handle inside TTSProcessor
    tts = TTSProcessor()
    dubbed_audio_path = str(Config.TEMP_DIR / "dubbed_audio.wav")
    tts.generate_full_audio(translated_segments, dubbed_audio_path)
    
    # 5. LipSync (Wav2Lip)
    # This is where we need the most VRAM
    lipsync = LipSyncProcessor()
    final_output_name = f"final_{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
    final_video_path = str(Config.OUTPUT_DIR / final_output_name)
    
    lipsync.sync(video_path, dubbed_audio_path, final_video_path)
    
    print(f"\n🎉 Process Finished!")
    print(f"📦 Final Output: {final_video_path}")

if __name__ == "__main__":
    # Example usage: python main.py input_video.mp4 zh-cn
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path> [target_lang]")
        print("Example: python main.py test.mp4 zh-cn")
        sys.exit(1)
        
    video_input = sys.argv[1]
    lang_input = sys.argv[2] if len(sys.argv) > 2 else "zh-cn"
    
    run_pipeline(video_input, lang_input)
