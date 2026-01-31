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

async def run_pipeline(video_path, target_lang="zh-cn"):
    Config.print_info()
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    # 获取视频文件名（不含扩展名）并创建输出子目录
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    project_output_dir = Config.OUTPUT_DIR / video_name
    project_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*50)
    print(f"📂 Project Output Directory: {project_output_dir}")
    print("="*50)

    print("\n" + "="*50)
    print("🚀 STEP 1: Extracting Audio from Video...")
    audio_path = AudioProcessor.extract_audio(video_path)
    print(f"✅ Audio extracted: {audio_path}")
    
    print("\n" + "="*50)
    print("🚀 STEP 2: Automatic Speech Recognition (ASR)...")
    asr = ASRProcessor()
    segments = asr.transcribe(audio_path)
    asr.unload() 
    cleanup_vram()
    print(f"✅ Transcription complete. {len(segments)} segments detected.")
    
    print("\n" + "="*50)
    print(f"🚀 STEP 3: Translating segments to {target_lang}...")
    translator = Translator(target_lang=target_lang)
    translated_segments = translator.translate_segments(segments)
    print(f"✅ Translation complete.")
    
    print("\n" + "="*50)
    print("🚀 STEP 4: Text-to-Speech (TTS) Generation...")
    tts = TTSProcessor()
    # 将配音音频也保存在项目文件夹中
    dubbed_audio_path = str(project_output_dir / "dubbed_audio.wav")
    await tts.generate_full_audio(translated_segments, dubbed_audio_path)
    print(f"✅ Dubbed audio generated: {dubbed_audio_path}")
    
    print("\n" + "="*50)
    print("🚀 STEP 5: Lip-Syncing (Wav2Lip)...")
    lipsync = LipSyncProcessor()
    final_video_path = str(project_output_dir / f"final_{video_name}_{target_lang}.mp4")
    
    lipsync.sync(video_path, dubbed_audio_path, final_video_path)
    
    print("\n" + "="*50)
    print(f"🎉 Pipeline Finished Successfully!")
    print(f"📦 Final Result: {final_video_path}")
    print(f"📄 Also check: {dubbed_audio_path}")
    print("="*50)

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
