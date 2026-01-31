import moviepy.editor as mp
import os
from config import Config

class AudioProcessor:
    @staticmethod
    def extract_audio(video_path, output_audio_path=None):
        """Extracts audio track from video."""
        if output_audio_path is None:
            output_audio_path = str(Config.TEMP_DIR / "original_audio.wav")
            
        print(f"🎬 Extracting audio from {video_path}...")
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path, codec='pcm_s16le', verbose=False, logger=None)
        video.close()
        print(f"✅ Audio extracted to: {output_audio_path}")
        return output_audio_path

    @staticmethod
    def combine_video_audio(video_path, audio_path, output_path):
        """Combines video and audio, replacing the original audio."""
        print(f"🎥 Merging audio and video...")
        video = mp.VideoFileClip(video_path)
        audio = mp.AudioFileClip(audio_path)
        
        final_video = video.set_audio(audio)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        
        video.close()
        audio.close()
        print(f"✅ Final video saved at: {output_path}")
        return output_path
