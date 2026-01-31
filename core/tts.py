import asyncio
import edge_tts
from config import Config
from pydub import AudioSegment
import os

class TTSProcessor:
    def __init__(self, voice="en-US-ChristopherNeural"):
        self.voice = voice

    async def _generate_audio(self, text, output_file):
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_file)

    def generate_full_audio(self, segments, output_path):
        """
        Generates a single audio file from translated segments.
        Ideally, this should respect the timing of original segments.
        """
        print(f"🗣️ Generating TTS audio...")
        
        # Create an empty audio track (silent) to start, or just concatenate
        # For MVP: We will just concatenate with small pauses, 
        # ignoring strict original timing for now (LipSync will handle the mouth).
        # A Better approach: Place audio at specific timestamps.
        
        combined_audio = AudioSegment.empty()
        
        # If segments have huge gaps, we might want to fill silence.
        # But for 'Dubbing', we often want continuous speech or matched speech.
        # Let's try to match original timing roughly.
        
        current_time_ms = 0
        
        temp_segment_file = Config.TEMP_DIR / "temp_seg.mp3"
        
        for i, seg in enumerate(segments):
            start_ms = int(seg['start'] * 1000)
            
            # Add silence if there is a gap between current time and next segment start
            silence_duration = start_ms - current_time_ms
            if silence_duration > 0:
                combined_audio += AudioSegment.silent(duration=silence_duration)
                current_time_ms += silence_duration
            
            # Generate speech for this segment
            # Run async function in sync context
            asyncio.run(self._generate_audio(seg['text'], str(temp_segment_file)))
            
            seg_audio = AudioSegment.from_mp3(temp_segment_file)
            
            # Speed control (Optional): 
            # If generated audio is much longer than original duration (seg['end'] - seg['start']),
            # we might want to speed it up. For now, we skip this complexity.
            
            combined_audio += seg_audio
            current_time_ms += len(seg_audio)
            
        # Export final audio
        combined_audio.export(output_path, format="wav")
        
        if os.path.exists(temp_segment_file):
            os.remove(temp_segment_file)
            
        print(f"✅ TTS Audio generated at: {output_path}")
        return output_path
