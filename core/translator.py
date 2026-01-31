import os
# from openai import OpenAI # Uncomment when using OpenAI
from deep_translator import GoogleTranslator

class Translator:
    def __init__(self, target_lang="es"):
        self.target_lang = target_lang
        # self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def translate_text(self, text):
        """
        Translates a single string.
        """
        try:
            # Simple free translation for prototype
            translated = GoogleTranslator(source='auto', target=self.target_lang).translate(text)
            return translated
        except Exception as e:
            print(f"Translation Error: {e}")
            return text

    def translate_segments(self, segments):
        """
        Translates a list of segments (from ASR).
        """
        print(f"🌍 Translating {len(segments)} segments to {self.target_lang}...")
        translated_segments = []
        
        for seg in segments:
            # In a real scenario, you might batch these for LLM processing
            new_text = self.translate_text(seg['text'])
            translated_segments.append({
                "start": seg['start'],
                "end": seg['end'],
                "original_text": seg['text'],
                "text": new_text
            })
            
        print("✅ Translation complete.")
        return translated_segments
