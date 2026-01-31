import os
from deep_translator import GoogleTranslator
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class Translator:
    def __init__(self, target_lang="zh", use_local=False):
        self.target_lang = target_lang
        self.use_local = use_local
        self.model = None
        self.tokenizer = None
        
        if use_local:
            print("⏳ Loading local NLLB-200 translation model (600M)...")
            model_name = "facebook/nllb-200-distilled-600M"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
            print("✅ Local Translation Model Loaded.")

    def translate_text(self, text):
        if not self.use_local:
            try:
                return GoogleTranslator(source='auto', target=self.target_lang).translate(text)
            except:
                return text
        else:
            # Map common lang codes to NLLB lang codes
            # Simple mapping for demo: zh -> zho_Hans, en -> eng_Latn
            lang_map = {"zh": "zho_Hans", "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn"}
            target_code = lang_map.get(self.target_lang, "zho_Hans")
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            translated_tokens = self.model.generate(
                **inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[target_code], max_length=128
            )
            return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    def translate_segments(self, segments):
        print(f"🌍 Translating {len(segments)} segments...")
        translated_segments = []
        for seg in segments:
            new_text = self.translate_text(seg['text'])
            translated_segments.append({
                "start": seg['start'],
                "end": seg['end'],
                "original_text": seg['text'],
                "text": new_text
            })
        return translated_segments
