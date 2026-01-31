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

    def translate_segments(self, segments, batch_size=16):
        print(f"🌍 Translating {len(segments)} segments (Dubbing Strategy: Conciseness)...")
        translated_segments = []
        
        # 预处理：如果是翻译成英文，且中文原句很短，我们需要提示或采用精简策略
        # 对于 NLLB 这种模型，我们通过控制 max_length 和生成参数来控制长度
        
        texts = [seg['text'] for seg in segments]
        translated_texts = []

        if self.use_local:
            lang_map = {"zh": "zho_Hans", "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn"}
            target_code = lang_map.get(self.target_lang, "zho_Hans")
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
                
                # 工业级技巧：通过 penalty 鼓励模型生成更精炼的句子，避免啰嗦
                translated_tokens = self.model.generate(
                    **inputs, 
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[target_code], 
                    max_length=100,      # 限制最大长度
                    length_penalty=1.0,   # 长度惩罚因子
                    num_beams=4
                )
                batch_results = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                translated_texts.extend(batch_results)
        else:
            # 在线 API 模式：如果是 GPT 可以加提示词，Google 则通过后续逻辑修剪
            for text in texts:
                translated_texts.append(self.translate_text(text))

        # 组装结果
        for i, seg in enumerate(segments):
            translated_segments.append({
                "start": seg['start'],
                "end": seg['end'],
                "original_text": seg['text'],
                "text": translated_texts[i]
            })
            
        return translated_segments
