import time
import threading
import sys

class ProgressTracker:
    def __init__(self):
        self.current_step = "Initializing"
        self.sub_status = "Waiting"
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.is_running = False
        self.steps = [
            "Audio Extraction",
            "ASR Transcription",
            "Translation",
            "TTS Generation",
            "Lip-Syncing"
        ]
        self.step_index = 0

    def set_step(self, index, status="Processing"):
        self.step_index = index
        self.current_step = self.steps[index] if index < len(self.steps) else "Finishing"
        self.sub_status = status
        self.step_start_time = time.time()

    def stop(self):
        self.is_running = False

    def _report_loop(self):
        while self.is_running:
            elapsed_total = time.time() - self.start_time
            elapsed_step = time.time() - self.step_start_time
            
            # 构造进度条样式
            progress = (self.step_index / len(self.steps)) * 100
            bar_len = 20
            filled_len = int(bar_len * self.step_index // len(self.steps))
            bar = '█' * filled_len + '-' * (bar_len - filled_len)
            
            # 实时播报日志 (使用 \r 实现原地更新)
            sys.stdout.write(
                f"\r⏳ [PROGRESS] |{bar}| {progress:.0f}% | "
                f"STEP: {self.current_step} ({self.sub_status}) | "
                f"Step Time: {elapsed_step:.1f}s | Total: {elapsed_total:.1f}s"
            )
            sys.stdout.flush()
            time.sleep(1)
        print("\n")

    def start_reporting(self):
        self.is_running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._report_loop, daemon=True)
        self.thread.start()

class SubtitleGenerator:
    @staticmethod
    def format_time(seconds):
        """Converts seconds to SRT time format: HH:MM:SS,mmm"""
        milliseconds = int((seconds - int(seconds)) * 1000)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    @staticmethod
    def wrap_text(text, max_width=40):
        """Wraps text to a maximum width without breaking words."""
        import textwrap
        return "\n".join(textwrap.wrap(text, width=max_width))

    @staticmethod
    def save_srt(segments, output_path, max_width=50):
        """
        Saves segments to an SRT file with professional formatting.
        - Handles long line wrapping
        - Cleans up whitespace
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                start = SubtitleGenerator.format_time(seg['start'])
                end = SubtitleGenerator.format_time(seg['end'])
                # 过滤空内容
                text = seg.get('text', '').strip()
                if not text:
                    continue
                
                # 自动换行处理
                wrapped_text = SubtitleGenerator.wrap_text(text, max_width=max_width)
                
                f.write(f"{i}\n{start} --> {end}\n{wrapped_text}\n\n")
        print(f"📄 Subtitles saved to: {output_path}")
