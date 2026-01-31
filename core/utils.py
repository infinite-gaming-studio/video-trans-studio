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
