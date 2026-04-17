import sys
import traceback
from PySide6.QtCore import QThread, Signal


class TaskRunner(QThread):
    """Run a heavy function in a background thread."""

    progress = Signal(int)          # 0-100
    finished_result = Signal(object)
    error = Signal(str)

    def __init__(self, func, **kwargs):
        super().__init__()
        self._func = func
        self._kwargs = kwargs

    def run(self):
        try:
            print(f"[Worker] Starting: {self._func.__name__}", file=sys.stderr)
            self._kwargs["progress_cb"] = self._emit_progress
            result = self._func(**self._kwargs)
            print(f"[Worker] Finished: {self._func.__name__}", file=sys.stderr)
            self.finished_result.emit(result)
        except Exception as e:
            msg = f"{e}\n{traceback.format_exc()}"
            print(f"[Worker] ERROR in {self._func.__name__}: {msg}",
                  file=sys.stderr)
            self.error.emit(msg)

    def _emit_progress(self, value):
        self.progress.emit(int(value))