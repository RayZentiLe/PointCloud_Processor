from datetime import datetime
from PySide6.QtWidgets import QTextEdit


class LogPanel(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumHeight(180)
        self.log("Application started.")

    def log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.append(f"[{ts}] {message}")
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())