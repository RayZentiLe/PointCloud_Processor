from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QProgressBar
from PySide6.QtCore import Qt, Signal


class LoadingDialog(QDialog):
    """Modal loading dialog with progress bar and cancel button."""
    
    cancel_clicked = Signal()
    
    def __init__(self, title="Processing...", message="Please wait...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setMinimumHeight(150)
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        
        layout = QVBoxLayout(self)
        
        # Message label
        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel_clicked)
        layout.addWidget(self.cancel_button)
        
        self.setLayout(layout)
    
    def on_cancel_clicked(self):
        """Handle cancel button click."""
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling...")
        self.cancel_clicked.emit()
    
    def set_progress(self, value):
        """Update progress bar (0-100)."""
        self.progress_bar.setValue(int(value))
    
    def set_message(self, text):
        """Update the message label."""
        self.message_label.setText(text)
