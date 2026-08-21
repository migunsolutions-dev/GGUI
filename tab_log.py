import os
import time
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout, QLabel
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

LOG_FLUSH_INTERVAL_S = 0.15
LOG_FLUSH_MAX_BYTES = 8192


def should_flush_log_buffer(buffer_text: str, *, idle: bool, elapsed_s: float) -> bool:
    """Coalesce log lines so the GUI is not painted once per solver line."""
    if not buffer_text:
        return False
    if idle:
        return True
    if elapsed_s >= LOG_FLUSH_INTERVAL_S:
        return True
    if len(buffer_text) >= LOG_FLUSH_MAX_BYTES:
        return True
    return False


class LogWorker(QThread):
    """
    Worker thread that reads the log file in the background.
    This prevents the GUI from freezing during file I/O operations (especially on WSL).
    """
    new_text_signal = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self._is_running = True

    def run(self):
        # Wait for file to be created
        while self._is_running and not os.path.exists(self.file_path):
            self.msleep(500)

        if not self._is_running:
            return

        # Open file and monitor
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
                # Start from the beginning (or seek(0, 2) to start from end)
                # For a full log view, we start from 0.
                f.seek(0)
                buf = []
                last_emit = time.monotonic()

                while self._is_running:
                    line = f.readline()
                    if line:
                        buf.append(line)
                    idle = not line
                    chunk = "".join(buf)
                    if should_flush_log_buffer(
                        chunk,
                        idle=idle and bool(buf),
                        elapsed_s=time.monotonic() - last_emit,
                    ):
                        self.new_text_signal.emit(chunk)
                        buf.clear()
                        last_emit = time.monotonic()
                    if idle:
                        self.msleep(100)
                if buf:
                    self.new_text_signal.emit("".join(buf))
        except Exception as e:
            self.new_text_signal.emit(f"\n[Error reading log: {e}]\n")

    def stop(self):
        self._is_running = False
        self.wait()

class LogTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # Layout
        self.layout = QVBoxLayout(self)
        
        # Header / Controls
        ctrl_layout = QHBoxLayout()
        self.lbl_path = QLabel("No log file loaded.")
        self.lbl_path.setStyleSheet("color: gray; font-style: italic;")
        ctrl_layout.addWidget(self.lbl_path)
        
        self.btn_clear = QPushButton("Clear Log")
        self.btn_clear.clicked.connect(self.clear_log)
        self.btn_clear.setFixedWidth(100)
        ctrl_layout.addWidget(self.btn_clear)
        
        self.layout.addLayout(ctrl_layout)

        # Log Viewer (Text Edit)
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        # Terminal Style
        self.text_view.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e; 
                color: #00ff00;
                border: 1px solid #333;
            }
        """)
        self.text_view.setFont(QFont("Consolas", 10)) # Monospace font
        self.text_view.setLineWrapMode(QTextEdit.NoWrap) # Create horizontal scrollbar
        self.layout.addWidget(self.text_view)

        # Worker Thread Reference
        self.worker = None
        self.text_view.setUpdatesEnabled(False)

    def start_monitoring(self, file_path):
        """Start the background worker thread."""
        self.stop_monitoring() # Clean up old thread if exists
        
        self.clear_log()
        self.lbl_path.setText(f"Monitoring: {os.path.basename(file_path)}")
        self.append_text(f"--- Listening to: {file_path} ---\n")

        # Create and start the thread
        self.worker = LogWorker(file_path)
        self.worker.new_text_signal.connect(self.append_text, Qt.QueuedConnection)
        if not self.isVisible():
            self.text_view.setUpdatesEnabled(False)
        self.worker.start()

    def stop_monitoring(self):
        """Stop the background worker."""
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
            self.append_text("\n--- Monitoring Stopped ---")

    def showEvent(self, event):
        super().showEvent(event)
        self.text_view.setUpdatesEnabled(True)

    def hideEvent(self, event):
        self.text_view.setUpdatesEnabled(False)
        super().hideEvent(event)

    def append_text(self, text):
        """Slot to receive text from thread and update GUI."""
        view = self.text_view
        live = self.isVisible()
        if not live and view.updatesEnabled():
            view.setUpdatesEnabled(False)
        # Check if scrollbar is at the bottom before inserting
        sb = view.verticalScrollBar()
        at_bottom = (sb.value() == sb.maximum())

        view.insertPlainText(text)

        # Auto-scroll only if we were already at the bottom
        if at_bottom:
            sb.setValue(sb.maximum())
        if live and not view.updatesEnabled():
            view.setUpdatesEnabled(True)

    def clear_log(self):
        self.text_view.clear()