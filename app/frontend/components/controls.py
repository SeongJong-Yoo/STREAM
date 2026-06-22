"""Playback controls: play/pause, frame slider, time display."""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt5.QtCore import Qt, pyqtSignal


class ControlsWidget(QWidget):
    play_pause_clicked = pyqtSignal(bool)  # True = playing
    slider_changed = pyqtSignal(int)       # frame index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_playing = False
        self.total_frames = 0
        self.fps = 30

        layout = QHBoxLayout(self)

        # Play / Pause
        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedWidth(180)
        self.play_btn.setFixedHeight(60)
        self.play_btn.setStyleSheet("font-size: 26px;")
        self.play_btn.clicked.connect(self._toggle)
        layout.addWidget(self.play_btn)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self._on_move)
        self.slider.sliderReleased.connect(self._on_release)
        layout.addWidget(self.slider, stretch=1)

        # Time label
        self.time_label = QLabel("0:00.0 / 0:00.0")
        self.time_label.setFixedWidth(400)
        self.time_label.setStyleSheet("font-size: 26px;")
        layout.addWidget(self.time_label)

    # ----------------------------------------------------------- public
    def set_total_frames(self, total, fps=30):
        self.total_frames = total
        self.fps = fps
        self.slider.setRange(0, max(total - 1, 0))
        self._update_label(0)

    def set_current_frame(self, frame):
        self.slider.blockSignals(True)
        self.slider.setValue(frame)
        self.slider.blockSignals(False)
        self._update_label(frame)

    # ----------------------------------------------------------- private
    def _toggle(self):
        self.is_playing = not self.is_playing
        self.play_btn.setText("Pause" if self.is_playing else "Play")
        self.play_pause_clicked.emit(self.is_playing)

    def _on_move(self, val):
        self._update_label(val)
        self.slider_changed.emit(val)

    def _on_release(self):
        self.slider_changed.emit(self.slider.value())

    def _update_label(self, frame):
        def _fmt(f):
            s = f / self.fps
            m = int(s) // 60
            s = s - m * 60
            return f"{m}:{s:04.1f}"

        self.time_label.setText(f"{_fmt(frame)} / {_fmt(self.total_frames)}")

    def stop(self):
        """Reset to stopped state."""
        self.is_playing = False
        self.play_btn.setText("Play")
