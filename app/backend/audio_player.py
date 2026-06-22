"""Reliable audio player using ffplay subprocess.

QMediaPlayer + GStreamer on Linux is unreliable. This wraps ffplay
(from ffmpeg, already installed) for guaranteed playback with seek support.
"""

import os
import signal
import subprocess
import time


class AudioPlayer:
    """Audio player backed by an ffplay subprocess."""

    def __init__(self):
        self._process = None
        self._audio_path = None
        self._position = 0.0        # current position in seconds
        self._playing = False
        self._play_start_wall = 0.0  # wall-clock when play() was called
        self._play_start_pos = 0.0   # audio position when play() was called
        self._volume = 100

    # -------------------------------------------------------- media
    def set_file(self, path):
        """Load an audio file (absolute path)."""
        self.stop()
        self._audio_path = os.path.abspath(path)
        self._position = 0.0

    # -------------------------------------------------------- transport
    def play(self):
        if not self._audio_path or not os.path.isfile(self._audio_path):
            print(f"[AudioPlayer] No audio file: {self._audio_path}")
            return
        self._kill()
        vol = max(0, min(self._volume, 100))
        self._process = subprocess.Popen(
            [
                "ffplay", "-nodisp", "-autoexit",
                "-loglevel", "error",
                "-ss", f"{self._position:.3f}",
                "-volume", str(vol),
                self._audio_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._playing = True
        self._play_start_wall = time.time()
        self._play_start_pos = self._position

    def pause(self):
        if self._playing and self._process:
            elapsed = time.time() - self._play_start_wall
            self._position = self._play_start_pos + elapsed
            self._kill()
            self._playing = False

    def stop(self):
        self._kill()
        self._playing = False
        self._position = 0.0

    def set_position(self, ms):
        """Seek to *ms* milliseconds."""
        self._position = ms / 1000.0
        if self._playing:
            # Restart playback from new position
            self.play()

    def set_volume(self, vol):
        """Set volume 0-100 (applied on next play)."""
        self._volume = max(0, min(vol, 100))

    # -------------------------------------------------------- queries
    @property
    def is_playing(self):
        if self._playing and self._process:
            # Check if ffplay is still alive
            if self._process.poll() is not None:
                self._playing = False
        return self._playing

    @property
    def position_ms(self):
        """Approximate current position in milliseconds."""
        if self._playing:
            elapsed = time.time() - self._play_start_wall
            return int((self._play_start_pos + elapsed) * 1000)
        return int(self._position * 1000)

    # -------------------------------------------------------- internal
    def _kill(self):
        if self._process is not None:
            try:
                self._process.kill()
                self._process.wait(timeout=2)
            except Exception:
                pass
            self._process = None

    def __del__(self):
        self._kill()
