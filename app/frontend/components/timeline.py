"""Timeline widget: displays genre+label blocks as coloured strips.

Supports click-to-select and drag-to-multi-select.
"""

from PyQt5.QtWidgets import QWidget, QToolTip
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QRectF
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QBrush, QPen, QFontMetrics,
    QLinearGradient,
)

# Vivid colours per genre
GENRE_COLORS = {
    "charleston": (QColor(0x3A, 0x86, 0xFF), QColor(0x1B, 0x5E, 0xD4)),  # bright blue
    "house":      (QColor(0xFF, 0x9F, 0x1C), QColor(0xE0, 0x7A, 0x00)),  # amber
    "hip_hop":    (QColor(0xFF, 0x47, 0x5A), QColor(0xD6, 0x1F, 0x33)),  # coral red
    "krump":      (QColor(0x2E, 0xCC, 0xB1), QColor(0x17, 0xA4, 0x8B)),  # teal
    "jazz":       (QColor(0x4E, 0xC9, 0x4E), QColor(0x2D, 0x9E, 0x2D)),  # green
    "tap":        (QColor(0xFF, 0xD6, 0x3A), QColor(0xE0, 0xB0, 0x00)),  # gold
    "popping":    (QColor(0xC7, 0x6E, 0xFF), QColor(0x9B, 0x4D, 0xD2)),  # purple
    "locking":    (QColor(0xFF, 0x6B, 0x9D), QColor(0xD9, 0x45, 0x78)),  # pink
    "random":     (QColor(0xA8, 0x8B, 0x6E), QColor(0x87, 0x6B, 0x50)),  # warm brown
}
DEFAULT_COLOR = (QColor(0x88, 0x88, 0x88), QColor(0x66, 0x66, 0x66))

GENRE_ROW_HEIGHT = 32
LABEL_ROW_HEIGHT = 76
HEADER_HEIGHT = 36
BLOCK_HEIGHT = GENRE_ROW_HEIGHT + LABEL_ROW_HEIGHT


class TimelineWidget(QWidget):
    """Draws a horizontal bar of coloured blocks for each label segment."""

    # Emits list of selected segment indices
    selection_changed = pyqtSignal(list)
    # Emits frame index when playhead position is clicked
    playhead_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.segments = []          # list of dicts (genre, label, start, end, fps, ...)
        self.fps = 30
        self.total_frames = 0
        self.playhead_frame = 0

        self._selected = set()      # set of segment indices
        self._drag_start = None     # (x pixel) for rubber-band
        self._drag_end = None

        self.setMinimumHeight(BLOCK_HEIGHT + HEADER_HEIGHT + 12)
        self.setMaximumHeight(BLOCK_HEIGHT + HEADER_HEIGHT + 24)
        self.setMouseTracking(True)

    # -------------------------------------------------------- public API
    def set_segments(self, segments, fps=30):
        self.segments = segments
        self.fps = fps
        if segments:
            self.total_frames = max(s.get("end", 0) for s in segments) + 1
        else:
            self.total_frames = 0
        self._selected.clear()
        self.update()

    def set_playhead(self, frame):
        self.playhead_frame = frame
        self.update()

    def selected_indices(self):
        return sorted(self._selected)

    def clear_selection(self):
        self._selected.clear()
        self.selection_changed.emit([])
        self.update()

    # -------------------------------------------------------- geometry
    def _x_for_frame(self, frame):
        if self.total_frames <= 0:
            return 0
        return int(frame / self.total_frames * self.width())

    def _frame_for_x(self, x):
        if self.width() <= 0:
            return 0
        return int(x / self.width() * self.total_frames)

    def _seg_rect(self, seg):
        x1 = self._x_for_frame(seg["start"])
        x2 = self._x_for_frame(seg["end"])
        return QRect(x1, HEADER_HEIGHT, max(x2 - x1, 2), BLOCK_HEIGHT)

    # -------------------------------------------------------- painting
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor(24, 24, 28))

        if not self.segments:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignCenter, "No labels loaded")
            p.end()
            return

        genre_font = QFont("sans-serif", 14, QFont.Bold)
        label_font = QFont("sans-serif", 18)
        genre_fm = QFontMetrics(genre_font)
        label_fm = QFontMetrics(label_font)

        # Draw time header (every 5 seconds)
        header_font = QFont("sans-serif", 14)
        p.setFont(header_font)
        p.setPen(QColor(100, 100, 110))
        step = max(int(5 * self.fps), 1)
        for frame in range(0, self.total_frames, step):
            x = self._x_for_frame(frame)
            sec = frame / self.fps
            p.drawText(x + 3, HEADER_HEIGHT - 4, f"{sec:.0f}s")
            p.setPen(QPen(QColor(55, 55, 60), 1))
            p.drawLine(x, HEADER_HEIGHT, x, h)
            p.setPen(QColor(100, 100, 110))

        # Draw segment blocks
        for i, seg in enumerate(self.segments):
            rect = self._seg_rect(seg)
            genre = seg.get("genre", "")
            color_top, color_bot = GENRE_COLORS.get(genre, DEFAULT_COLOR)

            selected = i in self._selected
            if selected:
                color_top = color_top.lighter(130)
                color_bot = color_bot.lighter(130)

            # Gradient fill
            grad = QLinearGradient(rect.x(), rect.y(), rect.x(), rect.y() + rect.height())
            grad.setColorAt(0.0, color_top)
            grad.setColorAt(1.0, color_bot)
            p.setBrush(QBrush(grad))
            p.setPen(QPen(QColor(15, 15, 18), 1))
            p.drawRoundedRect(QRectF(rect), 3, 3)

            # Selection highlight border
            if selected:
                p.setPen(QPen(QColor(255, 255, 255, 180), 2))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(QRectF(rect).adjusted(1, 1, -1, -1), 3, 3)

            # Genre tag (top row, small bold)
            genre_rect = QRect(rect.x(), rect.y(), rect.width(), GENRE_ROW_HEIGHT)
            p.setFont(genre_font)
            p.setPen(QColor(255, 255, 255, 200))
            genre_text = genre_fm.elidedText(genre.upper(), Qt.ElideRight, rect.width() - 6)
            p.drawText(genre_rect.adjusted(4, 1, -2, 0), Qt.AlignLeft | Qt.AlignVCenter, genre_text)

            # Label text (bottom row, larger)
            label_rect = QRect(rect.x(), rect.y() + GENRE_ROW_HEIGHT, rect.width(), LABEL_ROW_HEIGHT)
            p.setFont(label_font)
            p.setPen(QColor(255, 255, 255))
            label_text = seg.get("label", "")
            elided = label_fm.elidedText(label_text, Qt.ElideRight, rect.width() - 6)
            p.drawText(label_rect.adjusted(4, 0, -2, -2), Qt.AlignLeft | Qt.AlignVCenter, elided)

        # Draw rubber-band
        if self._drag_start is not None and self._drag_end is not None:
            x1 = min(self._drag_start, self._drag_end)
            x2 = max(self._drag_start, self._drag_end)
            p.setBrush(QBrush(QColor(255, 255, 255, 30)))
            p.setPen(QPen(QColor(255, 255, 255, 100), 1, Qt.DashLine))
            p.drawRect(x1, HEADER_HEIGHT, x2 - x1, BLOCK_HEIGHT)

        # Playhead
        px = self._x_for_frame(self.playhead_frame)
        p.setPen(QPen(QColor(255, 70, 70), 2))
        p.drawLine(px, 0, px, h)
        # Playhead knob
        p.setBrush(QBrush(QColor(255, 70, 70)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(px - 4, 0, 8, 8)

        p.end()

    # ------------------------------------------------------ mouse events
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.x()
            self._drag_end = event.x()

            # Check if clicking on a segment
            clicked = self._seg_at(event.x(), event.y())
            if clicked is not None:
                mods = event.modifiers()
                if mods & Qt.ControlModifier:
                    # Toggle
                    if clicked in self._selected:
                        self._selected.discard(clicked)
                    else:
                        self._selected.add(clicked)
                elif mods & Qt.ShiftModifier:
                    # Range select
                    if self._selected:
                        lo = min(self._selected)
                        hi = max(self._selected)
                        for j in range(min(lo, clicked), max(hi, clicked) + 1):
                            self._selected.add(j)
                    else:
                        self._selected = {clicked}
                else:
                    self._selected = {clicked}
                self.selection_changed.emit(self.selected_indices())
            else:
                self._selected.clear()
                self.selection_changed.emit([])

            # Also emit playhead position
            frame = self._frame_for_x(event.x())
            self.playhead_clicked.emit(frame)
            self.update()

    def mouseMoveEvent(self, event):
        if self._drag_start is not None and event.buttons() & Qt.LeftButton:
            self._drag_end = event.x()
            # Rubber-band selection
            x1 = min(self._drag_start, self._drag_end)
            x2 = max(self._drag_start, self._drag_end)
            self._selected.clear()
            for i, seg in enumerate(self.segments):
                rect = self._seg_rect(seg)
                if rect.right() >= x1 and rect.left() <= x2:
                    self._selected.add(i)
            self.selection_changed.emit(self.selected_indices())
            self.update()
        else:
            # Tooltip
            idx = self._seg_at(event.x(), event.y())
            if idx is not None:
                seg = self.segments[idx]
                tip = f"{seg.get('genre','')}: {seg.get('label','')}"
                QToolTip.showText(event.globalPos(), tip)

    def mouseReleaseEvent(self, event):
        self._drag_start = None
        self._drag_end = None
        self.update()

    # ------------------------------------------------------------- helpers
    def _seg_at(self, x, y):
        for i, seg in enumerate(self.segments):
            if self._seg_rect(seg).contains(x, y):
                return i
        return None
