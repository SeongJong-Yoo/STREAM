#!/usr/bin/env python3
"""Dance Design Studio – entry point."""

import sys
import os
import inspect
import warnings

import numpy as np

# ---------- Compatibility monkeypatches ----------
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    for attr, fallback in [
        ("bool", bool), ("int", int), ("float", float),
        ("complex", complex), ("object", object), ("str", str), ("unicode", str),
    ]:
        if not hasattr(np, attr):
            setattr(np, attr, fallback)
# --------------------------------------------------

# Add project root to sys.path so `src.*` imports work
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from PyQt5.QtWidgets import QApplication
from app.frontend.main_window import MainWindow


def main():
    # Force unbuffered output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    app = QApplication(sys.argv)
    app.setApplicationName("Dance Design Studio")

    # Dark palette
    from PyQt5.QtGui import QPalette, QColor
    from PyQt5.QtCore import Qt

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(55, 55, 55))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.BrightText, QColor(255, 50, 50))
    palette.setColor(QPalette.Highlight, QColor(80, 120, 200))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    # Global stylesheet tweaks
    app.setStyleSheet("""
        QGroupBox {
            border: 1px solid #555;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 14px;
            font-weight: bold;
            color: #bbb;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }
        QPushButton {
            border: 1px solid #666;
            border-radius: 3px;
            padding: 4px 14px;
            background: #444;
            color: #ddd;
        }
        QPushButton:hover { background: #555; }
        QPushButton:pressed { background: #333; }
        QPushButton:disabled { color: #666; background: #3a3a3a; }
        QComboBox {
            border: 1px solid #555;
            border-radius: 3px;
            padding: 3px 8px;
            background: #3a3a3a;
            color: #ddd;
        }
        QTextEdit {
            border: 1px solid #555;
            border-radius: 3px;
            background: #2a2a2a;
            color: #ddd;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 3px;
            text-align: center;
            color: #ddd;
            background: #333;
        }
        QProgressBar::chunk {
            background: #4a7abf;
            border-radius: 2px;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #444;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #888;
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover { background: #aaa; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
