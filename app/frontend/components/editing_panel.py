"""Editing panel: genre / label dropdowns + description text editor."""

import json
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QTextEdit, QPushButton, QGroupBox,
    QDoubleSpinBox,
)
from PyQt5.QtCore import pyqtSignal, QThread


class _SuggestWorker(QThread):
    """Run Gemini API call in background thread."""
    finished = pyqtSignal(str)  # suggestion text or error prefixed with "ERROR:"

    def __init__(self, generator, genre, label):
        super().__init__()
        self.generator = generator
        self.genre = genre
        self.label = label

    def run(self):
        try:
            text = self.generator.suggest(self.genre, self.label)
            self.finished.emit(text)
        except Exception as e:
            self.finished.emit(f"ERROR: {e}")


class EditingPanel(QWidget):
    """Right-side panel for editing the selected timeline segment(s)."""

    # Emits when the user clicks Apply
    applied = pyqtSignal()

    def __init__(self, label_desc_path=None, parent=None):
        super().__init__(parent)
        self.label_descriptions = {}   # {genre: {label: [desc, ...]}}
        self._selected_indices = []    # indices into project segments
        self._segments_ref = []        # reference to project manager segments list
        self._suggestion_generator = None
        self._suggest_worker = None

        if label_desc_path:
            self._load_label_descriptions(label_desc_path)

        self._build_ui()

    # ----------------------------------------------------------- UI
    def _build_ui(self):
        root = QVBoxLayout(self)

        # Title
        self.title_label = QLabel("Select a segment on the timeline")
        self.title_label.setStyleSheet("font-weight:bold; font-size:39px; color:#ccc;")
        root.addWidget(self.title_label)

        # Time range
        time_grp = QGroupBox("Time Range")
        time_grp.setStyleSheet("QGroupBox { font-size: 39px; } QLabel { font-size: 39px; }")
        time_lay = QHBoxLayout(time_grp)
        time_lay.addWidget(QLabel("Start (s):"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setStyleSheet("font-size: 39px;")
        self.start_spin.setDecimals(1)
        self.start_spin.setSingleStep(0.1)
        self.start_spin.setMinimum(0.0)
        self.start_spin.setMaximum(9999.0)
        time_lay.addWidget(self.start_spin)
        time_lay.addWidget(QLabel("End (s):"))
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setStyleSheet("font-size: 39px;")
        self.end_spin.setDecimals(1)
        self.end_spin.setSingleStep(0.1)
        self.end_spin.setMinimum(0.0)
        self.end_spin.setMaximum(9999.0)
        time_lay.addWidget(self.end_spin)
        root.addWidget(time_grp)

        # Genre
        grp = QGroupBox("Genre")
        grp.setStyleSheet("QGroupBox { font-size: 39px; }")
        grp_lay = QVBoxLayout(grp)
        self.genre_combo = QComboBox()
        self.genre_combo.setStyleSheet("font-size: 39px; padding: 6px;")
        self.genre_combo.addItems(sorted(self.label_descriptions.keys()))
        self.genre_combo.currentTextChanged.connect(self._on_genre_changed)
        grp_lay.addWidget(self.genre_combo)
        root.addWidget(grp)

        # Label
        grp2 = QGroupBox("Label")
        grp2.setStyleSheet("QGroupBox { font-size: 39px; }")
        grp2_lay = QVBoxLayout(grp2)
        self.label_combo = QComboBox()
        self.label_combo.setStyleSheet("font-size: 39px; padding: 6px;")
        self.label_combo.currentTextChanged.connect(self._on_label_changed)
        grp2_lay.addWidget(self.label_combo)
        root.addWidget(grp2)

        # Description
        grp3 = QGroupBox("Description")
        grp3.setStyleSheet("QGroupBox { font-size: 39px; }")
        grp3_lay = QVBoxLayout(grp3)
        self.desc_edit = QTextEdit()
        self.desc_edit.setPlaceholderText("Enter description...")
        self.desc_edit.setMaximumHeight(200)
        self.desc_edit.setStyleSheet("font-size: 39px;")
        grp3_lay.addWidget(self.desc_edit)
        root.addWidget(grp3)

        # Examples — shown when genre/label are selected
        grp4 = QGroupBox("Examples")
        grp4.setStyleSheet("QGroupBox { font-size: 39px; }")
        grp4_lay = QVBoxLayout(grp4)
        self.example_label = QLabel("")
        self.example_label.setWordWrap(True)
        self.example_label.setStyleSheet("color:#bbb; font-size:33px; line-height:1.4;")
        grp4_lay.addWidget(self.example_label)
        self.example_group = grp4
        self.example_group.hide()
        root.addWidget(grp4)

        # AI-Suggestion panel
        grp5 = QGroupBox("AI-Suggestion")
        grp5.setStyleSheet("QGroupBox { font-size: 39px; }")
        grp5_lay = QVBoxLayout(grp5)
        self.suggestion_edit = QTextEdit()
        self.suggestion_edit.setPlaceholderText("AI-generated description will appear here...")
        self.suggestion_edit.setMaximumHeight(160)
        self.suggestion_edit.setStyleSheet("font-size: 33px;")
        grp5_lay.addWidget(self.suggestion_edit)
        ai_btn_row = QHBoxLayout()
        self.suggest_btn = QPushButton("Suggest")
        self.suggest_btn.setFixedHeight(68)
        self.suggest_btn.setStyleSheet("font-size: 33px;")
        self.suggest_btn.clicked.connect(self._on_suggest)
        self.ai_apply_btn = QPushButton("Apply")
        self.ai_apply_btn.setFixedHeight(68)
        self.ai_apply_btn.setStyleSheet("font-size: 33px;")
        self.ai_apply_btn.clicked.connect(self._on_ai_apply)
        ai_btn_row.addWidget(self.suggest_btn)
        ai_btn_row.addWidget(self.ai_apply_btn)
        grp5_lay.addLayout(ai_btn_row)
        root.addWidget(grp5)

        # Apply button
        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setFixedHeight(80)
        self.apply_btn.setStyleSheet("font-size: 39px;")
        self.apply_btn.clicked.connect(self._on_apply)
        self.apply_btn.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(self.apply_btn)
        root.addLayout(btn_row)

        root.addStretch()
        self.setMinimumWidth(280)

    # ------------------------------------------------------ public API
    def set_label_descriptions(self, path):
        self._load_label_descriptions(path)
        self.genre_combo.clear()
        self.genre_combo.addItems(sorted(self.label_descriptions.keys()))

    def show_segment(self, segments_ref, selected_indices):
        """Display the given segment(s) for editing.

        Parameters
        ----------
        segments_ref : list  – the full segments list (from ProjectManager)
        selected_indices : list[int]
        """
        self._segments_ref = segments_ref
        self._selected_indices = selected_indices

        if not selected_indices:
            self.title_label.setText("Select a segment on the timeline")
            self.apply_btn.setEnabled(False)
            return

        self.apply_btn.setEnabled(True)
        fps = self._fps()

        if len(selected_indices) == 1:
            seg = segments_ref[selected_indices[0]]
            self.title_label.setText(
                f"Segment {selected_indices[0]}  "
                f"({seg.get('start',0)/fps:.1f}s – {seg.get('end',0)/fps:.1f}s)"
            )
            self._set_combo_silent(self.genre_combo, seg.get("genre", ""))
            self._on_genre_changed(seg.get("genre", ""))
            self._set_combo_silent(self.label_combo, seg.get("label", ""))
            self.desc_edit.setPlainText(seg.get("description", ""))
            # Populate time range from this segment
            self.start_spin.setValue(seg.get("start", 0) / fps)
            self.end_spin.setValue(seg.get("end", 0) / fps)
        else:
            genres = {segments_ref[i].get("genre", "") for i in selected_indices}
            labels = {segments_ref[i].get("label", "") for i in selected_indices}
            self.title_label.setText(f"{len(selected_indices)} segments selected")
            if len(genres) == 1:
                self._set_combo_silent(self.genre_combo, genres.pop())
            if len(labels) == 1:
                self._set_combo_silent(self.label_combo, labels.pop())
            self.desc_edit.clear()
            # Populate time range: min start to max end of all selected
            min_start = min(segments_ref[i].get("start", 0) for i in selected_indices)
            max_end = max(segments_ref[i].get("end", 0) for i in selected_indices)
            self.start_spin.setValue(min_start / fps)
            self.end_spin.setValue(max_end / fps)

        self._update_suggestion()

    # ------------------------------------------------------ getters
    @property
    def current_genre(self):
        return self.genre_combo.currentText()

    @property
    def current_label(self):
        return self.label_combo.currentText()

    @property
    def current_description(self):
        return self.desc_edit.toPlainText()

    @property
    def edit_start_frame(self):
        """Start of the edit range, in frames."""
        return int(round(self.start_spin.value() * self._fps()))

    @property
    def edit_end_frame(self):
        """End of the edit range, in frames."""
        return int(round(self.end_spin.value() * self._fps()))

    # ------------------------------------------------------ slots
    def _on_genre_changed(self, genre):
        self.label_combo.blockSignals(True)
        self.label_combo.clear()
        labels = self.label_descriptions.get(genre, {})
        self.label_combo.addItems(sorted(labels.keys()))
        self.label_combo.blockSignals(False)
        self._update_suggestion()

    def _on_label_changed(self, label):
        self._update_suggestion()

    def _on_suggest(self):
        genre = self.genre_combo.currentText()
        label = self.label_combo.currentText()
        if not genre or not label:
            return

        if self._suggestion_generator is None:
            self.suggestion_edit.setPlainText(
                "Please set your Gemini API key first (toolbar button)."
            )
            return

        self.suggest_btn.setEnabled(False)
        self.suggest_btn.setText("Generating...")
        self._suggest_worker = _SuggestWorker(self._suggestion_generator, genre, label)
        self._suggest_worker.finished.connect(self._on_suggest_done)
        self._suggest_worker.start()

    def _on_suggest_done(self, text):
        self.suggest_btn.setEnabled(True)
        self.suggest_btn.setText("Suggest")
        self.suggestion_edit.setPlainText(text)

    def _on_ai_apply(self):
        text = self.suggestion_edit.toPlainText().strip()
        if not text or text.startswith("ERROR:"):
            return
        self.desc_edit.setPlainText(text)
        self._on_apply()

    def _on_apply(self):
        self.applied.emit()

    # ------------------------------------------------------ helpers
    def _fps(self):
        if self._segments_ref:
            return self._segments_ref[0].get("fps", 30)
        return 30

    def _load_label_descriptions(self, path):
        path = Path(path)
        if path.exists():
            with open(path, "r") as f:
                self.label_descriptions = json.load(f)

    def _update_suggestion(self):
        genre = self.genre_combo.currentText()
        label = self.label_combo.currentText()
        descs = self.label_descriptions.get(genre, {}).get(label, [])
        if descs:
            examples = descs[:3]
            numbered = [f"{i+1}. {ex}" for i, ex in enumerate(examples)]
            self.example_label.setText("\n\n".join(numbered))
            self.example_group.show()
        else:
            self.example_label.setText("")
            self.example_group.hide()

    @staticmethod
    def _set_combo_silent(combo, text):
        combo.blockSignals(True)
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)
