"""Main window for the Dance Design Studio."""

import os
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QFileDialog, QMessageBox, QProgressBar, QLabel,
    QStatusBar, QInputDialog, QListWidget, QListWidgetItem,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from app.frontend.components.smpl_viewer import SMPLViewer
from app.frontend.components.timeline import TimelineWidget
from app.frontend.components.editing_panel import EditingPanel
from app.frontend.components.controls import ControlsWidget

from app.backend.project_manager import ProjectManager
from app.backend.model_runner import ModelRunner
from app.backend.recorder import render_frames, encode_video
from app.backend.audio_player import AudioPlayer


# ======================================================= Worker threads

class PrepWorker(QThread):
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(str)   # error or ""

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        from app.backend.prep.data_prep import prepare_project
        err = prepare_project(self.folder, progress_callback=self._cb)
        self.finished.emit(err or "")

    def _cb(self, msg, frac):
        self.progress.emit(msg, frac)


class GenerateWorker(QThread):
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(object)  # (smpl, joints, verts, faces) or str error

    def __init__(self, runner, folder, segments):
        super().__init__()
        self.runner = runner
        self.folder = folder
        self.segments = segments

    def run(self):
        try:
            result = self.runner.generate(
                self.folder, self.segments, progress_callback=self._cb
            )
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(str(e))

    def _cb(self, msg, frac):
        self.progress.emit(msg, frac)


class EncodeWorker(QThread):
    """Runs ffmpeg encoding in a background thread (no OpenGL needed)."""
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(str)

    def __init__(self, frame_pattern, audio_path, output_path, fps, tmpdir):
        super().__init__()
        self.frame_pattern = frame_pattern
        self.audio_path = audio_path
        self.output_path = output_path
        self.fps = fps
        self._tmpdir = tmpdir          # prevent cleanup until done

    def run(self):
        try:
            encode_video(
                self.frame_pattern, self.audio_path, self.output_path,
                fps=self.fps, progress_callback=self._cb,
            )
            self.finished.emit("")
        except Exception as e:
            self.finished.emit(str(e))

    def _cb(self, msg, frac):
        self.progress.emit(msg, frac)


class PartialRegenWorker(QThread):
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(object)  # (smpl, joints, verts, faces) or str error

    def __init__(self, runner, folder, segments, chunk_indices):
        super().__init__()
        self.runner = runner
        self.folder = folder
        self.segments = segments
        self.chunk_indices = chunk_indices

    def run(self):
        try:
            result = self.runner.regenerate_partial(
                self.folder, self.segments, self.chunk_indices, progress_callback=self._cb
            )
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(str(e))

    def _cb(self, msg, frac):
        self.progress.emit(msg, frac)


# ============================================================ Main Window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dance Design Studio")
        self.resize(1400, 850)

        self.project = ProjectManager()
        self.model_runner = ModelRunner()
        self._worker = None

        self._build_ui()
        self._connect_signals()

        # Audio player (subprocess-based, bypasses unreliable Qt multimedia)
        self.audio_player = AudioPlayer()
        self.audio_player.set_volume(100)

        # Playback timer
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._tick)
        self._playback_start_time = None
        self._playback_start_frame = 0

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- Toolbar ----
        toolbar = QHBoxLayout()
        self.btn_load = QPushButton("Load Data")
        self.btn_load_model = QPushButton("Load Model")
        self.btn_generate = QPushButton("Generate")
        self.btn_record = QPushButton("Record")
        self.btn_save = QPushButton("Save")
        self.btn_api_key = QPushButton("Gemini API Key")
        self.btn_load_model.setEnabled(False)
        self.btn_generate.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.btn_save.setEnabled(False)
        for b in (self.btn_load, self.btn_load_model, self.btn_generate, self.btn_record, self.btn_save, self.btn_api_key):
            b.setFixedHeight(64)
            b.setStyleSheet("font-size: 26px;")
            toolbar.addWidget(b)
        toolbar.addStretch()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(600)
        self.progress_bar.setStyleSheet("font-size: 26px;")
        self.progress_bar.setFixedHeight(40)
        self.progress_bar.hide()
        toolbar.addWidget(self.progress_bar)
        root.addLayout(toolbar)

        # ---- Main content: [Folder list | SMPL viewer | Editing panel] ----
        self.splitter = QSplitter(Qt.Horizontal)

        # Folder list (left)
        self.folder_list = QListWidget()
        self.folder_list.setStyleSheet(
            "font-size: 40px; background: #2a2a2a; color: #ddd;"
        )
        self.folder_list.setMinimumWidth(150)

        self.smpl_viewer = SMPLViewer()
        self.editing_panel = EditingPanel(
            label_desc_path=str(
                Path(__file__).resolve().parents[1] / "data" / "label_descriptions.json"
            )
        )

        self.splitter.addWidget(self.folder_list)
        self.splitter.addWidget(self.smpl_viewer)
        self.splitter.addWidget(self.editing_panel)
        self.splitter.setSizes([200, 800, 350])
        root.addWidget(self.splitter, stretch=1)

        # ---- Timeline ----
        self.timeline = TimelineWidget()
        root.addWidget(self.timeline)

        # ---- Controls ----
        self.controls = ControlsWidget()
        root.addWidget(self.controls)

        # ---- Status bar ----
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready. Load a project folder to begin.")

    # --------------------------------------------------------- signals
    def _connect_signals(self):
        self.btn_load.clicked.connect(self._on_load)
        self.btn_load_model.clicked.connect(self._on_load_model)
        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_record.clicked.connect(self._on_record)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_api_key.clicked.connect(self._on_api_key)

        self.controls.slider_changed.connect(self._set_frame)
        self.controls.play_pause_clicked.connect(self._toggle_playback)
        self.timeline.selection_changed.connect(self._on_selection)
        self.timeline.playhead_clicked.connect(self._set_frame)
        self.editing_panel.applied.connect(self._on_apply_edit)
        self.folder_list.itemClicked.connect(self._on_folder_clicked)

    # =========================================================== LOAD DATA
    def _on_load(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if not folder:
            return
        self._load_folder(folder)

    def _load_folder(self, folder):
        """Load a data folder and update UI. Populates sibling folder list."""
        err = self.project.load_data(folder)
        if err:
            QMessageBox.warning(self, "Load Error", err)
            return

        self.status.showMessage(f"Data loaded: {folder}")
        self.timeline.set_segments(self.project.segments, self.project.fps)
        self.controls.set_total_frames(self.project.total_frames, self.project.fps)
        self.smpl_viewer.clear()
        self.btn_load_model.setEnabled(True)
        self.btn_generate.setEnabled(self.model_runner.model is not None)
        self.btn_record.setEnabled(False)
        self.btn_save.setEnabled(False)

        # Set audio
        self.audio_player.set_file(self.project.audio_path)

        # Populate sibling folder list
        self._populate_folder_list(Path(folder))

        # Check if audio features need preparation
        if self.project.needs_audio_prep:
            reply = QMessageBox.question(
                self, "Audio Features",
                "Audio features not found. Prepare now?\n(This may take a while.)",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._run_prep()

    # ======================================================= FOLDER LIST
    def _populate_folder_list(self, loaded_folder):
        """List sibling folders (same parent) that contain audio.wav + label/."""
        parent = loaded_folder.parent
        self.folder_list.clear()
        self._folder_list_parent = parent

        for child in sorted(parent.iterdir()):
            if not child.is_dir():
                continue
            # Only show folders that look like valid data folders
            if (child / "audio.wav").exists() and (child / "label").is_dir():
                item = QListWidgetItem(child.name)
                item.setData(Qt.UserRole, str(child))
                self.folder_list.addItem(item)
                if child == loaded_folder:
                    item.setSelected(True)
                    self.folder_list.setCurrentItem(item)

    def _on_folder_clicked(self, item):
        folder = item.data(Qt.UserRole)
        if folder and folder != str(self.project.folder):
            self._load_folder(folder)

    # ========================================================= LOAD MODEL
    def _on_load_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if not folder:
            return

        folder = Path(folder)
        ckpt = folder / "checkpoints" / "last.ckpt"
        cfg = folder / "config.yaml"

        if not ckpt.exists():
            QMessageBox.warning(self, "Load Error", "Missing checkpoints/last.ckpt")
            return
        if not cfg.exists():
            QMessageBox.warning(self, "Load Error", "Missing config.yaml")
            return

        self.status.showMessage("Loading model...")
        self.progress_bar.show()
        self.progress_bar.setFormat("Loading model...")
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

        err = self.model_runner.load_model(
            str(cfg), str(ckpt), progress_callback=self._on_progress
        )
        self.progress_bar.hide()

        if err:
            QMessageBox.critical(self, "Model Error", err)
            return

        self.project.model_path = str(ckpt)
        self.btn_generate.setEnabled(self.project.folder is not None)
        self.status.showMessage(f"Model loaded: {folder.name}")

    def _run_prep(self):
        self._set_busy(True)
        self._worker = PrepWorker(str(self.project.folder))
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_prep_done)
        self._worker.start()

    def _on_prep_done(self, err):
        self._set_busy(False)
        if err:
            QMessageBox.critical(self, "Prep Error", err)
        else:
            self.status.showMessage("Audio features ready.")

    # ======================================================== GENERATE
    def _on_generate(self):
        if self.project.folder is None:
            return

        if self.model_runner.model is None:
            QMessageBox.warning(
                self, "No Model",
                "Please load a model first (Load Model).",
            )
            return

        if self.project.needs_audio_prep:
            QMessageBox.warning(
                self, "Not Ready",
                "Audio features are not prepared yet. Please prepare first.",
            )
            return

        self._set_busy(True)
        self._worker = GenerateWorker(
            self.model_runner, str(self.project.folder), self.project.segments
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_generate_done)
        self._worker.start()

    def _on_generate_done(self, result):
        self._set_busy(False)
        if isinstance(result, str):
            QMessageBox.critical(self, "Generation Error", result)
            return

        smpl_params, joints, verts, faces = result
        self.project.motion_data = smpl_params
        self.project.joint_data = joints
        self.project.vert_data = verts
        self.project.faces = faces

        self.smpl_viewer.set_mesh_data(verts, faces)
        total = verts.shape[0]
        self.controls.set_total_frames(total, self.project.fps)
        self.timeline.total_frames = total
        self.timeline.update()

        self.btn_record.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.status.showMessage(f"Generated {total} frames ({total/self.project.fps:.1f}s)")

    # ========================================================== RECORD
    def _on_record(self):
        if self.project.vert_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Video", str(self.project.folder / "output.mp4"), "MP4 (*.mp4)"
        )
        if not path:
            return

        self._set_busy(True)
        self._record_output_path = path

        # Phase 1: render frames on the main thread (OpenGL requires it)
        import tempfile
        self._record_tmpdir = tempfile.mkdtemp()
        try:
            pattern = render_frames(
                self.project.vert_data, self.project.faces,
                self._record_tmpdir, fps=self.project.fps,
                progress_callback=self._on_progress,
                viewer=self.smpl_viewer,
            )
        except Exception as e:
            self._set_busy(False)
            self._cleanup_record_tmp()
            QMessageBox.critical(self, "Record Error", f"Rendering failed:\n{e}")
            return

        # Phase 2: encode with ffmpeg in a background thread
        self._worker = EncodeWorker(
            pattern, self.project.audio_path, path, self.project.fps,
            self._record_tmpdir,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_record_done)
        self._worker.start()

    def _on_record_done(self, err):
        self._set_busy(False)
        self._cleanup_record_tmp()
        if err:
            QMessageBox.critical(self, "Record Error", err)
        else:
            self.status.showMessage(f"Video saved: {self._record_output_path}")

    def _cleanup_record_tmp(self):
        import shutil
        d = getattr(self, "_record_tmpdir", None)
        if d and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        self._record_tmpdir = None

    # ============================================================ SAVE
    def _on_save(self):
        if self.project.motion_data is None:
            return
        err = self.project.save_results()
        if err:
            QMessageBox.warning(self, "Save Error", err)
        else:
            self.status.showMessage("Results saved to result.npy")

    # ======================================================= API KEY
    def _on_api_key(self):
        key, ok = QInputDialog.getText(
            self, "Gemini API Key",
            "Enter your Google Gemini API key:",
        )
        if not ok or not key.strip():
            return
        try:
            from app.backend.suggestion_generator import SuggestionGenerator
            desc_path = str(
                Path(__file__).resolve().parents[1] / "data" / "label_descriptions.json"
            )
            self.editing_panel._suggestion_generator = SuggestionGenerator(
                api_key=key.strip(), label_desc_path=desc_path,
            )
            self.btn_api_key.setStyleSheet("font-size: 26px; border: 2px solid #5a5;")
            self.status.showMessage("Gemini API key set successfully.")
        except Exception as e:
            QMessageBox.critical(self, "API Key Error", str(e))

    # ======================================================= PLAYBACK
    def _set_frame(self, frame):
        self.smpl_viewer.set_frame(frame)
        self.timeline.set_playhead(frame)
        self.controls.set_current_frame(frame)
        self._update_current_label_overlay(frame)
        # Sync audio position
        ms = int(frame / self.project.fps * 1000)
        self.audio_player.set_position(ms)

    def _toggle_playback(self, playing):
        if playing:
            if self.smpl_viewer.total_frames == 0 and self.project.total_frames == 0:
                self.controls.stop()
                return
            # Sync audio to current slider position before starting
            start_frame = self.controls.slider.value()
            ms = int(start_frame / self.project.fps * 1000)
            self.audio_player.set_position(ms)

            self._playback_start_time = time.time()
            self._playback_start_frame = start_frame
            interval = int(1000 / self.project.fps)
            self.play_timer.start(interval)
            self.audio_player.play()
        else:
            self.play_timer.stop()
            self.audio_player.pause()
            self._playback_start_time = None

    def _tick(self):
        total = self.smpl_viewer.total_frames or self.project.total_frames
        if total == 0 or self._playback_start_time is None:
            return

        # Use wall-clock time for reliable frame pacing
        elapsed = time.time() - self._playback_start_time
        frame = self._playback_start_frame + int(elapsed * self.project.fps)
        frame = max(0, min(frame, total - 1))

        self.smpl_viewer.set_frame(frame)
        self.timeline.set_playhead(frame)
        self.controls.set_current_frame(frame)
        self._update_current_label_overlay(frame)

        if frame >= total - 1:
            self._toggle_playback(False)
            self.controls.stop()

    # ================================================ TIMELINE EDITING
    def _on_selection(self, indices):
        self.editing_panel.show_segment(self.project.segments, indices)

    def _on_apply_edit(self):
        indices = self.editing_panel._selected_indices
        if not indices:
            return
        genre = self.editing_panel.current_genre
        label = self.editing_panel.current_label
        desc = self.editing_panel.current_description
        edit_start = self.editing_panel.edit_start_frame
        edit_end = self.editing_panel.edit_end_frame

        # Use range-aware split/merge edit
        affected = self.project.apply_edit_with_range(
            edit_start, edit_end, genre, label, desc
        )

        # Refresh timeline with new segment list
        self.timeline.set_segments(self.project.segments, self.project.fps)
        self.controls.set_total_frames(self.project.total_frames, self.project.fps)
        self.status.showMessage(
            f"Edited range {edit_start}-{edit_end} "
            f"({len(self.project.segments)} segments total)"
        )

        # Trigger partial regeneration if we have previous generation results
        if affected and self.project.motion_data is not None and self.model_runner.model is not None:
            chunk_indices = self.model_runner.segments_to_chunks(
                affected, self.project.segments, self.project.fps
            )
            if chunk_indices:
                self._set_busy(True)
                self.status.showMessage(
                    f"Regenerating {len(chunk_indices)} chunk(s)..."
                )
                self._worker = PartialRegenWorker(
                    self.model_runner, str(self.project.folder),
                    self.project.segments, chunk_indices,
                )
                self._worker.progress.connect(self._on_progress)
                self._worker.finished.connect(self._on_generate_done)
                self._worker.start()

    # ================================================ LABEL OVERLAY
    def _update_current_label_overlay(self, frame):
        """Show the genre and label of the segment at the current frame."""
        for seg in self.project.segments:
            if seg.get("start", 0) <= frame < seg.get("end", 0):
                genre = seg.get("genre", "")
                label = seg.get("label", "")
                self.smpl_viewer.set_label_text(f"{genre}  |  {label}")
                return
        self.smpl_viewer.set_label_text("")

    # ====================================================== PROGRESS
    def _on_progress(self, msg, frac):
        self.progress_bar.setValue(int(frac * 100))
        self.progress_bar.setFormat(msg)

    def _set_busy(self, busy):
        self.progress_bar.setVisible(busy)
        self.progress_bar.setValue(0)
        self.btn_load.setEnabled(not busy)
        self.btn_load_model.setEnabled(not busy and self.project.folder is not None)
        self.btn_generate.setEnabled(not busy and self.model_runner.model is not None)
        self.btn_record.setEnabled(not busy and self.project.vert_data is not None)
        self.btn_save.setEnabled(not busy and self.project.motion_data is not None)

    # ====================================================== CLEANUP
    def closeEvent(self, event):
        self.audio_player.stop()
        super().closeEvent(event)
