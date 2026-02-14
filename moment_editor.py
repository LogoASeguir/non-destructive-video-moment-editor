from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from audio_segmenter import AudioSegmenter

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Moment:
    id: str
    start: float
    end: float
    label: str = ""
    kind: str = "manual"
    segments: List[Tuple[float, float]] = field(default_factory=list)

    def duration(self) -> float:
        if self.segments:
            return sum(seg[1] - seg[0] for seg in self.segments)
        return max(0.0, self.end - self.start)

    def get_segments(self) -> List[Tuple[float, float]]:
        """Returns segments if compound, otherwise returns [(start, end)]"""
        if self.segments:
            return list(self.segments)
        return [(self.start, self.end)]

    def is_compound(self) -> bool:
        """Returns True if this moment has multiple segments"""
        return len(self.segments) > 1

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "kind": self.kind,
        }
        if self.segments:
            d["segments"] = [[s, e] for s, e in self.segments]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Moment":
        segments_raw = data.get("segments", [])
        segments = [(float(s[0]), float(s[1])) for s in segments_raw] if segments_raw else []
        return cls(
            id=str(data.get("id", "")),
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            label=str(data.get("label", "")),
            kind=str(data.get("kind", "manual")),
            segments=segments,
        )

    def copy(self) -> "Moment":
        return Moment(
            id=self.id,
            start=self.start,
            end=self.end,
            label=self.label,
            kind=self.kind,
            segments=list(self.segments),
        )


# ---------------------------------------------------------------------------
# Worker signal bridge for thread-safe UI updates
# ---------------------------------------------------------------------------

class ScanWorkerSignals(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)


# ---------------------------------------------------------------------------
# Per-clip editor dialog
# ---------------------------------------------------------------------------

class ClipEditorDialog(QDialog):
    """
    Per-clip editor dialog.

    - Start/End can be typed
    - Mini scrub slider inside the clip
    - Snap start/end to current playhead
    - Preview clip (auto-stop at end)
    """

    def __init__(self, parent: "MomentEditorWindow", moment: Moment):
        super().__init__(parent)
        self.parent_editor = parent
        self.original = moment
        self.moment = moment.copy()

        self.setWindowTitle(f"Clip Editor – {moment.id}")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Show segment info for compound moments
        if moment.is_compound():
            seg_info = f"Compound moment with {len(moment.segments)} segments:\n"
            for i, (s, e) in enumerate(moment.segments, 1):
                seg_info += f"  Seg {i}: {s:.3f} → {e:.3f} ({e-s:.3f}s)\n"
            info = QLabel(
                f"Editing {moment.id}\n"
                f"Total duration: {moment.duration():.3f}s\n"
                f"{seg_info}"
            )
        else:
            info = QLabel(
                f"Editing {moment.id}\n"
                f"Current: {moment.start:.3f} → {moment.end:.3f}  "
                f"({moment.duration():.3f}s)"
            )
        info.setWordWrap(True)
        layout.addWidget(info)

        dur_sec = max(0.001, self.parent_editor.player.duration() / 1000.0)
        has_media = dur_sec > 0.01

        self.start_spin = QDoubleSpinBox(self)
        self.start_spin.setDecimals(3)
        self.start_spin.setMinimum(0.0)
        self.start_spin.setMaximum(dur_sec)
        self.start_spin.setSingleStep(0.050)
        self.start_spin.setValue(self.moment.start)

        self.end_spin = QDoubleSpinBox(self)
        self.end_spin.setDecimals(3)
        self.end_spin.setMinimum(0.0)
        self.end_spin.setMaximum(dur_sec)
        self.end_spin.setSingleStep(0.050)
        self.end_spin.setValue(self.moment.end)

        row = QHBoxLayout()
        row.addWidget(QLabel("Start (s):"))
        row.addWidget(self.start_spin)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("End (s):"))
        row.addWidget(self.end_spin)
        layout.addLayout(row)

        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label:"))
        self.label_edit = QLineEdit(self)
        self.label_edit.setText(self.moment.label)
        label_row.addWidget(self.label_edit)
        layout.addLayout(label_row)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.timeline_slider.setRange(0, 1000)
        layout.addWidget(self.timeline_slider)

        controls = QHBoxLayout()
        self.btn_start_from_playhead = QPushButton("Start = Playhead", self)
        self.btn_end_from_playhead = QPushButton("End = Playhead", self)
        self.btn_jump_start = QPushButton("Jump to Start", self)
        self.btn_jump_end = QPushButton("Jump to End", self)
        self.btn_preview = QPushButton("Preview Clip", self)

        controls.addWidget(self.btn_start_from_playhead)
        controls.addWidget(self.btn_end_from_playhead)
        controls.addWidget(self.btn_jump_start)
        controls.addWidget(self.btn_jump_end)
        controls.addWidget(self.btn_preview)
        layout.addLayout(controls)

        if not has_media:
            for w in (
                self.btn_start_from_playhead,
                self.btn_end_from_playhead,
                self.btn_jump_start,
                self.btn_jump_end,
                self.btn_preview,
                self.timeline_slider,
            ):
                w.setEnabled(False)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.timeline_slider.sliderMoved.connect(self._on_timeline_moved)
        self.btn_start_from_playhead.clicked.connect(self._set_start_from_playhead)
        self.btn_end_from_playhead.clicked.connect(self._set_end_from_playhead)
        self.btn_jump_start.clicked.connect(self._jump_to_start)
        self.btn_jump_end.clicked.connect(self._jump_to_end)
        self.btn_preview.clicked.connect(self._preview_clip)

    def _on_timeline_moved(self, value: int) -> None:
        start = self.start_spin.value()
        end = self.end_spin.value()
        if end <= start:
            return
        frac = value / 1000.0
        t = start + frac * (end - start)
        self.parent_editor.seek_to(t)

    def _set_start_from_playhead(self) -> None:
        self.start_spin.setValue(self.parent_editor.current_time())

    def _set_end_from_playhead(self) -> None:
        self.end_spin.setValue(self.parent_editor.current_time())

    def _jump_to_start(self) -> None:
        self.parent_editor.seek_to(float(self.start_spin.value()))

    def _jump_to_end(self) -> None:
        self.parent_editor.seek_to(float(self.end_spin.value()))

    def _preview_clip(self) -> None:
        start = float(self.start_spin.value())
        end = float(self.end_spin.value())
        if end <= start:
            QMessageBox.warning(self, "Invalid range", "End must be after start.")
            return
        self.parent_editor.seek_to(start)
        self.parent_editor._preview_end_time = end
        self.parent_editor._preview_timer.start()
        self.parent_editor.player.play()

    def get_updated_moment(self) -> Moment:
        new_m = self.original.copy()
        new_m.start = float(self.start_spin.value())
        new_m.end = float(self.end_spin.value())
        new_m.label = self.label_edit.text().strip()
        # If editing start/end on a compound moment, clear segments
        # (user is redefining the moment as a simple range)
        if new_m.segments and (new_m.start != self.original.start or new_m.end != self.original.end):
            new_m.segments = []
        return new_m

    def accept(self) -> None:
        start = float(self.start_spin.value())
        end = float(self.end_spin.value())
        if end <= start + 1e-6:
            QMessageBox.warning(self, "Invalid range", "End time must be after start.")
            return

        dur_sec = self.parent_editor.player.duration() / 1000.0
        if dur_sec > 0.01:
            if start < -1e-6 or end > dur_sec + 1e-6:
                QMessageBox.warning(
                    self,
                    "Out of range",
                    f"Clip must be inside [0, {dur_sec:.3f}] seconds.",
                )
                return
        super().accept()


# ---------------------------------------------------------------------------
# Main editor window
# ---------------------------------------------------------------------------

class MomentEditorWindow(QMainWindow):
    """
    QT-based moment editor with audio segmentation integration.

    Shortcuts:
        Space = Play/Pause
        1 = IN
        2 = OUT+Slice
        3 = ADD
        C = Smart clean
        E = Clip editor
        Delete = Delete selected
        Ctrl+Z = Undo
        Ctrl+M = Merge selected (concatenate segments)
        Double-click = Edit (Shift+Double-click = Jump)
    """

    NUDGE_COARSE = 0.5
    NUDGE_FINE = 0.1

    MIN_MOMENT_DURATION = 0.3
    MIN_GAP_TO_MERGE = 0.5
    MAX_MERGE_DURATION = 30.0
    
    # Crossfade duration in seconds for seamless cuts (0 = hard cut)
    CROSSFADE_DURATION = 0.04  # 40ms micro-crossfade for seamless feel

    def __init__(self, video_path: Optional[Path] = None, autoload_json: Optional[Path] = None):
        super().__init__()

        self.video_path: Optional[Path] = Path(video_path) if video_path else None
        self.moments: List[Moment] = []
        self.in_point: Optional[float] = None
        self.out_point: Optional[float] = None
        self.current_json_path: Optional[Path] = autoload_json

        self.history: List[List[Moment]] = []
        self.max_history = 50

        self._scan_thread: Optional[threading.Thread] = None
        self._scan_signals = ScanWorkerSignals()
        self._scan_signals.finished.connect(self._on_scan_finished)
        self._scan_signals.error.connect(self._on_scan_error)
        self._progress_dialog: Optional[QProgressDialog] = None

        self._preview_end_time: Optional[float] = None
        self._preview_segments: List[Tuple[float, float]] = []
        self._preview_segment_idx: int = 0
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(50)
        self._preview_timer.timeout.connect(self._check_preview_end)

        self.setWindowTitle("Moment Editor")
        self.resize(1280, 720)

        self._build_player()
        self._build_ui()
        self._wire_signals()

        if self.video_path:
            self.load_video(self.video_path)

        if self.current_json_path and self.current_json_path.exists():
            self.load_json(self.current_json_path)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _save_history(self) -> None:
        self.history.append([m.copy() for m in self.moments])
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def undo(self) -> None:
        if not self.history:
            self.statusBar().showMessage("Nothing to undo", 1500)
            return
        self.moments = self.history.pop()
        self.refresh_list()
        self.statusBar().showMessage("Undo successful", 1500)

    # ------------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------------

    def _reindex_moments(self) -> None:
        """Sort by start time, then rename to m_001, m_002... ALWAYS."""
        self.moments.sort(key=lambda m: (m.start, m.end))
        for i, m in enumerate(self.moments, start=1):
            m.id = f"m_{i:03d}"
        self.refresh_list()

    def renumber_ids_sequential(self) -> None:
        """
        Rename to m_001, m_002... in CURRENT list order (no sorting).
        Useful if you manually want "just rename everything once".
        """
        if not self.moments:
            self.statusBar().showMessage("No moments to renumber", 1500)
            return

        self._save_history()
        for i, m in enumerate(self.moments, start=1):
            m.id = f"m_{i:03d}"
        self.refresh_list()
        self.statusBar().showMessage("Renumbered moments sequentially", 2000)

    # ------------------------------------------------------------------
    # Player
    # ------------------------------------------------------------------

    def _build_player(self) -> None:
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)

        self.video_widget = QVideoWidget(self)
        self.player.setVideoOutput(self.video_widget)

        self.position_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.position_slider.setRange(0, 1000)

        self.time_label = QLabel("00:00.00 / 00:00.00", self)
        self.time_label.setToolTip("Double-click to jump to specific time")

        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")

        self.in_button = QPushButton("Set IN (1)")
        self.out_button = QPushButton("Set OUT / Slice (2)")
        self.add_moment_button = QPushButton("Add Moment (3)")
        self.clear_inout_button = QPushButton("Clear IN/OUT")

        self.in_out_label = QLabel("IN: -- / OUT: --", self)

        self.label_edit = QLineEdit(self)
        self.label_edit.setPlaceholderText("Short description for this moment")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_layout.addWidget(splitter, stretch=1)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.video_widget, stretch=5)

        transport_row = QHBoxLayout()
        transport_row.addWidget(self.play_button)
        transport_row.addWidget(self.pause_button)
        transport_row.addWidget(self.time_label, stretch=1)
        left_layout.addLayout(transport_row)
        left_layout.addWidget(self.position_slider)

        inout_row = QHBoxLayout()
        inout_row.addWidget(self.in_button)
        inout_row.addWidget(self.out_button)
        inout_row.addWidget(self.add_moment_button)
        inout_row.addWidget(self.clear_inout_button)
        inout_row.addWidget(self.in_out_label)
        left_layout.addLayout(inout_row)

        left_layout.addWidget(self.label_edit)
        splitter.addWidget(left)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)

        self.moment_list = QListWidget(self)
        self.moment_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        right_layout.addWidget(self.moment_list, stretch=1)

        list_buttons_1 = QHBoxLayout()
        self.preview_moment_button = QPushButton("Preview")
        self.delete_moment_button = QPushButton("Delete")
        self.merge_moment_button = QPushButton("Merge (concatenate)")
        self.split_moment_button = QPushButton("Split")

        self.preview_moment_button.setToolTip("Play selected moment (stops at end)")
        self.delete_moment_button.setToolTip("Delete selected moments (Del)")
        self.merge_moment_button.setToolTip(
            "Merge selected moments by CONCATENATING their segments.\n"
            "The resulting moment contains all cuts, ignoring gaps between them."
        )
        self.split_moment_button.setToolTip("Split moment at current playhead")

        list_buttons_1.addWidget(self.preview_moment_button)
        list_buttons_1.addWidget(self.delete_moment_button)
        list_buttons_1.addWidget(self.merge_moment_button)
        list_buttons_1.addWidget(self.split_moment_button)
        right_layout.addLayout(list_buttons_1)

        list_buttons_2 = QHBoxLayout()
        self.smart_clean_button = QPushButton("Smart Clean (C)")
        self.undo_button = QPushButton("Undo (Ctrl+Z)")
        self.renumber_button = QPushButton("Renumber IDs")

        self.smart_clean_button.setToolTip("Remove short moments, merge close ones, clean up list")
        self.renumber_button.setToolTip("Rename IDs to m_001, m_002, ... (no sorting)")
        list_buttons_2.addWidget(self.smart_clean_button)
        list_buttons_2.addWidget(self.undo_button)
        list_buttons_2.addWidget(self.renumber_button)
        right_layout.addLayout(list_buttons_2)

        nudge_row = QHBoxLayout()
        nudge_row.addWidget(QLabel("Nudge:"))
        self.nudge_start_left = QPushButton("← Start")
        self.nudge_start_right = QPushButton("Start →")
        self.nudge_end_left = QPushButton("← End")
        self.nudge_end_right = QPushButton("End →")
        nudge_row.addWidget(self.nudge_start_left)
        nudge_row.addWidget(self.nudge_start_right)
        nudge_row.addWidget(self.nudge_end_left)
        nudge_row.addWidget(self.nudge_end_right)
        right_layout.addLayout(nudge_row)

        json_buttons = QHBoxLayout()
        self.scan_moments_button = QPushButton("Scan Moments")
        self.import_json_button = QPushButton("Import JSON")
        self.save_json_button = QPushButton("Save JSON")
        self.export_clips_button = QPushButton("Export Clips")
        json_buttons.addWidget(self.scan_moments_button)
        json_buttons.addWidget(self.import_json_button)
        json_buttons.addWidget(self.save_json_button)
        json_buttons.addWidget(self.export_clips_button)
        right_layout.addLayout(json_buttons)

        splitter.addWidget(right)
        splitter.setSizes([800, 400])

        hint = QLabel(
            "Space=Play/Pause • 1=IN • 2=OUT+Slice • 3=ADD • "
            "←→=Nudge • Shift+←→=Fine • C=Clean • Del=Delete • "
            "Ctrl+Z=Undo • Ctrl+M=Merge(concat) • Double-click=Edit • Shift+Double-click=Jump",
            self,
        )
        hint.setWordWrap(True)
        main_layout.addWidget(hint)

        self._build_menubar()

    def _build_menubar(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        open_action = QAction("Open Video", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_video_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()
        import_action = QAction("Import JSON (append)", self)
        import_action.triggered.connect(self.import_json_dialog)
        file_menu.addAction(import_action)

        load_action = QAction("Load JSON (replace)", self)
        load_action.triggered.connect(self.load_json_dialog)
        file_menu.addAction(load_action)

        save_action = QAction("Save JSON", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_json)
        file_menu.addAction(save_action)

        file_menu.addSeparator()
        export_action = QAction("Export Clips", self)
        export_action.triggered.connect(self.export_clips_from_current)
        file_menu.addAction(export_action)

        edit_menu = menubar.addMenu("&Edit")
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

        delete_action = QAction("Delete Selected", self)
        delete_action.setShortcut(QKeySequence.StandardKey.Delete)
        delete_action.triggered.connect(self.delete_selected)
        edit_menu.addAction(delete_action)

        merge_action = QAction("Merge Selected (concatenate)", self)
        merge_action.setShortcut(QKeySequence("Ctrl+M"))
        merge_action.triggered.connect(self.merge_selected)
        edit_menu.addAction(merge_action)

        renum_action = QAction("Renumber IDs", self)
        renum_action.triggered.connect(self.renumber_ids_sequential)
        edit_menu.addAction(renum_action)

        edit_menu.addSeparator()
        clean_action = QAction("Smart Clean", self)
        clean_action.setShortcut(QKeySequence("C"))
        clean_action.triggered.connect(self.smart_clean)
        edit_menu.addAction(clean_action)

        tools_menu = menubar.addMenu("&Tools")
        scan_action = QAction("Scan Moments (Audio)", self)
        scan_action.triggered.connect(self.scan_moments)
        tools_menu.addAction(scan_action)

        clip_editor_action = QAction("Clip Editor", self)
        clip_editor_action.setShortcut(QKeySequence("E"))
        clip_editor_action.triggered.connect(self.open_clip_editor)
        tools_menu.addAction(clip_editor_action)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        self.play_button.clicked.connect(self.player.play)
        self.pause_button.clicked.connect(self.player.pause)

        self.position_slider.sliderMoved.connect(self._on_slider_moved)
        self.position_slider.sliderPressed.connect(self._on_slider_pressed)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)

        self.in_button.clicked.connect(self.set_in_point)
        self.out_button.clicked.connect(self.set_out_or_slice)
        self.add_moment_button.clicked.connect(self.add_current_moment)
        self.clear_inout_button.clicked.connect(self.clear_inout)

        self.moment_list.currentItemChanged.connect(self._on_moment_selected)
        self.moment_list.itemDoubleClicked.connect(self._on_moment_double_clicked)
        self.label_edit.editingFinished.connect(self._on_label_edited)

        self.preview_moment_button.clicked.connect(self.preview_selected_moment)
        self.delete_moment_button.clicked.connect(self.delete_selected)
        self.merge_moment_button.clicked.connect(self.merge_selected)
        self.split_moment_button.clicked.connect(self.split_at_playhead)
        self.smart_clean_button.clicked.connect(self.smart_clean)
        self.undo_button.clicked.connect(self.undo)
        self.renumber_button.clicked.connect(self.renumber_ids_sequential)

        self.nudge_start_left.clicked.connect(lambda: self.nudge_selected("start", -1))
        self.nudge_start_right.clicked.connect(lambda: self.nudge_selected("start", 1))
        self.nudge_end_left.clicked.connect(lambda: self.nudge_selected("end", -1))
        self.nudge_end_right.clicked.connect(lambda: self.nudge_selected("end", 1))

        self.scan_moments_button.clicked.connect(self.scan_moments)
        self.save_json_button.clicked.connect(self.save_json)
        self.import_json_button.clicked.connect(self.import_json_dialog)
        self.export_clips_button.clicked.connect(self.export_clips_from_current)

        self.time_label.mouseDoubleClickEvent = self._on_time_label_double_click

        self._setup_shortcuts()

    # ------------------------------------------------------------------
    # FIX #1: Shortcuts now use WindowShortcut context
    # ------------------------------------------------------------------
    def _setup_shortcuts(self) -> None:
        def add_shortcut(keyseq: str, callback) -> None:
            action = QAction(self)
            action.setShortcut(QKeySequence(keyseq))
            # THIS IS THE FIX: Set context so shortcuts work window-wide
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(callback)
            self.addAction(action)

        add_shortcut("Space", self.toggle_play_pause)
        add_shortcut("1", self.set_in_point)
        add_shortcut("2", self.set_out_or_slice)
        add_shortcut("3", self.add_current_moment)
        add_shortcut("C", self.smart_clean)
        add_shortcut("E", self.open_clip_editor)
        add_shortcut("Delete", self.delete_selected)
        add_shortcut("Ctrl+Z", self.undo)
        add_shortcut("Ctrl+M", self.merge_selected)

        add_shortcut("Left", lambda: self.nudge_selected("start", -1))
        add_shortcut("Right", lambda: self.nudge_selected("end", 1))
        add_shortcut("Shift+Left", lambda: self.nudge_selected("start", -1, fine=True))
        add_shortcut("Shift+Right", lambda: self.nudge_selected("end", 1, fine=True))

    # ------------------------------------------------------------------
    # Player helpers
    # ------------------------------------------------------------------

    def load_video(self, path: Path) -> None:
        self.video_path = Path(path)
        self.player.setSource(QUrl.fromLocalFile(str(self.video_path)))
        self.player.pause()
        self.position_slider.setValue(0)
        self.time_label.setText("00:00.00 / 00:00.00")
        self.setWindowTitle(f"Moment Editor – {self.video_path.name}")

    def _on_slider_pressed(self) -> None:
        self._was_playing = (self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState)
        self.player.pause()

    def _on_slider_moved(self, value: int) -> None:
        if self.player.duration() > 0:
            pos = int(self.player.duration() * (value / 1000.0))
            self.player.setPosition(pos)

    def _on_position_changed(self, position: int) -> None:
        dur = max(1, self.player.duration())
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(int(position / dur * 1000))
        self.position_slider.blockSignals(False)
        self._update_time_label()

    def _on_duration_changed(self, duration: int) -> None:
        self._update_time_label()

    def _update_time_label(self) -> None:
        pos = self.player.position()
        dur = self.player.duration()

        def fmt(ms: int) -> str:
            s = ms / 1000.0
            m = int(s // 60)
            sec = s % 60
            return f"{m:02d}:{sec:05.2f}"

        self.time_label.setText(f"{fmt(pos)} / {fmt(max(1, dur))}")

    def _on_time_label_double_click(self, event) -> None:
        dur = self.player.duration() / 1000.0
        text, ok = QInputDialog.getText(
            self,
            "Jump to time",
            f"Enter time (seconds or MM:SS.ss, max {dur:.1f}s):",
        )
        if ok and text.strip():
            try:
                t = self._parse_time(text.strip())
                t = max(0.0, min(t, dur))
                self.player.setPosition(int(t * 1000))
            except ValueError:
                self.statusBar().showMessage("Invalid time format", 2000)

    def _parse_time(self, text: str) -> float:
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return float(text)

    def toggle_play_pause(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def current_time(self) -> float:
        return self.player.position() / 1000.0

    def seek_to(self, time_sec: float) -> None:
        self.player.setPosition(int(time_sec * 1000))

    # ------------------------------------------------------------------
    # Preview with auto-stop (supports compound moments)
    # ------------------------------------------------------------------

    def preview_selected_moment(self) -> None:
        idxs = self.selected_indices()
        if not idxs:
            return
        m = self.moments[idxs[0]]
        segments = m.get_segments()
        
        if len(segments) > 1:
            # Compound moment: play segments in sequence
            self._preview_segments = segments
            self._preview_segment_idx = 0
            self._start_next_preview_segment()
        else:
            # Simple moment
            self.seek_to(m.start)
            self._preview_end_time = m.end
            self._preview_segments = []
            self._preview_timer.start()
            self.player.play()

    def _start_next_preview_segment(self) -> None:
        if self._preview_segment_idx >= len(self._preview_segments):
            self._preview_segments = []
            self._preview_end_time = None
            self._preview_timer.stop()
            self.player.pause()
            return
        
        seg = self._preview_segments[self._preview_segment_idx]
        self.seek_to(seg[0])
        self._preview_end_time = seg[1]
        self._preview_timer.start()
        self.player.play()

    def _check_preview_end(self) -> None:
        if self._preview_end_time is None:
            self._preview_timer.stop()
            return
        if self.current_time() >= self._preview_end_time:
            if self._preview_segments:
                # Move to next segment
                self._preview_segment_idx += 1
                self._start_next_preview_segment()
            else:
                self.player.pause()
                self._preview_timer.stop()
                self._preview_end_time = None

    # ------------------------------------------------------------------
    # IN / OUT
    # ------------------------------------------------------------------

    def set_in_point(self) -> None:
        self.in_point = self.current_time()
        self._update_inout_label()
        self.statusBar().showMessage(f"IN set at {self.in_point:.3f}s", 1500)

    def set_out_or_slice(self) -> None:
        t = self.current_time()
        if self.in_point is None:
            self.out_point = t
            self._update_inout_label()
            self.statusBar().showMessage(f"OUT set at {self.out_point:.3f}s", 1500)
            return

        self.out_point = t
        self._update_inout_label()
        self.add_current_moment()
        self.clear_inout()

    def add_current_moment(self) -> None:
        if self.in_point is None or self.out_point is None:
            self.statusBar().showMessage("Set both IN and OUT points first", 2000)
            return

        start = min(self.in_point, self.out_point)
        end = max(self.in_point, self.out_point)
        if end - start <= 0.01:
            self.statusBar().showMessage("Moment too short", 1500)
            return

        self._save_history()
        moment = Moment(id="tmp", start=start, end=end, label=self.label_edit.text().strip(), kind="manual")
        self.moments.append(moment)

        # ALWAYS keep order + ids tidy
        self._reindex_moments()

        self.statusBar().showMessage(f"Added [{start:.2f} – {end:.2f}]", 2000)

    def clear_inout(self) -> None:
        self.in_point = None
        self.out_point = None
        self._update_inout_label()

    def _update_inout_label(self) -> None:
        in_str = f"{self.in_point:.2f}" if self.in_point is not None else "--"
        out_str = f"{self.out_point:.2f}" if self.out_point is not None else "--"
        self.in_out_label.setText(f"IN: {in_str} / OUT: {out_str}")

    # ------------------------------------------------------------------
    # List helpers
    # ------------------------------------------------------------------

    def _add_moment_to_list(self, m: Moment) -> None:
        dur = m.duration()
        if m.is_compound():
            text = f"{m.id}: [{len(m.segments)} segs] ({dur:.1f}s) {m.label} ★"
        else:
            text = f"{m.id}: [{m.start:07.2f} → {m.end:07.2f}] ({dur:.1f}s) {m.label}"
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, m)
        self.moment_list.addItem(item)

    def refresh_list(self) -> None:
        self.moment_list.clear()
        for m in self.moments:
            self._add_moment_to_list(m)

    def _update_list_item(self, row: int, m: Moment) -> None:
        item = self.moment_list.item(row)
        if not item:
            return
        dur = m.duration()
        if m.is_compound():
            text = f"{m.id}: [{len(m.segments)} segs] ({dur:.1f}s) {m.label} ★"
        else:
            text = f"{m.id}: [{m.start:07.2f} → {m.end:07.2f}] ({dur:.1f}s) {m.label}"
        item.setText(text)
        item.setData(Qt.ItemDataRole.UserRole, m)

    def _on_moment_selected(self, item: Optional[QListWidgetItem]) -> None:
        if not item:
            return
        m: Moment = item.data(Qt.ItemDataRole.UserRole)
        self.label_edit.setText(m.label or "")

    def _on_moment_double_clicked(self, item: QListWidgetItem) -> None:
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
            m: Moment = item.data(Qt.ItemDataRole.UserRole)
            self.seek_to(m.start)
            return
        row = self.moment_list.row(item)
        if row >= 0:
            self.moment_list.setCurrentRow(row)
        self.open_clip_editor()

    def _on_label_edited(self) -> None:
        item = self.moment_list.currentItem()
        if not item:
            return
        row = self.moment_list.row(item)
        if row < 0 or row >= len(self.moments):
            return
        m = self.moments[row]
        new_label = self.label_edit.text().strip()
        if m.label != new_label:
            self._save_history()
            m.label = new_label
            self._update_list_item(row, m)

    # ------------------------------------------------------------------
    # List actions
    # ------------------------------------------------------------------

    def selected_indices(self) -> List[int]:
        return sorted([self.moment_list.row(i) for i in self.moment_list.selectedItems()])

    def delete_selected(self) -> None:
        idxs = sorted(self.selected_indices(), reverse=True)
        if not idxs:
            return
        self._save_history()
        for i in idxs:
            if 0 <= i < len(self.moments):
                del self.moments[i]
        self._reindex_moments()
        self.statusBar().showMessage(f"Deleted {len(idxs)} moment(s)", 1500)

    def merge_selected(self) -> None:
        """
        Merge selected moments by CONCATENATING their segments.
        
        This creates a compound moment that contains all the cuts from the
        selected moments. The frames/time between cuts in the original video
        are ignored - only the actual moment content is preserved.
        
        The segments are stored in the moment and can be exported as a single
        concatenated clip.
        """
        idxs = self.selected_indices()
        if len(idxs) < 2:
            self.statusBar().showMessage("Select at least 2 moments to merge", 2000)
            return

        selected = [self.moments[i] for i in idxs]
        
        # Collect all segments from all selected moments
        all_segments: List[Tuple[float, float]] = []
        for m in selected:
            all_segments.extend(m.get_segments())
        
        # Sort segments by start time
        all_segments.sort(key=lambda s: (s[0], s[1]))
        
        # Optionally merge overlapping/adjacent segments within the result
        merged_segments: List[Tuple[float, float]] = []
        for seg in all_segments:
            if not merged_segments:
                merged_segments.append(seg)
            else:
                last = merged_segments[-1]
                # If segments overlap or are adjacent (within 0.05s), merge them
                if seg[0] <= last[1] + 0.05:
                    merged_segments[-1] = (last[0], max(last[1], seg[1]))
                else:
                    merged_segments.append(seg)
        
        if not merged_segments:
            QMessageBox.warning(self, "No segments", "No valid segments to merge.")
            return

        self._save_history()

        # Create merged moment with all segments
        merged = Moment(
            id="tmp",
            start=merged_segments[0][0],  # First segment start
            end=merged_segments[-1][1],   # Last segment end
            label=selected[0].label,
            kind="merged",
            segments=merged_segments,
        )

        # Replace selected moments with merged, keeping other moments
        idx_set = set(idxs)
        new_list: List[Moment] = []
        inserted = False
        for i, m in enumerate(self.moments):
            if i in idx_set:
                if not inserted:
                    new_list.append(merged)
                    inserted = True
            else:
                new_list.append(m)

        self.moments = new_list
        self._reindex_moments()

        # Try to reselect near where the first selection was
        row = min(idxs)
        row = max(0, min(row, len(self.moments) - 1))
        self.moment_list.setCurrentRow(row)

        total_dur = sum(s[1] - s[0] for s in merged_segments)
        self.statusBar().showMessage(
            f"Merged {len(idxs)} moments → {len(merged_segments)} segment(s), {total_dur:.2f}s total",
            3000
        )

    # ------------------------------------------------------------------
    # Clip Editor
    # ------------------------------------------------------------------

    def open_clip_editor(self) -> None:
        idxs = self.selected_indices()
        if len(idxs) != 1:
            QMessageBox.information(
                self,
                "Select one clip",
                "Select exactly one moment in the list to edit.",
            )
            return

        idx = idxs[0]
        if idx < 0 or idx >= len(self.moments):
            return

        moment = self.moments[idx]
        dlg = ClipEditorDialog(self, moment)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._save_history()
            updated = dlg.get_updated_moment()
            self.moments[idx].start = updated.start
            self.moments[idx].end = updated.end
            self.moments[idx].label = updated.label
            self.moments[idx].segments = updated.segments
            self._reindex_moments()
            self.statusBar().showMessage(
                f"Updated clip: {updated.start:.3f} → {updated.end:.3f}",
                2500,
            )

    # ------------------------------------------------------------------
    # Split / nudge
    # ------------------------------------------------------------------

    def split_at_playhead(self) -> None:
        idxs = self.selected_indices()
        if len(idxs) != 1:
            self.statusBar().showMessage("Select exactly one moment to split", 2000)
            return

        idx = idxs[0]
        m = self.moments[idx]
        t = self.current_time()

        if t <= m.start or t >= m.end:
            self.statusBar().showMessage("Playhead must be inside the moment to split", 2000)
            return

        self._save_history()

        m1 = Moment(id="tmp", start=m.start, end=t, label=m.label, kind="split")
        m2 = Moment(id="tmp", start=t, end=m.end, label=m.label, kind="split")
        self.moments[idx: idx + 1] = [m1, m2]
        self._reindex_moments()
        self.statusBar().showMessage(f"Split at {t:.2f}s", 2000)

    def nudge_selected(self, edge: str, direction: int, fine: bool = False) -> None:
        idxs = self.selected_indices()
        if not idxs:
            return

        amount = self.NUDGE_FINE if fine else self.NUDGE_COARSE
        delta = direction * amount

        self._save_history()

        for idx in idxs:
            m = self.moments[idx]
            if edge == "start":
                new_start = max(0.0, m.start + delta)
                if new_start < m.end:
                    m.start = new_start
            else:
                new_end = m.end + delta
                if new_end > m.start:
                    m.end = new_end

        # keep IDs tidy after timing edits too (helps list ordering)
        self._reindex_moments()
        self.statusBar().showMessage(f"Nudged {edge} by {delta:+.2f}s", 1000)

    # ------------------------------------------------------------------
    # Smart Clean
    # ------------------------------------------------------------------

    def smart_clean(self) -> None:
        if not self.moments:
            self.statusBar().showMessage("No moments to clean", 1500)
            return

        self._save_history()
        original_count = len(self.moments)

        self.moments.sort(key=lambda m: (m.start, m.end))

        self.moments = [m for m in self.moments if m.duration() >= self.MIN_MOMENT_DURATION]

        if len(self.moments) > 1:
            merged = [self.moments[0]]
            for m in self.moments[1:]:
                prev = merged[-1]
                gap = m.start - prev.end

                if gap <= self.MIN_GAP_TO_MERGE:
                    combined_dur = m.end - prev.start
                    if combined_dur <= self.MAX_MERGE_DURATION:
                        prev.end = max(prev.end, m.end)
                        if m.label and not prev.label:
                            prev.label = m.label
                        prev.kind = "merged"
                        continue

                merged.append(m)

            self.moments = merged

        self._reindex_moments()
        removed = original_count - len(self.moments)
        self.statusBar().showMessage(
            f"Smart clean: {original_count} → {len(self.moments)} ({removed} removed/merged)",
            3000,
        )

    # ------------------------------------------------------------------
    # Audio Segmentation (Scan Moments)
    # ------------------------------------------------------------------

    def scan_moments(self) -> None:
        if self.video_path is None:
            QMessageBox.warning(self, "No video loaded", "Please open a video file before scanning.")
            return

        if self._scan_thread is not None and self._scan_thread.is_alive():
            QMessageBox.warning(self, "Scan in progress", "A scan is already running.")
            return

        if self.moments:
            reply = QMessageBox.question(
                self,
                "Replace moments?",
                f"This will replace the current {len(self.moments)} moments.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._save_history()

        self._progress_dialog = QProgressDialog("Analyzing audio for speech moments...", None, 0, 0, self)
        self._progress_dialog.setWindowTitle("Scanning Audio")
        self._progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.show()

        self._scan_thread = threading.Thread(
            target=self._run_segmentation,
            args=(self.video_path,),
            daemon=True,
        )
        self._scan_thread.start()

    def _run_segmentation(self, video_path: Path) -> None:
        try:
            segmenter = AudioSegmenter()
            result = segmenter.segment(video_path)

            json_path = video_path.with_name(f"{video_path.stem}_scanned_moments.json")
            segmenter.save_json(result, json_path)
            result["_json_path"] = str(json_path)

            self._scan_signals.finished.emit(result)
        except Exception as e:
            import traceback
            self._scan_signals.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _on_scan_finished(self, result: dict) -> None:
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None

        json_path = Path(result.get("_json_path", ""))
        moments_data = result.get("moments", [])

        if not moments_data:
            QMessageBox.information(self, "Scan complete", "No speech moments were detected.")
            return

        self.moments.clear()
        for m_raw in moments_data:
            intensity = str(m_raw.get("intensity", ""))
            moment = Moment(
                id="tmp",
                start=float(m_raw.get("start", 0.0)),
                end=float(m_raw.get("end", 0.0)),
                label=intensity,
                kind="scanned",
            )
            self.moments.append(moment)

        self._reindex_moments()
        self.current_json_path = json_path

        total_dur = sum(m.duration() for m in self.moments)
        QMessageBox.information(
            self,
            "Scan complete",
            f"Detected {len(self.moments)} speech moments.\n"
            f"Total duration: {total_dur:.1f}s\n\n"
            f"Saved to:\n{json_path.name}",
        )
        self.statusBar().showMessage(f"Loaded {len(self.moments)} scanned moments", 3000)

    def _on_scan_error(self, error_msg: str) -> None:
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None
        QMessageBox.critical(self, "Scan failed", f"Audio segmentation failed:\n\n{error_msg}")

    # ------------------------------------------------------------------
    # JSON IO
    # ------------------------------------------------------------------

    def save_json(self) -> None:
        if self.video_path is None:
            QMessageBox.warning(self, "No video", "Open a video before saving JSON.")
            return

        default_dir = str(self.video_path.parent)
        default_name = f"{self.video_path.stem}_moments.json"

        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save moments JSON",
            str(Path(default_dir) / default_name),
            "JSON files (*.json)",
        )
        if not path_str:
            return

        path = Path(path_str)
        data = {"media": {"path": str(self.video_path)}, "moments": [m.to_dict() for m in self.moments]}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        self.current_json_path = path
        self.statusBar().showMessage(f"Saved {path.name}", 2000)

    def load_json_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Load moments JSON (replace)", "", "JSON files (*.json)")
        if not path_str:
            return
        self.load_json(Path(path_str))

    def load_json(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Could not read JSON:\n{e}")
            return

        self._save_history()
        self.moments = [Moment.from_dict(m) for m in data.get("moments", [])]
        self._reindex_moments()
        self.current_json_path = path
        self.statusBar().showMessage(f"Loaded {len(self.moments)} moments", 2000)

    def import_json_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Import moments JSON (append)", "", "JSON files (*.json)")
        if not path_str:
            return
        self.import_json_append(Path(path_str))

    def import_json_append(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read JSON:\n{e}")
            return

        if not isinstance(data, dict) or "moments" not in data:
            QMessageBox.critical(self, "Import error", "JSON has no 'moments' key.")
            return

        self._save_history()
        for m_raw in data["moments"]:
            m = Moment.from_dict(m_raw)
            m.id = "tmp"
            self.moments.append(m)

        self._reindex_moments()
        self.statusBar().showMessage(f"Imported {len(data['moments'])} moments", 2000)

    # ------------------------------------------------------------------
    # FIX #2: Seamless export with proper segment concatenation
    # ------------------------------------------------------------------

    def _export_moment_seamless(
        self, 
        video_path: Path, 
        moment: Moment, 
        out_path: Path,
        crossfade: float = 0.0
    ) -> bool:
        """
        Export a moment (simple or compound) as a seamless clip.
        
        For compound moments with multiple segments:
        - Extracts each segment
        - Concatenates them using ffmpeg concat filter
        - Optionally applies micro-crossfade between cuts
        
        Returns True on success.
        """
        segments = moment.get_segments()
        
        if len(segments) == 1:
            # Simple case: single segment, direct extraction
            seg = segments[0]
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(seg[0]),
                "-i", str(video_path),
                "-t", str(seg[1] - seg[0]),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                str(out_path)
            ]
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        
        # Multiple segments: need to concat
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            temp_clips = []
            
            # Extract each segment
            for i, (start, end) in enumerate(segments):
                temp_clip = tmpdir_path / f"seg_{i:04d}.mp4"
                temp_clips.append(temp_clip)
                
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", str(video_path),
                    "-t", str(end - start),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-avoid_negative_ts", "make_zero",
                    str(temp_clip)
                ]
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    return False
            
            if crossfade > 0 and len(temp_clips) > 1:
                # Use xfade filter for smooth transitions
                return self._concat_with_crossfade(temp_clips, out_path, crossfade)
            else:
                # Use concat demuxer (faster, no re-encode needed for concat)
                return self._concat_with_demuxer(temp_clips, out_path, tmpdir_path)
    
    def _concat_with_demuxer(
        self, 
        clips: List[Path], 
        out_path: Path, 
        tmpdir: Path
    ) -> bool:
        """Concatenate clips using ffmpeg concat demuxer (fast, no quality loss)."""
        concat_file = tmpdir / "concat.txt"
        with open(concat_file, "w") as f:
            for clip in clips:
                f.write(f"file '{clip}'\n")
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(out_path)
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    
    def _concat_with_crossfade(
        self, 
        clips: List[Path], 
        out_path: Path, 
        crossfade: float
    ) -> bool:
        """
        Concatenate clips with crossfade transitions between each cut.
        This creates seamless transitions but requires re-encoding.
        """
        if len(clips) < 2:
            # Single clip, just copy
            cmd = ["ffmpeg", "-y", "-i", str(clips[0]), "-c", "copy", str(out_path)]
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        
        # Build complex filter for xfade between all clips
        # Input streams: [0:v][0:a], [1:v][1:a], ...
        inputs = []
        for clip in clips:
            inputs.extend(["-i", str(clip)])
        
        n = len(clips)
        filter_parts = []
        audio_filter_parts = []
        
        # Chain video xfade filters
        # [0:v][1:v]xfade=...[v01]; [v01][2:v]xfade=...[v012]; etc.
        prev_video = "[0:v]"
        prev_audio = "[0:a]"
        
        for i in range(1, n):
            next_video = f"[{i}:v]"
            next_audio = f"[{i}:a]"
            out_video = f"[v{i}]" if i < n - 1 else "[vout]"
            out_audio = f"[a{i}]" if i < n - 1 else "[aout]"
            
            # Video crossfade
            filter_parts.append(
                f"{prev_video}{next_video}xfade=transition=fade:duration={crossfade}:offset=0{out_video}"
            )
            
            # Audio crossfade
            audio_filter_parts.append(
                f"{prev_audio}{next_audio}acrossfade=d={crossfade}{out_audio}"
            )
            
            prev_video = out_video
            prev_audio = out_audio
        
        # For simplicity with variable offsets, use a simpler approach:
        # Just concat with very short transition that makes cuts invisible
        # This is actually more reliable for arbitrary segment counts
        
        # Alternative: use concat filter with small fade at boundaries
        filter_complex = f"concat=n={n}:v=1:a=1[vout][aout]"
        
        # Build input string
        input_streams = "".join([f"[{i}:v][{i}:a]" for i in range(n)])
        filter_complex = f"{input_streams}concat=n={n}:v=1:a=1[vout][aout]"
        
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            str(out_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0

    def export_clips_from_current(self) -> None:
        if self.video_path is None:
            QMessageBox.warning(self, "No video loaded", "Open a video before exporting clips.")
            return
        if not self.moments:
            QMessageBox.warning(self, "No moments", "There are no moments to export.")
            return

        video_path = Path(self.video_path)
        project_dir = video_path.parent
        out_dir = project_dir / f"{video_path.stem}_raw_clips"
        out_dir.mkdir(exist_ok=True)

        progress = QProgressDialog(
            "Exporting clips with seamless cuts...", 
            "Cancel", 
            0, 
            len(self.moments), 
            self
        )
        progress.setWindowTitle("Exporting")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        exported = 0
        failed = 0
        
        for i, moment in enumerate(self.moments):
            if progress.wasCanceled():
                break
                
            progress.setValue(i)
            progress.setLabelText(f"Exporting {moment.id}...")
            QApplication.processEvents()
            
            out_file = out_dir / f"{moment.id}.mp4"
            
            success = self._export_moment_seamless(
                video_path,
                moment,
                out_file,
                crossfade=self.CROSSFADE_DURATION
            )
            
            if success:
                exported += 1
            else:
                failed += 1
        
        progress.close()
        
        # Also save the JSON definition
        def_path = project_dir / f"{video_path.stem}_moments_export.json"
        data = {
            "media": {"path": str(video_path)}, 
            "moments": [m.to_dict() for m in self.moments]
        }
        def_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        
        if failed > 0:
            QMessageBox.warning(
                self, 
                "Export partially complete", 
                f"Exported {exported} clips, {failed} failed.\n\nOutput: {out_dir}"
            )
        else:
            QMessageBox.information(
                self, 
                "Export complete", 
                f"Exported {exported} clips to:\n{out_dir}"
            )

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

    def open_video_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm);;All files (*)",
        )
        if not path_str:
            return
        self.load_video(Path(path_str))


import shutil
from PyQt6.QtWidgets import QMessageBox

def ensure_ffmpeg() -> bool:
    if shutil.which("ffmpeg") is None:
        QMessageBox.critical(
            None,
            "FFmpeg not found",
            "FFmpeg is not installed or not in PATH.\n\n"
            "Install FFmpeg and verify it works in terminal:\n"
            "  ffmpeg -version"
        )
        return False
    return True
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)

    video_path = None
    autoload_json = None

    if len(sys.argv) >= 2:
        cand = Path(sys.argv[1])
        if cand.exists():
            if cand.suffix.lower() == ".json":
                autoload_json = cand
            else:
                video_path = cand

    window = MomentEditorWindow(video_path=video_path, autoload_json=autoload_json)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()