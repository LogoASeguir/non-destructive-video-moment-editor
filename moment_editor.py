from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, QTimer, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
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

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_ffmpeg(parent=None) -> bool:
    if shutil.which("ffmpeg") is None:
        QMessageBox.critical(
            parent,
            "FFmpeg not found",
            "FFmpeg is not installed or not in PATH.\n\n"
            "Install FFmpeg, then verify in a terminal:\n"
            "  ffmpeg -version\n\n"
            "Windows tip:\n"
            "- Download a build (e.g. gyan.dev)\n"
            "- Add the 'bin' folder to PATH\n"
            "- Restart terminal/app",
        )
        return False
    return True


def ensure_ffplay(parent=None) -> bool:
    if shutil.which("ffplay") is None:
        QMessageBox.critical(
            parent,
            "ffplay not found",
            "ffplay is required for audio playback on systems where QtMultimedia audio fails.\n\n"
            "Your FFmpeg build may not include ffplay.\n\n"
            "Fix:\n"
            "- Install a full FFmpeg build that includes ffplay\n"
            "- Ensure ffplay.exe is in PATH (same folder as ffmpeg.exe usually)\n\n"
            "Verify in terminal:\n"
            "  ffplay -version",
        )
        return False
    return True


def make_qt_playback_proxy(src: Path) -> Path:
    """
    Build a Qt-friendly proxy MP4.
    We still do this so Qt video decoding stays stable,
    even though audio will be handled by ffplay in parallel.
    """
    src = Path(src).expanduser().resolve()
    tmp_dir = Path(tempfile.gettempdir()) / "moment_editor_qt_proxy"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    proxy = tmp_dir / f"{src.stem}__qtproxy.mp4"

    # reuse if up-to-date
    try:
        if proxy.exists() and proxy.stat().st_mtime >= src.stat().st_mtime:
            return proxy
    except Exception:
        pass

    # Keep video as-is, re-encode audio to AAC stereo 48k (safe container)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c:v", "copy",
        "-c:a", "aac",
        "-ac", "2",
        "-ar", "48000",
        "-b:a", "192k",
        str(proxy),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("Failed to build playback proxy:\n" + (r.stderr or ""))
    return proxy


def make_ffplay_audio_proxy(src: Path) -> Path:
    """
    Make an audio-only WAV proxy (PCM stereo 48k), which ffplay plays reliably.
    """
    src = Path(src).expanduser().resolve()
    tmp_dir = Path(tempfile.gettempdir()) / "moment_editor_audio_proxy"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = tmp_dir / f"{src.stem}__audioproxy.wav"

    try:
        if wav_path.exists() and wav_path.stat().st_mtime >= src.stat().st_mtime:
            return wav_path
    except Exception:
        pass

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn",
        "-ac", "2",
        "-ar", "48000",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("Failed to build audio proxy:\n" + (r.stderr or ""))
    return wav_path


# -----------------------------------------------------------------------------
# Audio playback sync (ffplay)
# -----------------------------------------------------------------------------

class FFPlayAudioSync:
    """
    Uses ffplay for audio because QtMultimedia audio can be broken on Windows (SWR layout errors).
    Strategy:
      - Build WAV proxy
      - On Play: start ffplay at current time
      - On Pause: stop ffplay
      - On Seek while playing: restart ffplay (debounced)
    """

    def __init__(self) -> None:
        self.audio_wav: Optional[Path] = None
        self.proc: Optional[subprocess.Popen] = None
        self.playing: bool = False
        self._restart_timer: Optional[QTimer] = None
        self._pending_seek_sec: float = 0.0

    def attach_restart_timer(self, timer: QTimer) -> None:
        self._restart_timer = timer

    def set_audio_source(self, wav_path: Path) -> None:
        self.audio_wav = Path(wav_path).expanduser().resolve()

    def stop(self) -> None:
        self.playing = False
        self._kill_proc()

    def _kill_proc(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=0.7)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        except Exception:
            pass
        self.proc = None

    def play_from(self, start_sec: float) -> None:
        if self.audio_wav is None:
            return
        self.playing = True
        self._kill_proc()

        # ffplay options:
        # -nodisp: no window
        # -autoexit: exit on EOF
        # -ss: start time
        # -loglevel error: silence spam
        cmd = [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel", "error",
            "-ss", f"{max(0.0, float(start_sec)):.3f}",
            str(self.audio_wav),
        ]
        # CREATE_NO_WINDOW on Windows to avoid flashing console
        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except Exception:
            self.proc = None

    def on_seek(self, new_time_sec: float) -> None:
        if not self.playing:
            return
        self._pending_seek_sec = float(new_time_sec)
        if self._restart_timer is not None:
            self._restart_timer.start()

    def _do_restart_from_pending(self) -> None:
        if not self.playing:
            return
        self.play_from(self._pending_seek_sec)


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

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
            return sum(max(0.0, e - s) for (s, e) in self.segments)
        return max(0.0, self.end - self.start)

    def get_segments(self) -> List[Tuple[float, float]]:
        return list(self.segments) if self.segments else [(self.start, self.end)]

    def is_compound(self) -> bool:
        return len(self.segments) > 1

    def to_dict(self) -> dict:
        d = {"id": self.id, "start": float(self.start), "end": float(self.end), "label": self.label, "kind": self.kind}
        if self.segments:
            d["segments"] = [[float(s), float(e)] for (s, e) in self.segments]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Moment":
        segs_raw = data.get("segments", [])
        segs = [(float(s), float(e)) for (s, e) in segs_raw] if segs_raw else []
        return cls(
            id=str(data.get("id", "")),
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            label=str(data.get("label", "")),
            kind=str(data.get("kind", "manual")),
            segments=segs,
        )

    def copy(self) -> "Moment":
        return Moment(self.id, float(self.start), float(self.end), self.label, self.kind, list(self.segments))


# -----------------------------------------------------------------------------
# Audio segmenter
# -----------------------------------------------------------------------------

class AudioSegmenter:
    def __init__(
        self,
        window_ms: float = 50.0,
        hop_ms: float = 25.0,
        noise_percentile: float = 25.0,
        speech_margin_db: float = 8.0,
        min_segment_duration: float = 0.15,
        short_gap: float = 0.8,
        long_gap: float = 2.5,
    ):
        self.window_ms = window_ms
        self.hop_ms = hop_ms
        self.noise_percentile = noise_percentile
        self.speech_margin_db = speech_margin_db
        self.min_segment_duration = min_segment_duration
        self.short_gap = short_gap
        self.long_gap = long_gap

    def ffmpeg_extract_mono_wav(self, input_media: Path, out_wav: Path) -> None:
        cmd = ["ffmpeg", "-y", "-i", str(input_media), "-vn", "-ac", "1", "-ar", "44100", "-f", "wav", str(out_wav)]
        subprocess.run(cmd, check=True, capture_output=True)

    def _read_wav_mono_float32(self, wav_path: Path) -> Tuple[np.ndarray, int]:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if nframes == 0:
            return np.zeros((0,), dtype=np.float32), sr

        if sampwidth == 1:
            x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            x = (x - 128.0) / 128.0
        elif sampwidth == 2:
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
            x = (x << 8) >> 8
            x = x.astype(np.float32) / 8388608.0
        elif sampwidth == 4:
            x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

        if channels > 1:
            x = x.reshape(-1, channels).mean(axis=1)
        return x, sr

    def compute_rms_db(self, wav_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
        data, sr = self._read_wav_mono_float32(wav_path)
        window = int(sr * self.window_ms / 1000.0)
        hop = int(sr * self.hop_ms / 1000.0)
        if window <= 0 or hop <= 0:
            raise ValueError("Invalid window/hop configuration")
        if len(data) < window:
            return np.zeros((0,), np.float32), np.zeros((0,), np.float32), sr

        frames = []
        times = []
        i = 0
        while i + window <= len(data):
            seg = data[i : i + window]
            frames.append(seg)
            times.append((i + window / 2) / sr)
            i += hop

        X = np.stack(frames, axis=0)
        t = np.array(times, dtype=np.float32)
        rms = np.sqrt(np.mean(X**2, axis=1) + 1e-12)
        db = 20 * np.log10(rms + 1e-12)
        return db.astype(np.float32), t, sr

    def label_activity(self, rms_db: np.ndarray) -> Tuple[np.ndarray, float]:
        if rms_db.size == 0:
            return np.zeros((0,), dtype=bool), 0.0
        noise_floor = float(np.percentile(rms_db, self.noise_percentile))
        thresh = float(noise_floor + self.speech_margin_db)
        return (rms_db > thresh), thresh

    def group_segments(self, active: np.ndarray, times: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        speech: List[Dict[str, Any]] = []
        silence: List[Dict[str, Any]] = []

        def flush(seg_type: str, a: int, b: int) -> None:
            t0 = float(times[a])
            t1 = float(times[b])
            dur = t1 - t0
            if dur < self.min_segment_duration:
                return
            seg = {"start": t0, "end": t1, "duration": dur, "type": seg_type}
            (speech if seg_type == "speech" else silence).append(seg)

        if active.size == 0:
            return speech, silence

        cur = "speech" if bool(active[0]) else "silence"
        start = 0
        for i in range(1, len(active)):
            t = "speech" if bool(active[i]) else "silence"
            if t != cur:
                flush(cur, start, i - 1)
                cur = t
                start = i
        flush(cur, start, len(active) - 1)
        return speech, silence

    def classify_intensity(self, rms_db: np.ndarray, times: np.ndarray, seg: Dict[str, Any]) -> str:
        mask = (times >= seg["start"]) & (times <= seg["end"])
        if not np.any(mask):
            seg["mean_db"] = None
            seg["peak_db"] = None
            return "unknown"
        vals = rms_db[mask]
        mean_db = float(vals.mean())
        seg["mean_db"] = mean_db
        seg["peak_db"] = float(vals.max())
        if mean_db < -30:
            return "very_soft"
        if mean_db < -24:
            return "soft"
        if mean_db < -18:
            return "normal"
        if mean_db < -12:
            return "loud"
        return "very_loud"

    def build_moments_and_fluxes(
        self, speech: List[Dict[str, Any]], silence: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        speech = sorted(speech, key=lambda s: s["start"])
        silence = sorted(silence, key=lambda s: s["start"])

        sil_idx = 0
        fluxes: List[Dict[str, Any]] = []
        moments: List[Dict[str, Any]] = []

        def flux_id(i: int) -> str:
            return f"flux_{i:03d}"

        def find_max_silence_between(t1: float, t2: float) -> float:
            nonlocal sil_idx
            max_gap = 0.0
            while sil_idx < len(silence) and silence[sil_idx]["end"] <= t1:
                sil_idx += 1
            j = sil_idx
            while j < len(silence) and silence[j]["start"] < t2:
                max_gap = max(max_gap, float(silence[j]["duration"]))
                j += 1
            return max_gap

        fi = 1
        cur_flux: Dict[str, Any] = {"id": flux_id(fi), "moments": [], "start": None, "end": None}

        for idx, seg in enumerate(speech, start=1):
            mid = f"m_{idx:03d}"
            moments.append(
                {
                    "id": mid,
                    "index": idx - 1,
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration": seg["duration"],
                    "intensity": seg.get("intensity", "unknown"),
                    "mean_db": seg.get("mean_db"),
                    "peak_db": seg.get("peak_db"),
                }
            )

            if idx > 1:
                prev = speech[idx - 2]
                gap = find_max_silence_between(prev["end"], seg["start"])
                if gap > self.long_gap:
                    if cur_flux["moments"]:
                        cur_flux["end"] = prev["end"]
                        cur_flux["duration"] = cur_flux["end"] - cur_flux["start"]
                        fluxes.append(cur_flux)
                    fi += 1
                    cur_flux = {"id": flux_id(fi), "moments": [], "start": None, "end": None}
                elif gap > self.short_gap:
                    cur_flux["has_pause"] = True

            if cur_flux["start"] is None:
                cur_flux["start"] = seg["start"]
            cur_flux["moments"].append(mid)

        if cur_flux["moments"]:
            cur_flux["end"] = speech[-1]["end"] if speech else 0.0
            cur_flux["duration"] = cur_flux["end"] - (cur_flux["start"] or 0.0)
            fluxes.append(cur_flux)

        return moments, fluxes

    def segment(self, input_media: Path) -> Dict[str, Any]:
        input_media = Path(input_media).expanduser().resolve()
        tmp_dir = tempfile.mkdtemp()
        tmp_wav = Path(tmp_dir) / "temp_audio.wav"
        try:
            self.ffmpeg_extract_mono_wav(input_media, tmp_wav)
            rms_db, times, sr = self.compute_rms_db(tmp_wav)
            active, thresh = self.label_activity(rms_db)
            speech, silence = self.group_segments(active, times)
            for seg in speech:
                seg["intensity"] = self.classify_intensity(rms_db, times, seg)
            moments, fluxes = self.build_moments_and_fluxes(speech, silence)
            total_speech = float(sum(s["duration"] for s in speech))
            total_silence = float(sum(s["duration"] for s in silence))
            total_duration = float(times[-1]) if len(times) > 0 else 0.0
            return {
                "input_media": str(input_media),
                "sample_rate": int(sr),
                "total_duration": total_duration,
                "total_speech": total_speech,
                "total_silence": total_silence,
                "speech_ratio": total_speech / max(total_duration, 0.001),
                "threshold_db": float(thresh),
                "speech_segments": speech,
                "silence_segments": silence,
                "moments": moments,
                "fluxes": fluxes,
                "moment_count": len(moments),
                "flux_count": len(fluxes),
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def save_json(self, result: Dict[str, Any], out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path


# -----------------------------------------------------------------------------
# Thread-signal bridge
# -----------------------------------------------------------------------------

class ScanWorkerSignals(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)


# -----------------------------------------------------------------------------
# Clip editor dialog
# -----------------------------------------------------------------------------

class ClipEditorDialog(QDialog):
    def __init__(self, parent: "MomentEditorWindow", moment: Moment):
        super().__init__(parent)
        self.parent_editor = parent
        self.original = moment
        self.moment = moment.copy()

        self.setWindowTitle(f"Clip Editor – {moment.id}")
        self.setModal(True)

        layout = QVBoxLayout(self)

        if moment.is_compound():
            seg_info = f"Compound moment with {len(moment.segments)} segments:\n"
            for i, (s, e) in enumerate(moment.segments, 1):
                seg_info += f"  Seg {i}: {s:.3f} → {e:.3f} ({e-s:.3f}s)\n"
            info = QLabel(f"Editing {moment.id}\nTotal duration: {moment.duration():.3f}s\n{seg_info}")
        else:
            info = QLabel(
                f"Editing {moment.id}\nCurrent: {moment.start:.3f} → {moment.end:.3f} ({moment.duration():.3f}s)"
            )
        info.setWordWrap(True)
        layout.addWidget(info)

        dur_sec = max(0.001, self.parent_editor.player.duration() / 1000.0)
        has_media = dur_sec > 0.01

        self.start_spin = QDoubleSpinBox(self)
        self.start_spin.setDecimals(3)
        self.start_spin.setRange(0.0, dur_sec)
        self.start_spin.setSingleStep(0.050)
        self.start_spin.setValue(self.moment.start)

        self.end_spin = QDoubleSpinBox(self)
        self.end_spin.setDecimals(3)
        self.end_spin.setRange(0.0, dur_sec)
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

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
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
        t = start + (value / 1000.0) * (end - start)
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
        if dur_sec > 0.01 and (start < -1e-6 or end > dur_sec + 1e-6):
            QMessageBox.warning(self, "Out of range", f"Clip must be inside [0, {dur_sec:.3f}] seconds.")
            return
        super().accept()


# -----------------------------------------------------------------------------
# Main editor window
# -----------------------------------------------------------------------------

class MomentEditorWindow(QMainWindow):
    NUDGE_COARSE = 0.5
    NUDGE_FINE = 0.1

    MIN_MOMENT_DURATION = 0.3
    MIN_GAP_TO_MERGE = 0.5
    MAX_MERGE_DURATION = 30.0

    CROSSFADE_DURATION = 0.0

    def __init__(self, video_path: Optional[Path] = None, autoload_json: Optional[Path] = None):
        super().__init__()

        self.video_path: Optional[Path] = Path(video_path) if video_path else None
        self._playback_path: Optional[Path] = None
        self._audio_wav_path: Optional[Path] = None

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

        # ffplay audio sync
        self._ffplay = FFPlayAudioSync()
        self._ffplay_restart = QTimer(self)
        self._ffplay_restart.setSingleShot(True)
        self._ffplay_restart.setInterval(120)  # debounce seek scrubbing
        self._ffplay_restart.timeout.connect(self._ffplay._do_restart_from_pending)
        self._ffplay.attach_restart_timer(self._ffplay_restart)

        self.setWindowTitle("Moment Editor")
        self.resize(1280, 720)

        self._build_player()
        self._build_ui()
        self._wire_signals()

        if self.video_path:
            self.load_video(self.video_path)

        if self.current_json_path and self.current_json_path.exists():
            self.load_json(self.current_json_path)

    # ---------------- History ----------------

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

    # ---------------- IDs ----------------

    def _reindex_moments(self) -> None:
        self.moments.sort(key=lambda m: (m.start, m.end))
        for i, m in enumerate(self.moments, start=1):
            m.id = f"m_{i:03d}"
        self.refresh_list()

    def renumber_ids_sequential(self) -> None:
        if not self.moments:
            self.statusBar().showMessage("No moments to renumber", 1500)
            return
        self._save_history()
        for i, m in enumerate(self.moments, start=1):
            m.id = f"m_{i:03d}"
        self.refresh_list()
        self.statusBar().showMessage("Renumbered moments sequentially", 2000)

    # ---------------- Player ----------------

    def _build_player(self) -> None:
        self.player = QMediaPlayer(self)

        self.audio_output = QAudioOutput(self)
        self.audio_output.setMuted(True)   # IMPORTANT: Qt audio OFF (we use ffplay)
        self.audio_output.setVolume(0.0)
        try:
            self.audio_output.setDevice(QMediaDevices.defaultAudioOutput())
        except Exception:
            pass

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

        self.audio_device_combo = QComboBox(self)
        self._audio_devices = QMediaDevices.audioOutputs()
        for dev in self._audio_devices:
            self.audio_device_combo.addItem(dev.description())

        default_dev = QMediaDevices.defaultAudioOutput()
        if default_dev:
            for i, dev in enumerate(self._audio_devices):
                if dev.id() == default_dev.id():
                    self.audio_device_combo.setCurrentIndex(i)
                    break

    def _on_audio_device_changed(self, idx: int) -> None:
        # This only affects ffplay indirectly (Windows global default). We'll keep it for UI consistency.
        try:
            dev = self._audio_devices[idx]
            self.audio_output.setDevice(dev)
            self.statusBar().showMessage(f"(Qt) Audio device selected: {dev.description()}", 2000)
        except Exception as e:
            self.statusBar().showMessage(f"Audio device set failed: {e}", 3000)

    def _on_player_error(self, err, err_str="") -> None:
        try:
            msg = self.player.errorString()
        except Exception:
            msg = str(err_str) if err_str else str(err)
        QMessageBox.warning(self, "Playback error", msg)

    # ---------------- UI ----------------

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
        transport_row.addWidget(QLabel("Audio:"))
        transport_row.addWidget(self.audio_device_combo)
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

        row1 = QHBoxLayout()
        self.preview_moment_button = QPushButton("Preview")
        self.delete_moment_button = QPushButton("Delete")
        self.merge_moment_button = QPushButton("Merge (concatenate)")
        self.split_moment_button = QPushButton("Split")
        row1.addWidget(self.preview_moment_button)
        row1.addWidget(self.delete_moment_button)
        row1.addWidget(self.merge_moment_button)
        row1.addWidget(self.split_moment_button)
        right_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.smart_clean_button = QPushButton("Smart Clean (C)")
        self.undo_button = QPushButton("Undo (Ctrl+Z)")
        self.renumber_button = QPushButton("Renumber IDs")
        row2.addWidget(self.smart_clean_button)
        row2.addWidget(self.undo_button)
        row2.addWidget(self.renumber_button)
        right_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Nudge:"))
        self.nudge_start_left = QPushButton("← Start")
        self.nudge_start_right = QPushButton("Start →")
        self.nudge_end_left = QPushButton("← End")
        self.nudge_end_right = QPushButton("End →")
        row3.addWidget(self.nudge_start_left)
        row3.addWidget(self.nudge_start_right)
        row3.addWidget(self.nudge_end_left)
        row3.addWidget(self.nudge_end_right)
        right_layout.addLayout(row3)

        row4 = QHBoxLayout()
        self.scan_moments_button = QPushButton("Scan Moments")
        self.import_json_button = QPushButton("Import JSON")
        self.save_json_button = QPushButton("Save JSON")
        self.export_clips_button = QPushButton("Export Clips")
        row4.addWidget(self.scan_moments_button)
        row4.addWidget(self.import_json_button)
        row4.addWidget(self.save_json_button)
        row4.addWidget(self.export_clips_button)
        right_layout.addLayout(row4)

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

    # ---------------- Wiring ----------------

    def _wire_signals(self) -> None:
        self.play_button.clicked.connect(self._play_clicked)
        self.pause_button.clicked.connect(self._pause_clicked)

        self.audio_device_combo.currentIndexChanged.connect(self._on_audio_device_changed)
        self.player.errorOccurred.connect(self._on_player_error)

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

        # stop ffplay if Qt stops unexpectedly
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)

    def _setup_shortcuts(self) -> None:
        def add_shortcut(keyseq: str, callback) -> None:
            action = QAction(self)
            action.setShortcut(QKeySequence(keyseq))
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

    # ---------------- Playback control (Qt video + ffplay audio) ----------------

    def _play_clicked(self) -> None:
        self.player.play()
        if self._audio_wav_path:
            self._ffplay.play_from(self.current_time())

    def _pause_clicked(self) -> None:
        self.player.pause()
        self._ffplay.stop()

    def _on_playback_state_changed(self, state) -> None:
        if state != QMediaPlayer.PlaybackState.PlayingState:
            self._ffplay.stop()

    # ---------------- Player helpers ----------------

    def load_video(self, path: Path) -> None:
        self.video_path = Path(path).expanduser().resolve()

        if not ensure_ffmpeg(self):
            return
        if not ensure_ffplay(self):
            return

        # Build proxies
        try:
            self._playback_path = make_qt_playback_proxy(self.video_path)
        except Exception as e:
            self._playback_path = self.video_path
            QMessageBox.warning(self, "Proxy build failed", f"{e}\n\nFalling back to original file for video.")

        try:
            self._audio_wav_path = make_ffplay_audio_proxy(self.video_path)
            self._ffplay.set_audio_source(self._audio_wav_path)
        except Exception as e:
            self._audio_wav_path = None
            QMessageBox.warning(self, "Audio proxy failed", f"{e}\n\nAudio playback will be unavailable.")

        self.player.setSource(QUrl.fromLocalFile(str(self._playback_path)))
        self.player.pause()
        self.position_slider.setValue(0)
        self._update_time_label()
        self.setWindowTitle(f"Moment Editor – {self.video_path.name}")

    def _on_slider_pressed(self) -> None:
        self.player.pause()
        self._ffplay.stop()

    def _on_slider_moved(self, value: int) -> None:
        if self.player.duration() > 0:
            pos = int(self.player.duration() * (value / 1000.0))
            self.player.setPosition(pos)
            # if scrubbing while playing, restart ffplay (debounced)
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self._ffplay.on_seek(self.current_time())

    def _on_position_changed(self, position: int) -> None:
        dur = max(1, self.player.duration())
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(int(position / dur * 1000))
        self.position_slider.blockSignals(False)
        self._update_time_label()

    def _on_duration_changed(self, _duration: int) -> None:
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
        text, ok = QInputDialog.getText(self, "Jump to time", f"Enter time (seconds or MM:SS.ss, max {dur:.1f}s):")
        if ok and text.strip():
            try:
                t = self._parse_time(text.strip())
                t = max(0.0, min(t, dur))
                self.player.setPosition(int(t * 1000))
                if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                    self._ffplay.on_seek(t)
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
            self._ffplay.stop()
        else:
            self.player.play()
            if self._audio_wav_path:
                self._ffplay.play_from(self.current_time())

    def current_time(self) -> float:
        return self.player.position() / 1000.0

    def seek_to(self, time_sec: float) -> None:
        self.player.setPosition(int(time_sec * 1000))
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._ffplay.on_seek(float(time_sec))

    # ---------------- Preview ----------------

    def preview_selected_moment(self) -> None:
        idxs = self.selected_indices()
        if not idxs:
            return
        m = self.moments[idxs[0]]
        segs = m.get_segments()
        if len(segs) > 1:
            self._preview_segments = segs
            self._preview_segment_idx = 0
            self._start_next_preview_segment()
        else:
            self.seek_to(m.start)
            self._preview_end_time = m.end
            self._preview_segments = []
            self._preview_timer.start()
            self.player.play()
            if self._audio_wav_path:
                self._ffplay.play_from(m.start)

    def _start_next_preview_segment(self) -> None:
        if self._preview_segment_idx >= len(self._preview_segments):
            self._preview_segments = []
            self._preview_end_time = None
            self._preview_timer.stop()
            self.player.pause()
            self._ffplay.stop()
            return
        s, e = self._preview_segments[self._preview_segment_idx]
        self.seek_to(s)
        self._preview_end_time = e
        self._preview_timer.start()
        self.player.play()
        if self._audio_wav_path:
            self._ffplay.play_from(s)

    def _check_preview_end(self) -> None:
        if self._preview_end_time is None:
            self._preview_timer.stop()
            return
        if self.current_time() >= self._preview_end_time:
            if self._preview_segments:
                self._preview_segment_idx += 1
                self._start_next_preview_segment()
            else:
                self.player.pause()
                self._ffplay.stop()
                self._preview_timer.stop()
                self._preview_end_time = None

    # ---------------- IN/OUT ----------------

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
        self.moments.append(Moment("tmp", start, end, self.label_edit.text().strip(), "manual"))
        self._reindex_moments()
        self.statusBar().showMessage(f"Added [{start:.2f} – {end:.2f}]", 2000)

    def clear_inout(self) -> None:
        self.in_point = None
        self.out_point = None
        self._update_inout_label()

    def _update_inout_label(self) -> None:
        ins = f"{self.in_point:.2f}" if self.in_point is not None else "--"
        outs = f"{self.out_point:.2f}" if self.out_point is not None else "--"
        self.in_out_label.setText(f"IN: {ins} / OUT: {outs}")

    # ---------------- List helpers ----------------

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
            self.refresh_list()

    # ---------------- List actions ----------------

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
        idxs = self.selected_indices()
        if len(idxs) < 2:
            self.statusBar().showMessage("Select at least 2 moments to merge", 2000)
            return
        selected = [self.moments[i] for i in idxs]
        all_segs: List[Tuple[float, float]] = []
        for m in selected:
            all_segs.extend(m.get_segments())
        all_segs.sort(key=lambda s: (s[0], s[1]))

        merged: List[Tuple[float, float]] = []
        for s, e in all_segs:
            if not merged:
                merged.append((s, e))
            else:
                ls, le = merged[-1]
                if s <= le + 0.05:
                    merged[-1] = (ls, max(le, e))
                else:
                    merged.append((s, e))

        if not merged:
            QMessageBox.warning(self, "No segments", "No valid segments to merge.")
            return

        self._save_history()
        new_m = Moment("tmp", merged[0][0], merged[-1][1], selected[0].label, "merged", merged)

        idx_set = set(idxs)
        out: List[Moment] = []
        inserted = False
        for i, m in enumerate(self.moments):
            if i in idx_set:
                if not inserted:
                    out.append(new_m)
                    inserted = True
            else:
                out.append(m)
        self.moments = out
        self._reindex_moments()

        total = sum(e - s for (s, e) in merged)
        self.statusBar().showMessage(f"Merged {len(idxs)} → {len(merged)} segs, {total:.2f}s total", 3000)

    def open_clip_editor(self) -> None:
        idxs = self.selected_indices()
        if len(idxs) != 1:
            QMessageBox.information(self, "Select one clip", "Select exactly one moment to edit.")
            return
        idx = idxs[0]
        if idx < 0 or idx >= len(self.moments):
            return
        m = self.moments[idx]
        dlg = ClipEditorDialog(self, m)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._save_history()
            updated = dlg.get_updated_moment()
            self.moments[idx].start = updated.start
            self.moments[idx].end = updated.end
            self.moments[idx].label = updated.label
            self.moments[idx].segments = updated.segments
            self._reindex_moments()
            self.statusBar().showMessage(f"Updated clip: {updated.start:.3f} → {updated.end:.3f}", 2500)

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
        m1 = Moment("tmp", m.start, t, m.label, "split")
        m2 = Moment("tmp", t, m.end, m.label, "split")
        self.moments[idx : idx + 1] = [m1, m2]
        self._reindex_moments()
        self.statusBar().showMessage(f"Split at {t:.2f}s", 2000)

    def nudge_selected(self, edge: str, direction: int, fine: bool = False) -> None:
        idxs = self.selected_indices()
        if not idxs:
            return
        delta = direction * (self.NUDGE_FINE if fine else self.NUDGE_COARSE)
        self._save_history()
        for i in idxs:
            m = self.moments[i]
            if edge == "start":
                ns = max(0.0, m.start + delta)
                if ns < m.end:
                    m.start = ns
            else:
                ne = m.end + delta
                if ne > m.start:
                    m.end = ne
        self._reindex_moments()
        self.statusBar().showMessage(f"Nudged {edge} by {delta:+.2f}s", 1000)

    def smart_clean(self) -> None:
        if not self.moments:
            self.statusBar().showMessage("No moments to clean", 1500)
            return
        self._save_history()
        original = len(self.moments)
        self.moments.sort(key=lambda m: (m.start, m.end))
        self.moments = [m for m in self.moments if m.duration() >= self.MIN_MOMENT_DURATION]

        if len(self.moments) > 1:
            merged = [self.moments[0]]
            for m in self.moments[1:]:
                prev = merged[-1]
                gap = m.start - prev.end
                if gap <= self.MIN_GAP_TO_MERGE:
                    combined = m.end - prev.start
                    if combined <= self.MAX_MERGE_DURATION:
                        prev.end = max(prev.end, m.end)
                        if m.label and not prev.label:
                            prev.label = m.label
                        prev.kind = "merged"
                        continue
                merged.append(m)
            self.moments = merged

        self._reindex_moments()
        removed = original - len(self.moments)
        self.statusBar().showMessage(f"Smart clean: {original} → {len(self.moments)} ({removed} removed/merged)", 3000)

    # ---------------- Audio scanning ----------------

    def scan_moments(self) -> None:
        if self.video_path is None:
            QMessageBox.warning(self, "No video loaded", "Please open a video file before scanning.")
            return
        if not ensure_ffmpeg(self):
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

        self._scan_thread = threading.Thread(target=self._run_segmentation, args=(self.video_path,), daemon=True)
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
            self._scan_signals.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _on_scan_finished(self, result: dict) -> None:
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None

        json_path = Path(result.get("_json_path", "")) if result.get("_json_path") else None
        moments_data = result.get("moments", [])

        self.moments.clear()
        for m_raw in moments_data:
            intensity = str(m_raw.get("intensity", ""))
            self.moments.append(
                Moment(
                    id="tmp",
                    start=float(m_raw.get("start", 0.0)),
                    end=float(m_raw.get("end", 0.0)),
                    label=intensity,
                    kind="scanned",
                )
            )

        self._reindex_moments()
        if json_path:
            self.current_json_path = json_path

        total_dur = sum(m.duration() for m in self.moments)
        saved_name = json_path.name if json_path else "(not saved)"
        QMessageBox.information(
            self,
            "Scan complete",
            f"Detected {len(self.moments)} speech moments.\n"
            f"Total duration: {total_dur:.1f}s\n\n"
            f"Saved to:\n{saved_name}",
        )
        self.statusBar().showMessage(f"Loaded {len(self.moments)} scanned moments", 3000)

    def _on_scan_error(self, error_msg: str) -> None:
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None
        QMessageBox.critical(self, "Scan failed", f"Audio segmentation failed:\n\n{error_msg}")

    # ---------------- JSON IO ----------------

    def save_json(self) -> None:
        if self.video_path is None:
            QMessageBox.warning(self, "No video", "Open a video before saving JSON.")
            return
        default_dir = str(self.video_path.parent)
        default_name = f"{self.video_path.stem}_moments.json"
        path_str, _ = QFileDialog.getSaveFileName(self, "Save moments JSON", str(Path(default_dir) / default_name), "JSON files (*.json)")
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

    # ---------------- Export ----------------

    def _export_moment_seamless(self, video_path: Path, moment: Moment, out_path: Path, crossfade: float = 0.0) -> bool:
        segs = moment.get_segments()
        if len(segs) == 1:
            s, e = segs[0]
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(s), "-i", str(video_path),
                "-t", str(max(0.0, e - s)),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                str(out_path),
            ]
            r = subprocess.run(cmd, capture_output=True)
            return r.returncode == 0

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            clips: List[Path] = []
            for i, (s, e) in enumerate(segs):
                clip = tmpdir_path / f"seg_{i:04d}.mp4"
                clips.append(clip)
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(s), "-i", str(video_path),
                    "-t", str(max(0.0, e - s)),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k",
                    "-avoid_negative_ts", "make_zero",
                    str(clip),
                ]
                r = subprocess.run(cmd, capture_output=True)
                if r.returncode != 0:
                    return False

            concat_file = tmpdir_path / "concat.txt"
            with concat_file.open("w", encoding="utf-8") as f:
                for c in clips:
                    f.write(f"file '{c.as_posix()}'\n")

            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(out_path)]
            r = subprocess.run(cmd, capture_output=True)
            return r.returncode == 0

    def export_clips_from_current(self) -> None:
        if self.video_path is None:
            QMessageBox.warning(self, "No video loaded", "Open a video before exporting clips.")
            return
        if not self.moments:
            QMessageBox.warning(self, "No moments", "There are no moments to export.")
            return
        if not ensure_ffmpeg(self):
            return

        video_path = Path(self.video_path)
        out_dir = video_path.parent / f"{video_path.stem}_raw_clips"
        out_dir.mkdir(exist_ok=True)

        progress = QProgressDialog("Exporting clips...", "Cancel", 0, len(self.moments), self)
        progress.setWindowTitle("Exporting")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        exported = 0
        failed = 0
        for i, m in enumerate(self.moments):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            progress.setLabelText(f"Exporting {m.id}...")
            QApplication.processEvents()

            out_file = out_dir / f"{m.id}.mp4"
            ok = self._export_moment_seamless(video_path, m, out_file, self.CROSSFADE_DURATION)
            if ok:
                exported += 1
            else:
                failed += 1

        progress.close()

        meta_path = video_path.parent / f"{video_path.stem}_moments_export.json"
        meta_path.write_text(
            json.dumps({"media": {"path": str(video_path)}, "moments": [m.to_dict() for m in self.moments]}, indent=2),
            encoding="utf-8"
        )

        if failed:
            QMessageBox.warning(self, "Export partially complete", f"Exported {exported} clips, {failed} failed.\n\nOutput:\n{out_dir}")
        else:
            QMessageBox.information(self, "Export complete", f"Exported {exported} clips to:\n{out_dir}")

    # ---------------- File dialogs ----------------

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


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------

def main() -> None:
    if not ensure_ffmpeg(None):
        sys.exit(1)

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