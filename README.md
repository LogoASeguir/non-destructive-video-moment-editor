# Non-Destructive Video Moment Editor (Python + FFmpeg)

A Qt-based video moment editor for creating, editing and exporting non-destructive clip segments

Features:
- IN / OUT slicing workflow
- Compound moment merging (concatenated segments)
- Smart clean (auto merge & remove micro-cuts)
- Audio-based speech segmentation
- JSON-based project structure
- Seamless clip export via FFmpeg

---

## Requirements

- Python 3.10+
- FFmpeg (must be available in system PATH)

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/LogoASeguir/non-destructive-video-moment-editor.git
cd non-destructive-video-moment-editor
2. Install Python dependencies
pip install -r requirements.txt
3. Install FFmpeg
Windows

Download FFmpeg from:
https://www.gyan.dev/ffmpeg/builds/

Extract and add the bin folder to your system PATH.

Verify installation:
ffmpeg -version
	
	***
macOS (Homebrew)
brew install ffmpeg

Linux
sudo apt install ffmpeg
	***

Running the Editor
python moment_editor.py

Optional:

python moment_editor.py /path/to/video.mp4


Controls

Space — Play / Pause
1 — Set IN
2 — Set OUT + Slice
3 — Add Moment
C — Smart Clean
E — Clip Editor
Ctrl+Z — Undo
Ctrl+M — Merge (concatenate)

Export

Exports clips as:
- Seamless single-segment clips (merge all concatenate at the end)
- Concatenated compound clips (batch render individual segments)

Optional micro-crossfade between cuts

Notes
FFmpeg must be accessible from terminal: ffmpeg -version

If FFmpeg is not found, the application will not start.

Thank you! Hopefully will be handy :)
