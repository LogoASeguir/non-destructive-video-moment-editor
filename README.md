# Non-Destructive Video Moment Editor (Python + FFmpeg)

A PyQt6 desktop application for creating and managing video "moments" without altering the original media file. Includes smart audio-based speech segmentation and seamless FFmpeg export.

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
```
```bash
2. Install Python dependencies
pip install -r requirements.txt
```
```bash
3. Install FFmpeg
Windows
Download FFmpeg from:
https://www.gyan.dev/ffmpeg/builds/

Extract and add the bin folder to your system PATH.

Verify installation:
ffmpeg -version

macOS (Homebrew)
brew install ffmpeg

Linux
sudo apt install ffmpeg
```

Running the Editor: python moment_editor.py or python moment_editor.py /path/to/video.mp4

```bash
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
```

Exports clips as:
- Seamless single-segment clips (merge all concatenate at the end)
- Concatenated compound clips (batch render individual segments)
- (Optional micro-crossfade between cuts)

```bash
FFmpeg must be accessible from terminal: ffmpeg -version
If FFmpeg is not found, the application will not start.
```
Thank you! Hopefully it will be handy :)

```bash
NOTE:
## Audio Scan (Experimental) AND Smart audio Clean

The built-in audio scanner is optimized for speech detection (interviews, podcasts, dialogue-driven content).

It may not perform well on:
- Music-heavy videos
- Continuous ambient sound
- High-noise recordings
```
