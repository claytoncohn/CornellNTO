# Execution Log

This document records all procedures performed during the pipeline build, including what was done and why.

---

## Environment Setup

**Action:** Created a conda environment (`CornellNTO`) using `environment.yml`.

**How to reproduce:**
```bash
conda env create -f environment.yml
conda activate CornellNTO
```

**Why:** Establishing a reproducible environment before writing any pipeline code ensures consistent dependency resolution across machines.

**Tooling decisions:**
- `ffmpeg` and `tesseract` are installed via conda (prebuilt binaries required)
- `llvmlite`, `numba`, and `librosa` are installed via conda to avoid build failures on macOS ARM
- All other packages are installed via pip within the conda env
- PyTorch is installed from standard PyPI — the macOS wheel includes MPS support natively, no special index needed
- `DEVICE=mps` is set in `.env` to take advantage of Apple M2 Max hardware

**Caveats:**
- The environment includes more packages than will likely be used in this prototype (e.g., `speechbrain`, `mediapipe`, `xlsxwriter`, `jupyterlab`). This was intentional for the prototype phase to avoid repeated reinstalls during development.
- In a real-world production setting, unused packages would be audited and removed prior to deployment to reduce attack surface, image size, and dependency conflict risk.

---

## Repo Structure Setup

**Action:** Created project directory structure and moved source videos into `videos/`.

**Directories created:**
- `videos/` — source `.webm` files
- `src/` — pipeline script(s)
- `output/` — metadata and transcript JSONs
- `docs/` — validation plan and next-steps documents
- `logs/` — pipeline run logs
- `tmp/` — intermediate files (extracted audio, sampled frames, etc.)

**Videos moved:**
- `videos/1711656206762_full_composite.webm`
- `videos/1712170523563_full_composite.webm`

**Why:** Predictable artifact locations make the pipeline easier to implement, review, and reproduce. Separating source videos from generated outputs prevents accidental overwrites.

---

## Step 3: Video Inspection

**Action:** Ran `ffprobe` on both videos and sampled 2 frames per video (at 5min and 25min) to assess composite layout and screenshare visibility.

**Technical metadata:**

| | 1711656206762 | 1712170523563 |
|---|---|---|
| Duration (video) | 00:50:39 | 01:03:05 |
| Duration (audio) | 00:50:34 | 01:00:27 |
| Resolution | 1280×990 | 1920×990 |
| Video codec | VP8 | VP8 |
| Audio codec | Vorbis mono, 44100 Hz | Vorbis mono, 44100 Hz |
| Frame rate | 24 fps | 24 fps |
| Bitrate | ~340 kbps | ~355 kbps |
| Container | Matroska/WebM | Matroska/WebM |

**Composite layout observations:**

- Both videos use the same general layout: a screenshare/work panel occupying the upper ~70% of the frame, with a participant webcam strip along the bottom. Participant names and roles are visible as text labels on each webcam tile — a significant signal for role identification.
- A green vertical bar appears on the right edge of both videos throughout; this is a green screen background artifact from one or more participants and should be excluded from screenshare detection logic.

**Video 1 (1711656206762) specifics:**
- 4 participants visible: `Tutor Tutor (Tutor)` (no name listed), `Sebastian (Student)`, `Eli (Student)`, `Salvador (Student)`
- At 10:51: screen share is active digital math learning platform (Desmos-style activity builder) showing a grocery algebra problem; student work/avatars visible inside the platform in real time

**Video 2 (1712170523563) specifics:**
- 4 participants visible: `Tutor Rivers (Tutor)`, `Tutor Rivers (Tutor)` [black screen], `Eli (Student)`, `Sebastian (Student)`, `Salvador (Student)`, `Eli (Student)` [black screen] — the tutor and Eli both have two screens (each with one inactive)
- At the start: only the shared screen is visible (the participants' webcams are all black)
- Wider resolution (1920px) accommodates two side-by-side work panels in the upper portion
- Platform identified as **Saga** (logo visible bottom-right); math problem with student annotation visible
- At 2:21: Tutor Rivers appears.
- By 14:22: the three students, Eli, Sebastian, and Salvador, are all online and visible.
- **Audio/video duration mismatch:** audio ends at 01:00:27, video continues to 01:03:05 — a ~2.5 min gap. Transcription output must flag this; no speech data will be available for the final 2.5 minutes of video.

**Why:** Frame inspection before implementation ensures pipeline design is grounded in actual signal availability rather than assumptions. The visible participant name labels significantly simplify role mapping. The audio mismatch in video 2 is a known limitation that must be explicitly represented in outputs.

---

## Step 4: Output Schema Definition

**Action:** Wrote explicit JSON schemas for `metadata.json` and `transcript.json` before implementing any extraction logic. Schemas saved to `docs/schema_metadata.json` and `docs/schema_transcript.json`.

**Why:** Schema-first development ensures the pipeline produces coherent, validated outputs and makes downstream review straightforward. Defining schemas before writing code prevents ad hoc field additions that undermine consistency.

---
