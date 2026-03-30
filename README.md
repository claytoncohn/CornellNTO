# Cornell NTO — Video Processing Pipeline

A prototype pipeline for processing tutoring-session `.webm` recordings. Extracts technical metadata, ASR transcripts with per-utterance timestamps, speaker diarization, screenshare detection, nonverbal events (hand raises), and prosodic emotion inference.

---

## Outputs

For each input video the pipeline produces two JSON artifacts:

| File | Contents |
|---|---|
| `output/<video_id>_metadata.json` | Technical metadata (codec, resolution, duration, participants, screenshare, warnings) |
| `output/<video_id>_transcript.json` | Per-utterance transcript with timestamps, speaker labels, screenshare events, nonverbal events |

Both files are validated against JSON Schemas in `docs/`.

---

## Requirements

- macOS (tested on Apple M2 Max)
- conda + the `CornellNTO` environment (see `environment.yml`)
- No Hugging Face token required — diarization uses offline MFCC clustering (no model downloads)

### Environment setup

```bash
conda env create -f environment.yml
conda activate CornellNTO
```

Copy `.env.example` to `.env` and configure:

```
DEVICE=mps
WHISPER_MODEL=small
ENABLE_DIARIZATION=true
ENABLE_NONVERBAL=true
ENABLE_OCR=true
SAVE_INTERMEDIATE_FILES=false
```

> **Note:** Whisper runs on CPU — MPS sparse tensor ops are unsupported in `torch==2.2.2`. See `docs/next_steps.md` for details.

---

## Usage

```bash
# Activate the environment first
conda activate CornellNTO

# Full pipeline (all stages)
python src/process_video.py --video videos/<video_id>_full_composite.webm

# Audio-only run (skip mediapipe visual analysis; screenshare detection still runs)
python src/process_video.py --video videos/<video_id>_full_composite.webm --skip-visual
```

> **Note:** If `conda activate` resolves to a different environment, use the full interpreter path: `/path/to/anaconda3/envs/CornellNTO/bin/python src/process_video.py ...`

Outputs are written to `output/`. Intermediate files (extracted audio, sampled frames) go to `tmp/` and are deleted after the run unless `SAVE_INTERMEDIATE_FILES=true`.

---

## Pipeline Stages

| Stage | Description |
|---|---|
| 1. Ingest | Validate input, run ffprobe, detect audio/video mismatch |
| 2. Metadata | Build `metadata.json` from ffprobe data |
| 3. Audio | Export normalized mono 16kHz WAV via ffmpeg |
| 4. Transcribe | Whisper ASR with per-segment timestamps |
| 5. Diarize | MFCC + agglomerative clustering speaker diarization (fully offline) |
| 6. Visual | Frame sampling, OCR webcam labels, mediapipe pose/face mesh |
| 7. Screenshare | Brightness-variance heuristic on upper panel; OCR platform ID |
| 8. Merge | Combine all signals into transcript; overlap detection |
| 9. Export | Write validated `metadata.json` and `transcript.json` |

---

## Documentation

| Document | Purpose |
|---|---|
| `Execution.md` | Full procedure log — what was done and why |
| `docs/validation_plan.md` | How to validate pipeline outputs |
| `docs/next_steps.md` | Known limitations, unresolved issues, production roadmap |
| `docs/schema_metadata.json` | JSON Schema for metadata output |
| `docs/schema_transcript.json` | JSON Schema for transcript output |

---

## Known Limitations

- Whisper runs on CPU (MPS sparse ops unsupported in torch 2.2.2) — transcription is slower than on GPU
- Speaker diarization uses MFCC clustering — less accurate than neural embedding models (e.g., pyannote); speaker IDs are not persistent across sessions
- Platform OCR returning `null` — Tesseract not reliably detecting platform name from frames
- Face detection returning 0 results — virtual backgrounds and VP8 compression artifacts prevent mediapipe from detecting webcam faces
- Emotion inference is audio-only (prosodic features) — no facial expression or gesture data
- `participants` array empty — OCR of webcam tile labels implemented but insufficient; VP8 compression and virtual backgrounds prevent clean text extraction

See `docs/next_steps.md` for the full limitations section and upgrade paths.
