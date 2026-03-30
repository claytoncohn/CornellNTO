# Next Steps and Unresolved Issues

This document covers what would be needed to take this pipeline from prototype to production quality, open design questions, and a dedicated limitations section describing known failures and gaps in the current implementation.

---

## 1. Immediate Next Steps

### 1.1 Upgrade Speaker Diarization
- **Current state:** Speaker diarization is implemented using MFCC + agglomerative clustering (librosa + sklearn) and produces plausible speaker counts (4 for video 1, 5 for video 2). All utterances receive a `Speaker_N` label.
- **Limitation:** MFCC-based clustering is not speaker-discriminative across sessions — speaker IDs are not persistent or consistent between runs on different videos. It also conflates speakers who have similar vocal timbre.
- **Upgrade path:** `torch>=2.4` is already released and stable. The blocker is the numba/llvmlite macOS constraint in this environment that pins torch to 2.2.2. Resolve by decoupling numba from the PyTorch environment (separate conda env or Docker), then replace with pyannote.audio 3.x or resemblyzer for proper neural speaker embeddings.

### 1.2 Improve Screenshare Platform Detection (OCR)
- **Issue:** Tesseract OCR is currently returning `None` for platform name on sampled frames. Likely causes: low-contrast text, small font size at 1fps sampling, or the platform logo not being captured at the sampled timestamp.
- **Fix options:**
  - Increase frame sampling rate around screenshare segment boundaries.
  - Crop specifically to the platform logo region (bottom-right corner of screenshare panel) rather than running OCR over the full panel.
  - Use a template-matching approach for known platforms (Saga logo image) rather than relying on OCR.
  - Use a vision-language model (e.g., LLaVA, GPT-4V) for platform identification from frames.

### 1.3 Improve Emotion Inference — Multimodal Extension
- **Current state:** Emotion inference is audio-only, using prosodic features (RMS energy and speech rate). This produces `engaged` and `confused` labels for a subset of utterances.
- **Future work:**
  - **Facial expression recognition:** A face detection model robust to virtual backgrounds and VP8 compression (e.g., RetinaFace, InsightFace) feeding an expression classifier would add visual affect signals.
  - **Gesture and body language analysis:** mediapipe hand and body landmarks can detect affective gestures (animated hand movement, crossed arms, head nods) once webcam tile segmentation and face detection are reliable.
  - **Audio-visual fusion:** Models that jointly encode speech prosody, facial affect, and body language (e.g., MulT, AffectNet-based pipelines) would produce more accurate and nuanced emotion labels than any single modality alone.
  - **Vocabulary expansion:** The current two-label vocabulary (`engaged`, `confused`) should be expanded to include at least `neutral`, `frustrated`, `excited`, and `disengaged`, ideally using a model validated on classroom/tutoring data.

### 1.4 Participant Identity Extraction
- **Current state:** `participants` array is empty in all outputs. OCR of webcam tile labels is implemented in Stage 6 but returns 0 results — Tesseract cannot cleanly extract role-labeled name strings (`"Eli (Student)"`) from the full downsampled VP8-compressed frame.
- **Fix:** Isolate individual webcam tile regions by pixel position before OCR, apply higher-resolution cropping, and fuzzy-match results against known participant names. This requires knowing approximate tile geometry (deterministic in composite video layout).
- **Known signal:** Both videos have visible participant name labels with role annotations in parentheses — high-value signal for automatic role assignment once OCR targeting is improved.

### 1.5 Speaker-to-Participant Mapping
- Once diarization assigns speaker IDs (`Speaker_1`, `Speaker_2`, etc.) and OCR extracts participant names from webcam tiles, these two signals need to be cross-referenced by temporal overlap (active speaker during an utterance vs. participant whose webcam is active).
- This will require aligning diarization turns with frame timestamps.

---

## 2. Design Improvements

### 2.1 Whisper Model Size
- The prototype uses `whisper-small`. For a tutoring session corpus, `whisper-medium` or `whisper-large-v3` would produce meaningfully better transcription accuracy, especially for domain-specific math vocabulary (e.g., "Desmos", "linear equations", student names).
- Constraint: CPU-only inference for Whisper is slow. Upgrading the torch environment to support MPS (by resolving the numba conflict) would unblock GPU-accelerated Whisper.

### 2.2 Audio Normalization
- Currently extracting mono 16kHz WAV but not applying loudness normalization. Sessions with variable recording levels may produce inconsistent ASR quality.
- Consider applying `ffmpeg -filter:a loudnorm` or equivalent.

### 2.3 Screenshare Detection Threshold Tuning
- The brightness threshold (`BRIGHTNESS_THRESHOLD = 15.0`, mean pixel value on a 0–255 scale) is a coarse heuristic. False positives are possible if the screen is mostly black (e.g., loading screens) and false negatives if the screenshare panel is dark.
- A better approach: use frame-to-frame pixel variance in addition to mean brightness, or use motion detection to flag scene changes.

### 2.4 Transcript Deduplication / Hallucination Filtering
- Whisper is known to produce repetitive hallucinations on near-silent audio. Video 2 shows repeated single-word utterances ("you") at 30-second intervals during the first ~300 seconds of the session when participant webcams were off.
- A simple post-processing filter (e.g., drop utterances with `asr_confidence` below a threshold, or deduplicate consecutive identical short utterances) would improve transcript quality.

### 2.5 Overlap Detection
- Overlapping speech detection is implemented but relies on exact timestamp overlap from Whisper segments. Since Whisper segments are not always precisely aligned to speech boundaries, this may under-detect overlapping talk.
- A more robust approach uses diarization turn boundaries (which are more speech-boundary-aligned) for overlap detection.

---

## 3. Production Readiness

### 3.1 Environment Isolation
- In production, the numba/librosa environment should be completely decoupled from the PyTorch/Whisper/pyannote environment to remove the torch version constraint.
- Consider Docker containers or separate virtual environments for each stage.

### 3.2 PII Handling
- Source videos contain student faces and voices. The pipeline produces transcripts that may contain names. PII scrubbing (face blurring, name redaction) should be added before any output leaves the processing environment.
- Video files and extracted WAV audio are excluded from version control via `.gitignore`.

### 3.3 Idempotency and Caching
- The current pipeline re-runs all stages from scratch on each invocation. For production, add stage-level caching (e.g., skip Whisper if a transcript cache already exists for the same video ID and model version).

### 3.4 Error Recovery
- Stages currently log warnings and continue on failure. In production, failed stages should write structured error records to the output JSON so downstream consumers can distinguish "field not available" from "field was not attempted".

### 3.5 Scalability
- The pipeline processes one video sequentially. For a corpus of many sessions, a queue-based approach (e.g., Celery, Ray) with per-stage parallelism would be needed.

---

## 4. Limitations

This section documents known failures, gaps, and constraints in the current prototype. These are not design oversights — they represent deliberate decisions to defer complexity for the initial implementation, or constraints imposed by the environment.

### 4.1 Speaker Diarization Is MFCC-Based, Not Neural
**Current state:** Speaker diarization uses MFCC features + agglomerative clustering (Ward linkage, distance threshold 140). All utterances receive a `Speaker_N` label (4 speakers in video 1, 5 in video 2 — consistent with observed participant counts).

**Limitation:** MFCC-based clustering is less accurate than neural speaker embedding models (pyannote.audio, resemblyzer). Speaker IDs are not consistent across sessions and can conflate speakers with similar vocal characteristics.

**Root cause of original blocker:** pyannote.audio 3.x requires `torch>=2.4`; the numba/llvmlite constraint in this environment pins torch to `2.2.2`. Note: `torch>=2.4` is already released and stable — the constraint is purely environmental, not an upstream availability issue. MFCC approach was chosen as a functional, fully-offline baseline.

**Resolution path:** Decouple numba from the PyTorch environment (separate conda env or Docker), then replace clustering with pyannote.audio for neural-quality diarization.

---

### 4.2 No Participant Identity Extraction
**What failed:** The `participants` array is empty in all metadata outputs. Webcam tile labels (e.g., "Eli (Student)") were not extracted.

**Root cause:** OCR of webcam tile labels is implemented in Stage 6 but returns no usable results. Tesseract running on the full downsampled frame produces noisy garbage rather than clean name strings. The fix requires precisely cropping individual tile regions before OCR — not yet implemented.

**Impact:** Participant names and roles cannot be automatically assigned to speakers without manual input.

---

### 4.3 Screenshare Platform Not Identified
**What failed:** Tesseract OCR returned `null` for `platform_name` in all screenshare segments across both videos. Known platforms (Saga, Desmos-style tool) were not identified.

**Root cause:** Tesseract OCR on the sampled frames did not produce reliable text extraction, likely due to font rendering, resolution, or sampling timing (platform logo may not be visible in every sampled frame).

**Impact:** `platform_name` is `null` in all screenshare segment objects.

---

### 4.4 Nonverbal Signals Limited to Hand Raises (No Facial Gestures)
**Current state:** Stage 6 (visual analysis) is active and produces `hand_raise` events from mediapipe pose landmarks. Video 1: 160 events; Video 2: 295 events.

**Limitation:** Face mesh detection (for head nods, facial expressions) returns 0 detections. Lowering `min_detection_confidence` to 0.1 and applying 2× upscaling during debugging also produced 0 results. Cause: virtual backgrounds mask facial landmarks and VP8 compression degrades the low-resolution webcam tiles.

**Impact:** `nonverbal_events` contain only `hand_raise` events. Facial gesture signals (head nods, eyebrow raises, expressions) are unavailable.

**Resolution path:** Use a face detection model specifically designed for low-resolution or partially-occluded faces (e.g., RetinaFace, SCRFD) before feeding to mediapipe.

---

### 4.5 Emotion Inference Is Audio-Only
**Current state:** `inferred_emotions` are populated for a subset of utterances using prosodic features (RMS energy and speech rate). Labels: `engaged`, `confused`. Evidence type: `vocal_tone`.

**Limitation:** Only two emotion categories are inferred; inference is based on audio signal only. No facial expression or gesture data is incorporated.

**Root cause:** mediapipe face detection produced 0 results (see 4.4); visual emotion inference was not possible. Audio-based prosodic inference was implemented as a functional substitute.

**Future work:** Multimodal emotion inference combining facial expressions, gestures, and speech prosody. See Section 1.3 for full upgrade path.

---

### 4.6 Whisper Hallucinations on Near-Silent Audio
**What observed:** Video 2 contains repeated single-word utterances ("you") at regular ~30-second intervals during the first ~300 seconds of the session. During this period, participant webcams were off and no speech was audible.

**Root cause:** Whisper (especially smaller models) is known to hallucinate plausible-sounding tokens on near-silent or very low-energy audio segments rather than producing empty output.

**Impact:** Spurious utterances in the transcript during inactive session periods. Downstream consumers must treat low-confidence short utterances with caution.

---

### 4.7 Audio/Video Duration Mismatch (Video 2)
**What observed:** Video 2 has a 158-second gap (~2.6 min) at the end where video continues but audio has ended.

**What the pipeline does:** The mismatch is detected and logged in `warnings`. Whisper only transcribes the audio-covered portion; the final 2.6 minutes of video have no transcript entries.

**Impact:** Any speech that occurred in the final 2.6 minutes of the session is unrecoverable from the available audio stream. This is a data quality issue in the source recording, not a pipeline failure.

---

### 4.8 Screenshare Confidence Score is Fixed
**What observed:** All screenshare segments and events have `confidence: 0.7`. This is a fixed placeholder, not a computed score.

**Root cause:** The brightness-threshold heuristic does not produce a calibrated confidence value; 0.7 was chosen as a reasonable default indicating "heuristic detection, not verified."

**Impact:** Consumers cannot distinguish high-confidence detections from low-confidence ones using this field.

---

### 4.9 MPS Acceleration Not Used (Whisper on CPU)
**What observed:** Despite the system having an Apple M2 Max with MPS support and `DEVICE=mps` in `.env`, Whisper runs on CPU.

**Root cause:** `torch 2.2.2` does not support the sparse tensor operations required by Whisper's attention mechanism on MPS. This was fixed by forcing `whisper_device = "cpu"`.

**Impact:** Whisper transcription is significantly slower than it would be on MPS or CUDA. Video 1 (~51 min) took approximately 9 minutes; Video 2 (~63 min) took approximately 9 minutes. This scales roughly linearly with audio length.
