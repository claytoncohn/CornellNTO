# Validation Plan

This document describes how the outputs of the video processing pipeline should be validated — both for the prototype and for any future production deployment.

---

## 1. Metadata Validation

### Schema Conformance
- Each `metadata.json` is validated against `docs/schema_metadata.json` (JSON Schema draft-07) at export time via `jsonschema`.
- All required fields must be present and typed correctly.

### Spot-Check Procedure
For each processed video, manually verify the following fields against `ffprobe` output:

| Field | Verification Method |
|---|---|
| `duration_seconds` | `ffprobe -show_entries format=duration` |
| `resolution_width/height` | `ffprobe -show_entries stream=width,height` |
| `frame_rate` | `ffprobe -show_entries stream=r_frame_rate` |
| `audio_codec`, `video_codec` | `ffprobe -show_entries stream=codec_name` |
| `audio_sample_rate_hz` | `ffprobe -show_entries stream=sample_rate` |
| `file_size_bytes` | `stat -f%z <video_file>` |
| `audio_video_duration_mismatch` | Computed delta should match raw ffprobe durations |

### Expected Values (Current Corpus)

| Field | 1711656206762 | 1712170523563 |
|---|---|---|
| duration_seconds | 3039.5 | 3784.9 |
| resolution | 1280×990 | 1920×990 |
| frame_rate | 24.0 | 24.0 |
| video_codec | vp8 | vp8 |
| audio_codec | vorbis | vorbis |
| audio_video_duration_mismatch | true (5.5s) | true (158.0s) |

---

## 2. Transcript Validation

### Schema Conformance
- Each `transcript.json` is validated against `docs/schema_transcript.json` at export time.
- All utterance fields must be present; `start_time < end_time` for each utterance.

### Spot-Check Procedure

**Utterance count and coverage:**
- Confirm total utterance count is non-zero.
- Compute total transcribed duration: `sum(end_time - start_time for each utterance)`.
- This should be substantially less than `audio_duration_seconds` (silence is expected during inactive periods) but should not be near zero.

**Temporal ordering:**
- Utterances must be sorted by `start_time` in ascending order.
- No utterance `end_time` should exceed `audio_duration_seconds`.

**Manual ASR spot-check (sampling):**
- Sample 10 utterances evenly across the transcript timeline.
- Listen to the corresponding audio segment (using ffmpeg or a media player with timestamp seek).
- Verify transcribed text is a reasonable match to spoken content.
- Flag high-error utterances (e.g., hallucinated repetitions at low-signal segments).

**Known issue to watch for:** Whisper may hallucinate repeated short tokens (e.g., "you", "Bye.") at very quiet or silent audio segments. These appear in video 2 in the first ~300 seconds when participant webcams were off and the room was quiet. This is an expected limitation of ASR on near-silent audio.

**Screenshare segment validation:**
- Screenshare segment start/end times should be visually verifiable by seeking to those timestamps in the video.
- For video 1: screenshare should activate around 651s (10:51).
- For video 2: screenshare should be active from 0s to ~328s, then again from ~638s to end.

**Speaker assignment:**
- Speaker diarization uses MFCC + agglomerative clustering. Verify that `speaker_id` values are of the form `Speaker_1`, `Speaker_2`, etc. (not `Speaker_unknown`). Video 1 should have 4 unique speaker IDs; Video 2 should have 5. All utterances should have `speaker_role_confidence: "low"` (role assignment is deferred — see `docs/next_steps.md` §1.4–1.5).

---

## 3. End-to-End Regression Test (Recommended for Future Production)

Once neural speaker diarization (e.g., pyannote.audio) replaces the current MFCC baseline, the following regression checks should be added:

1. **Speaker count consistency:** Number of unique `speaker_id` values should not exceed the known number of session participants.
2. **Role assignment accuracy:** For sessions with known participant roles (tutor vs. student), compute the proportion of utterances correctly role-tagged. Ground truth can be derived from OCR of webcam tile labels.
3. **Screenshare platform detection:** OCR-extracted platform name should match known ground truth (Saga for video 2). Currently returning `null`; requires OCR improvement.
4. **Overlap detection:** Overlapping utterances should be cross-referenced and symmetrically populated in both `overlaps_with` arrays.

---

## 4. Statistical Validation Strategy (Recommended for Future Production)

The current validation approach relies on schema conformance checks, temporal consistency rules, and manual spot-checks on a small number of utterances. For a production system evaluated against a corpus of sessions, a more principled sampling and inference strategy is needed to make statistically defensible claims about output quality.

### 4.1 Sample Size Determination

Before any annotation effort, compute the required sample size per feature based on:

- **Effect size (Cohen's d or Cohen's h):** Define the minimum practically meaningful accuracy gap. For example, if diarization accuracy below 80% is unacceptable, the minimum detectable effect is the gap between 80% and the expected baseline.
- **Desired power (1 − β):** Typically 0.80 or 0.90. A power of 0.80 means an 80% chance of detecting a true effect of the specified size.
- **Significance level (α):** Typically 0.05. Use Bonferroni or Benjamini-Hochberg correction when testing multiple features simultaneously to control the family-wise error rate.
- **Variance estimate:** Use pilot data (e.g., the two prototype sessions) to estimate within-session variability for each metric before scaling to a larger corpus.

Standard formulas (or `statsmodels.stats.power` in Python) can translate these inputs into a minimum number of annotated utterances or sessions per feature.

### 4.2 Stratified Sampling

Uniform random sampling may under-represent low-frequency but high-importance events. For each feature, use stratified or importance-weighted sampling:

| Feature | Recommended Strata |
|---|---|
| ASR accuracy | By utterance duration bucket (< 2s, 2–10s, > 10s) and by Whisper `avg_logprob` decile |
| Speaker diarization | By speaker turn length; oversample short turns (< 1s) where clustering errors concentrate |
| Emotion inference | By inferred emotion label; ensure each label class is represented; oversample `confused` (low-frequency) |
| Nonverbal events | By event type; hand-raise events are the only populated type currently — sample proportionally once more types are added |
| Screenshare detection | By segment boundary proximity (within ±5s of a detected start/stop) to validate heuristic accuracy at transitions |

### 4.3 Annotation Rubric

Before any annotation begins, a shared rubric must be established for each subjective label. Without it, inter-rater agreement scores are uninterpretable — annotators may achieve high κ while agreeing on an undefined task.

**Emotion labels**

The pipeline currently emits `engaged` and `confused`. Human annotators should label the same using behaviorally grounded definitions anchored to tutoring research. Recommended starting point: BROMP (Baker-Rodrigo Observation Method Protocol for Affect), which defines affect states operationally for classroom/tutoring contexts. Minimal working definitions:

| Label | Observable criteria |
|---|---|
| `engaged` | Sustained on-task speech at normal or elevated rate; responsive to tutor/student prompts within 2 turns; no extended silence |
| `confused` | Slow, halting speech; hedging language ("um", "I don't know", "wait"); short incomplete utterances; explicit confusion markers ("I don't get it") |
| `neutral` | Neither of the above; on-task speech at normal rate with no affect signal |
| `frustrated` | Raised voice combined with negative content; repeated incorrect attempts with audible exasperation |

Annotators should label at the utterance level. Any utterance that does not clearly fit a label should be marked `ambiguous` and excluded from metric computation rather than forced into a category.

**Nonverbal events**

| Event | Definition | Exclusion criteria |
|---|---|---|
| `hand_raise` | Wrist visibly above shoulder level, held for ≥ 0.5s, in a webcam tile | Incidental arm movement (reaching, adjusting camera); wrist passes through but does not hold above shoulder |
| `nodding` | Repeated vertical head movement (≥ 2 cycles) in response to speech | Single head tilt; camera shake |

**Speaker diarization ground truth**

For DER computation, annotators produce a reference RTTM (Rich Transcription Time Mark) file — one row per contiguous speech turn with speaker label, start time, and duration. Recommended tool: ELAN or Audacity label tracks exported as RTTM. Each turn boundary should be agreed within ±0.5s before use as ground truth.

### 4.4 Inter-Rater Reliability

Any human annotation used as ground truth should be produced by at least two independent annotators. Compute inter-rater agreement before using annotations as validation labels:

- **Cohen's κ** for categorical labels (speaker identity, role, emotion, event type)
- **Intraclass correlation coefficient (ICC)** for continuous measurements (timestamp boundaries)
- Minimum acceptable κ: 0.70 (substantial agreement); exclude ambiguous items below threshold from evaluation rather than forcing a label

### 4.5 Per-Feature Metrics

| Feature | Primary Metric | Notes |
|---|---|---|
| ASR transcript | Word Error Rate (WER) + domain vocabulary coverage | Use `jiwer` for overall WER; separately track error rate on domain-specific terms (math vocabulary, platform names, student names) which Whisper-small systematically misses |
| Speaker diarization | Diarization Error Rate (DER) | Standard metric; includes missed speech, false alarm, speaker confusion; compute against RTTM ground truth (see §4.3) |
| Role assignment | Accuracy, precision/recall per role | Binary classification (tutor vs. student) once role assignment is implemented |
| Emotion inference | Precision/recall per emotion label, macro-F1, hallucination rate | Ground truth from annotators using rubric in §4.3; track hallucination rate separately (see §4.7) |
| Nonverbal events | Precision/recall at event level (IoU > 0.5 with annotated event window) | Standard detection metric; use rubric definitions from §4.3 as ground-truth criteria |
| Screenshare detection | Segment-level IoU and boundary error (seconds) | Compare detected segments against manually annotated ground-truth windows |

### 4.6 Confidence Intervals and Reporting

Point estimates alone are insufficient for production sign-off. For each metric, report:

- 95% confidence interval (bootstrap or exact binomial for proportions)
- Per-session variance — a system that achieves 85% WER on average but collapses to 60% on a specific session type is not production-ready
- Breakdown by condition (with/without screenshare active, session length quartile, speaker count)

### 4.7 Hallucination Rate

Whisper hallucinations on near-silent audio are a known failure mode (see `docs/next_steps.md` §4.6) but are not currently tracked as a metric. For production, operationalize and measure hallucination rate per session:

**Detection heuristic:** An utterance is a hallucination candidate if all of the following hold:
- `avg_logprob < -1.0` (low ASR confidence)
- `duration_seconds < 2.0`
- `word_count ≤ 1`
- RMS energy of the corresponding audio chunk is below a low-activity threshold (e.g., `rms < 0.005`)

**Metric:** Hallucination rate = (candidate hallucination count) / (total utterance count). Report per-session and overall.

**Acceptable threshold:** Define a maximum acceptable hallucination rate before production deployment (suggested starting point: < 2%). Sessions exceeding this threshold should trigger a manual review flag rather than passing automatically.

### 4.8 Threshold Sensitivity Analysis

The pipeline contains several empirically calibrated thresholds set on two sessions. Before deploying to new session types, validate that output quality is not highly sensitive to small threshold changes:

| Threshold | Location | Suggested sweep range |
|---|---|---|
| `BRIGHTNESS_THRESHOLD = 15.0` | Screenshare detection | 8.0 – 25.0 in steps of 2 |
| `distance_threshold = 140` | MFCC diarization clustering | 100 – 180 in steps of 10 |
| `EMOTION_MIN_CONFIDENCE = 0.50` | Emotion event gating | 0.40 – 0.70 in steps of 0.05 |
| `ACTION_MIN_CONFIDENCE = 0.50` | Nonverbal event gating | 0.40 – 0.70 in steps of 0.05 |

For each threshold, plot the target metric (DER, screenshare IoU, emotion F1) as a function of threshold value. A flat region around the chosen value indicates robustness; a steep slope indicates the system is sensitive and the threshold requires more careful calibration on a larger corpus before production use.

### 4.9 Reproducibility Check

The pipeline should be deterministic: re-running on the same input with the same configuration must produce bit-identical outputs. MFCC + Ward linkage clustering and Whisper on CPU are both deterministic. Verify this before production:

```bash
# Run pipeline twice, diff the outputs
python src/process_video.py --video videos/<id>.webm
cp output/<id>_transcript.json /tmp/run1_transcript.json
python src/process_video.py --video videos/<id>.webm
diff output/<id>_transcript.json /tmp/run1_transcript.json  # should produce no output
```

Any non-determinism should be investigated before production — it indicates either an uncontrolled random seed (e.g., if a non-deterministic model is introduced) or a race condition in intermediate file handling. If non-determinism is unavoidable (e.g., pyannote.audio with GPU), fix the random seed explicitly and document it.

---

## 5. Validation Tooling (Current)

| Check | Tool |
|---|---|
| Schema conformance | `jsonschema` (Python), run at export time |
| ffprobe baseline | `ffmpeg` / `ffprobe` CLI |
| Audio spot-check | ffmpeg seek + manual listening |
| Visual spot-check | ffplay or VLC, seeking to logged timestamps |
| Automated regression | Pytest suite (recommended for production) |

---

## 6. Acceptance Criteria (Prototype)

The prototype outputs are considered acceptable if:

- [x] Both `metadata.json` files pass schema validation
- [x] Both `transcript.json` files pass schema validation
- [x] Utterance `start_time` < `end_time` for all entries
- [x] No utterance `end_time` exceeds `audio_duration_seconds`
- [x] Audio/video duration mismatches are flagged in `warnings`
- [x] Screenshare segments are temporally consistent with manual video inspection
- [x] All fields present that are required by schema

All criteria met — see `Execution.md` Step 19 for full QA results. For statistical validation at scale, see §4.
