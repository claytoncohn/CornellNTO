"""
process_video.py

Tutoring session video processing pipeline.

Stages:
    1. ingest_video       — validate input, extract file-level metadata
    2. extract_metadata   — build metadata.json artifact
    3. extract_audio      — export normalized WAV for ASR/diarization
    4. transcribe_audio   — ASR via Whisper, produce timestamped segments
    5. diarize_speakers   — speaker diarization via MFCC features + agglomerative clustering
    6. analyze_visual     — nonverbal action + emotion inference via mediapipe / frame sampling
    7. detect_screenshare — screenshare on/off detection + coarse content classification
    8. merge_signals      — combine all signals into final transcript artifact
    9. export_artifacts   — write metadata.json and transcript.json to output/

Usage:
    python src/process_video.py --video videos/1711656206762_full_composite.webm
    python src/process_video.py --video videos/1712170523563_full_composite.webm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment + config
# ---------------------------------------------------------------------------

load_dotenv()

PIPELINE_VERSION = "0.1.0"

DEVICE              = os.getenv("DEVICE", "cpu")
WHISPER_MODEL       = os.getenv("WHISPER_MODEL", "small")
INPUT_DIR           = Path(os.getenv("INPUT_DIR", "./videos"))
OUTPUT_DIR          = Path(os.getenv("OUTPUT_DIR", "./output"))
TMP_DIR             = Path(os.getenv("TMP_DIR", "./tmp"))
LOG_DIR             = Path(os.getenv("LOG_DIR", "./logs"))
LOG_LEVEL           = os.getenv("LOG_LEVEL", "INFO")
AUDIO_SAMPLE_RATE   = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
FRAME_SAMPLE_RATE   = int(os.getenv("FRAME_SAMPLE_RATE", "1"))

ENABLE_DIARIZATION  = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"
ENABLE_NONVERBAL    = os.getenv("ENABLE_NONVERBAL", "true").lower() == "true"
ENABLE_EMOTION      = os.getenv("ENABLE_EMOTION", "true").lower() == "true"
ENABLE_SCREENSHARE  = os.getenv("ENABLE_SCREENSHARE", "true").lower() == "true"
ENABLE_OCR          = os.getenv("ENABLE_OCR", "true").lower() == "true"

EMOTION_MIN_CONFIDENCE    = float(os.getenv("EMOTION_MIN_CONFIDENCE", "0.50"))
ACTION_MIN_CONFIDENCE     = float(os.getenv("ACTION_MIN_CONFIDENCE", "0.50"))
SCREENSHARE_MIN_CONFIDENCE = float(os.getenv("SCREENSHARE_MIN_CONFIDENCE", "0.50"))

SAVE_INTERMEDIATE   = os.getenv("SAVE_INTERMEDIATE_FILES", "true").lower() == "true"

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(video_id: str) -> logging.Logger:
    """Configure file + console logging for a pipeline run."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{video_id}_pipeline.log"

    logger = logging.getLogger(video_id)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Stage 1 — Ingest video
# ---------------------------------------------------------------------------

def ingest_video(video_path: Path, logger: logging.Logger) -> dict:
    """
    Validate the input file and extract raw ffprobe metadata.

    Returns a dict with:
        - video_id (str)
        - video_path (Path)
        - ffprobe_data (dict)
        - warnings (list[str])
    """
    logger.info(f"[Stage 1] Ingesting video: {video_path}")

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_id = video_path.stem.split("_")[0]
    warnings = []

    # Run ffprobe
    import subprocess
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = f"ffprobe failed: {result.stderr}"
        logger.error(msg)
        raise RuntimeError(msg)

    ffprobe_data = json.loads(result.stdout)
    logger.info(f"[Stage 1] ffprobe succeeded. Streams: {len(ffprobe_data.get('streams', []))}")

    # Check for audio/video duration mismatch
    streams = ffprobe_data.get("streams", [])
    video_stream = next((s for s in streams if s["codec_type"] == "video"), None)
    audio_stream = next((s for s in streams if s["codec_type"] == "audio"), None)

    if not audio_stream:
        warnings.append("No audio stream found.")
        logger.warning("[Stage 1] No audio stream found.")

    if video_stream and audio_stream:
        v_dur = float(video_stream.get("tags", {}).get("DURATION", "0").replace(":", " ").split()[0] or
                      ffprobe_data["format"].get("duration", 0))
        a_dur = float(audio_stream.get("tags", {}).get("DURATION", "0").replace(":", " ").split()[0] or 0)
        # Parse HH:MM:SS.ms duration tags
        def parse_duration_tag(tag: str) -> float:
            try:
                parts = tag.strip().split(":")
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except Exception:
                return 0.0

        v_dur_tag = video_stream.get("tags", {}).get("DURATION", "")
        a_dur_tag = audio_stream.get("tags", {}).get("DURATION", "")
        if v_dur_tag and a_dur_tag:
            v_dur = parse_duration_tag(v_dur_tag)
            a_dur = parse_duration_tag(a_dur_tag)
            delta = abs(v_dur - a_dur)
            if delta > 5.0:
                msg = (f"Audio/video duration mismatch: video={v_dur:.1f}s, "
                       f"audio={a_dur:.1f}s, delta={delta:.1f}s")
                warnings.append(msg)
                logger.warning(f"[Stage 1] {msg}")

    return {
        "video_id": video_id,
        "video_path": video_path,
        "ffprobe_data": ffprobe_data,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Stage 2 — Extract metadata
# ---------------------------------------------------------------------------

def _extract_frames_ffmpeg(
    video_path: Path,
    video_id: str,
    fps: float,
    sample_rate: int,
    logger: logging.Logger,
) -> list[tuple[int, float, any]]:
    """
    Extract sampled frames from a video using ffmpeg (handles VP8/WebM reliably).
    Saves frames to tmp/<video_id>_frames/ and loads them with PIL.

    Returns a list of (frame_idx, timestamp_seconds, numpy_array) tuples.
    """
    import subprocess
    import numpy as np
    from PIL import Image

    frames_dir = TMP_DIR / f"{video_id}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg select filter to extract 1 frame every sample_rate seconds
    vf_filter = f"select='not(mod(n\\,{int(fps * sample_rate)}))',setpts=N/FRAME_RATE/TB"
    out_pattern = str(frames_dir / "frame_%06d.png")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf_filter,
        "-vsync", "vfr",
        "-update", "0",
        out_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"[ffmpeg frame extraction] Failed: {result.stderr[-300:]}")
        return []

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    sampled_frames = []
    for i, fpath in enumerate(frame_files):
        timestamp = i * sample_rate  # seconds
        frame_idx = int(i * fps * sample_rate)
        img = np.array(Image.open(fpath).convert("RGB"))
        # Convert RGB to BGR for OpenCV compatibility
        import cv2
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        sampled_frames.append((frame_idx, float(timestamp), bgr))

    logger.info(f"[ffmpeg frame extraction] Extracted {len(sampled_frames)} frames to {frames_dir}")
    return sampled_frames


def _parse_frame_rate(rate_str: str) -> float:
    """Safely parse a frame rate string like '24/1' or '30000/1001'."""
    try:
        num, den = rate_str.split("/")
        return int(num) / int(den)
    except Exception:
        return 0.0


def extract_metadata(ingest: dict, logger: logging.Logger) -> dict:
    """
    Build the metadata.json artifact from ffprobe data and frame observations.

    Returns a dict conforming to docs/schema_metadata.json.
    """
    logger.info("[Stage 2] Extracting metadata.")

    video_id    = ingest["video_id"]
    video_path  = ingest["video_path"]
    ffprobe     = ingest["ffprobe_data"]
    warnings    = list(ingest["warnings"])

    fmt     = ffprobe.get("format", {})
    streams = ffprobe.get("streams", [])
    v_stream = next((s for s in streams if s["codec_type"] == "video"), None)
    a_stream = next((s for s in streams if s["codec_type"] == "audio"), None)

    def parse_duration_tag(tag: str) -> float:
        try:
            parts = tag.strip().split(":")
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except Exception:
            return 0.0

    v_dur = parse_duration_tag(v_stream["tags"]["DURATION"]) if v_stream else float(fmt.get("duration", 0))
    a_dur = parse_duration_tag(a_stream["tags"]["DURATION"]) if a_stream else None
    a_v_delta = abs(v_dur - a_dur) if a_dur is not None else None
    mismatch = (a_v_delta is not None) and (a_v_delta > 5.0)

    metadata = {
        "video_id":                         video_id,
        "filename":                         video_path.name,
        "pipeline_version":                 PIPELINE_VERSION,
        "processed_at":                     datetime.now(timezone.utc).isoformat(),
        "duration_seconds":                 v_dur,
        "resolution_width":                 v_stream["width"] if v_stream else None,
        "resolution_height":                v_stream["height"] if v_stream else None,
        "frame_rate":                       _parse_frame_rate(v_stream["avg_frame_rate"]) if v_stream else None,
        "video_codec":                      v_stream["codec_name"] if v_stream else None,
        "audio_codec":                      a_stream["codec_name"] if a_stream else None,
        "audio_present":                    a_stream is not None,
        "audio_channels":                   a_stream["channels"] if a_stream else 0,
        "audio_sample_rate_hz":             int(a_stream["sample_rate"]) if a_stream else None,
        "audio_duration_seconds":           a_dur,
        "audio_video_duration_mismatch":    mismatch,
        "audio_video_duration_delta_seconds": a_v_delta,
        "container_format":                 fmt.get("format_name"),
        "file_size_bytes":                  int(fmt.get("size", 0)),
        "bitrate_kbps":                     int(fmt.get("bit_rate", 0)) / 1000,
        "composite_layout": {
            "layout_type":              "unknown",
            "webcam_strip_position":    "unknown",
            "screenshare_panel_count":  0,
            "notes":                    "Layout classification populated by visual analysis stage."
        },
        "screenshare_visible":      False,      # updated by Stage 7
        "screenshare_platform":     None,       # updated by Stage 7
        "participants":             [],          # updated by Stage 6
        "warnings":                 warnings,
    }

    logger.info(f"[Stage 2] Metadata built for {video_id}.")
    return metadata


# ---------------------------------------------------------------------------
# Stage 3 — Extract audio
# ---------------------------------------------------------------------------

def extract_audio(ingest: dict, logger: logging.Logger) -> Path:
    """
    Export the audio stream as a normalized mono WAV at AUDIO_SAMPLE_RATE.

    Returns the path to the WAV file.
    """
    logger.info("[Stage 3] Extracting audio.")

    video_id   = ingest["video_id"]
    video_path = ingest["video_path"]
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = TMP_DIR / f"{video_id}_audio.wav"

    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-sample_fmt", "s16",
        str(wav_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"[Stage 3] ffmpeg audio extraction failed: {result.stderr[-500:]}")
        raise RuntimeError("Audio extraction failed.")

    logger.info(f"[Stage 3] Audio saved to {wav_path}")
    return wav_path


# ---------------------------------------------------------------------------
# Stage 4 — Transcribe audio
# ---------------------------------------------------------------------------

def transcribe_audio(wav_path: Path, logger: logging.Logger) -> list[dict]:
    """
    Run Whisper ASR on the WAV file.

    Returns a list of segment dicts:
        [{"start": float, "end": float, "text": str, "avg_logprob": float}, ...]
    """
    # Whisper sparse tensor ops are not fully supported on MPS (PyTorch 2.2.x).
    # Force CPU for Whisper regardless of DEVICE setting.
    whisper_device = "cpu"
    logger.info(f"[Stage 4] Transcribing audio with Whisper ({WHISPER_MODEL}) on {whisper_device} "
                f"(MPS sparse ops unsupported; CPU used for Whisper).")

    import whisper
    model = whisper.load_model(WHISPER_MODEL, device=whisper_device)
    result = model.transcribe(
        str(wav_path),
        language="en",
        word_timestamps=False,
        verbose=False,
    )

    segments = [
        {
            "start":       seg["start"],
            "end":         seg["end"],
            "text":        seg["text"].strip(),
            "avg_logprob": seg.get("avg_logprob"),
        }
        for seg in result.get("segments", [])
    ]

    logger.info(f"[Stage 4] Transcription complete. {len(segments)} segments.")
    return segments


# ---------------------------------------------------------------------------
# Stage 5 — Diarize speakers
# ---------------------------------------------------------------------------

def diarize_speakers(wav_path: Path, logger: logging.Logger) -> list[dict]:
    """
    Speaker diarization using MFCC features + agglomerative clustering.

    Approach:
      1. Slide a 1.5s window across the audio (0.75s step).
      2. Compute 40-coefficient MFCCs per window (mean + std = 80-dim vector).
      3. Standardise features and cluster with AgglomerativeClustering (Ward, auto k).
      4. Merge consecutive same-speaker windows into contiguous turns.

    No external model downloads required; uses librosa and sklearn only.

    Returns a list of diarization turn dicts:
        [{"start": float, "end": float, "speaker": str}, ...]
    """
    if not ENABLE_DIARIZATION:
        logger.warning("[Stage 5] Diarization disabled via ENABLE_DIARIZATION=false.")
        return []

    logger.info("[Stage 5] Running speaker diarization (MFCC + agglomerative clustering).")

    try:
        import librosa
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler

        # Load audio as mono 16kHz
        audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)
        total_samples = len(audio)

        # Build sliding windows: 1.5s with 0.75s step
        window_samples = int(1.5 * sr)
        step_samples   = int(0.75 * sr)

        windows    = []
        timestamps = []
        pos = 0
        while pos + window_samples <= total_samples:
            windows.append(audio[pos : pos + window_samples])
            timestamps.append(pos / sr)
            pos += step_samples

        if len(windows) < 2:
            logger.warning("[Stage 5] Audio too short for diarization.")
            return []

        logger.info(f"[Stage 5] Extracting MFCC features for {len(windows)} windows...")

        # Per-window feature: mean + std of 40 MFCCs → 80-dim vector
        embeddings = []
        for window in windows:
            mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=40)
            feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
            embeddings.append(feat)
        embeddings = np.array(embeddings)  # [N, 80]

        # Standardise
        embeddings = StandardScaler().fit_transform(embeddings)

        # Auto-k agglomerative clustering (Ward linkage, Euclidean).
        # Threshold of 140 was calibrated on both session videos and recovers
        # 4 speakers for video 1 and 5 for video 2, matching observed participant counts.
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=140,
            metric="euclidean",
            linkage="ward",
        )
        labels = clustering.fit_predict(embeddings)

        # Merge consecutive same-speaker windows into contiguous turns
        turns = []
        cur_spk    = labels[0]
        turn_start = timestamps[0]
        for i in range(1, len(labels)):
            if labels[i] != cur_spk:
                turns.append({
                    "start":   turn_start,
                    "end":     timestamps[i],
                    "speaker": f"Speaker_{int(cur_spk) + 1}",
                })
                cur_spk    = labels[i]
                turn_start = timestamps[i]
        turns.append({
            "start":   turn_start,
            "end":     total_samples / sr,
            "speaker": f"Speaker_{int(cur_spk) + 1}",
        })

        n_speakers = len(set(labels.tolist()))
        logger.info(
            f"[Stage 5] Diarization complete. {len(turns)} turns, {n_speakers} speaker(s)."
        )
        return turns

    except Exception as e:
        logger.warning(
            f"[Stage 5] Diarization failed and will be skipped: {e}. "
            "Transcript will use 'Speaker_unknown' labels."
        )
        return []


# ---------------------------------------------------------------------------
# Stage 6 — Analyze visual events
# ---------------------------------------------------------------------------

def analyze_visual(ingest: dict, metadata: dict, logger: logging.Logger) -> dict:
    """
    Sample frames and extract:
      - composite layout classification
      - participant list from webcam tile labels (OCR)
      - nonverbal events (mediapipe pose/gesture)
      - inferred emotions (facial expression)

    Returns a dict:
        {
            "layout": dict,
            "participants": list[dict],
            "nonverbal_events": list[dict],   # [{speaker_id, timestamp, event_type, confidence}]
            "emotion_events": list[dict],      # [{speaker_id, timestamp, emotion, confidence, evidence_type}]
        }
    """
    logger.info("[Stage 6] Analyzing visual content.")

    video_path = ingest["video_path"]
    video_id   = ingest["video_id"]
    warnings   = []

    layout = {
        "layout_type":             "unknown",
        "webcam_strip_position":   "unknown",
        "screenshare_panel_count": 0,
        "notes":                   "",
    }
    participants    = []
    nonverbal_events = []
    emotion_events  = []

    # --- Frame sampling via ffmpeg (avoids OpenCV VP8/WebM read issues) ---
    fps = metadata.get("frame_rate") or 24.0
    sampled_frames = _extract_frames_ffmpeg(video_path, video_id, fps, FRAME_SAMPLE_RATE, logger)
    if not sampled_frames:
        logger.error("[Stage 6] Frame extraction failed.")
        return {
            "layout": layout,
            "participants": participants,
            "nonverbal_events": nonverbal_events,
            "emotion_events": emotion_events,
            "warnings": ["Frame extraction failed — visual analysis skipped."],
        }
    logger.info(f"[Stage 6] Sampled {len(sampled_frames)} frames.")

    # --- Layout classification (heuristic based on resolution) ---
    w = metadata.get("resolution_width", 0)
    h = metadata.get("resolution_height", 0)
    if w and h:
        if w >= 1900:
            layout["layout_type"]             = "dual_panel_plus_webcam_strip"
            layout["screenshare_panel_count"] = 2
        elif w >= 1200:
            layout["layout_type"]             = "screenshare_plus_webcam_strip"
            layout["screenshare_panel_count"] = 1
        layout["webcam_strip_position"] = "bottom"
        layout["notes"] = (
            "Green vertical bar visible on right edge throughout — "
            "green screen background artifact, excluded from screenshare detection."
        )

    # --- OCR: participant name labels from webcam strip ---
    if ENABLE_OCR and sampled_frames:
        participants = _extract_participants_from_frames(
            sampled_frames, video_id, w, h, logger
        )

    # --- Nonverbal / emotion (mediapipe) ---
    if ENABLE_NONVERBAL or ENABLE_EMOTION:
        nonverbal_events, emotion_events = _analyze_poses_and_emotions(
            sampled_frames, participants, logger
        )

    return {
        "layout":            layout,
        "participants":      participants,
        "nonverbal_events":  nonverbal_events,
        "emotion_events":    emotion_events,
        "warnings":          warnings,
    }


def _extract_participants_from_frames(
    sampled_frames: list,
    video_id: str,
    width: int,
    height: int,
    logger: logging.Logger,
) -> list[dict]:
    """
    OCR the webcam strip region to extract participant name labels.
    Uses tesseract on the bottom ~25% of the frame.
    Returns a list of participant dicts conforming to schema_metadata.json.
    """
    import pytesseract
    import cv2

    strip_top = int(height * 0.72)
    label_bottom = height
    seen_names: dict[str, dict] = {}

    # Sample a subset of frames for OCR (every 60 seconds is sufficient)
    ocr_frames = sampled_frames[::60] if len(sampled_frames) > 60 else sampled_frames

    for _, timestamp, frame in ocr_frames:
        strip = frame[strip_top:label_bottom, :, :]
        gray  = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config="--psm 6")

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Only accept lines that clearly identify a role — OCR on compressed
            # video produces too much noise to trust unidentified lines.
            role = "unknown"
            role_confidence = "low"
            role_evidence   = "OCR label"
            if "(Tutor)" in line:
                role = "tutor"
                role_confidence = "high"
                role_evidence = "label contains '(Tutor)'"
            elif "(Student)" in line:
                role = "student"
                role_confidence = "high"
                role_evidence = "label contains '(Student)'"
            else:
                # Skip lines with no role marker — almost certainly OCR noise.
                continue

            name_key = line.split("(")[0].strip()
            # Sanity-check: name should be a reasonable length and start with a letter.
            if not name_key or not name_key[0].isalpha() or len(name_key) > 50:
                continue

            if name_key not in seen_names:
                # Use "Participant_N" IDs to distinguish OCR-derived entries from
                # diarization-derived "Speaker_N" IDs — they cannot be reliably
                # cross-referenced without active-speaker alignment.
                seen_names[name_key] = {
                    "participant_id":  f"Participant_{len(seen_names) + 1}",
                    "display_name":    name_key,
                    "role":            role,
                    "role_confidence": role_confidence,
                    "role_evidence":   role_evidence,
                }

    participants = list(seen_names.values())
    logger.info(f"[Stage 6] OCR identified {len(participants)} participants.")
    return participants


def _analyze_poses_and_emotions(
    sampled_frames: list,
    participants: list[dict],
    logger: logging.Logger,
) -> tuple[list[dict], list[dict]]:
    """
    Run mediapipe pose + face mesh on sampled frames to detect nonverbal events
    and infer emotions. Falls back gracefully if mediapipe is unavailable.

    Returns (nonverbal_events, emotion_events).
    """
    nonverbal_events = []
    emotion_events   = []

    try:
        import mediapipe as mp
    except ImportError:
        logger.warning("[Stage 6] mediapipe not available. Nonverbal/emotion analysis skipped.")
        return nonverbal_events, emotion_events

    mp_pose     = mp.solutions.pose
    mp_face     = mp.solutions.face_mesh

    pose      = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=6,
                                  min_detection_confidence=0.5)

    import cv2

    frame_height = sampled_frames[0][2].shape[0] if sampled_frames else 990
    # Crop to webcam strip (bottom ~28%) to avoid spurious detections from screenshare content
    webcam_strip_top = int(frame_height * 0.72)

    for frame_idx, timestamp, frame in sampled_frames:
        # Only analyze the webcam strip region
        strip = frame[webcam_strip_top:, :, :]
        rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)

        # --- Pose: detect raised hands ---
        if ENABLE_NONVERBAL:
            pose_result = pose.process(rgb)
            if pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks.landmark
                left_wrist  = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                left_shoulder  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                confidence = 0.6
                if (left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y):
                    if confidence >= ACTION_MIN_CONFIDENCE:
                        nonverbal_events.append({
                            "event_type":       "hand_raise",
                            "timestamp":        timestamp,
                            "duration_seconds": None,
                            "confidence":       confidence,
                            "notes":            "Wrist landmark above shoulder landmark.",
                        })

        # --- Face mesh: coarse emotion inference from brow/mouth landmarks ---
        if ENABLE_EMOTION:
            face_result = face_mesh.process(rgb)
            if face_result.multi_face_landmarks:
                for face_landmarks in face_result.multi_face_landmarks:
                    emotion, confidence = _infer_emotion_from_landmarks(face_landmarks)
                    if emotion != "neutral" and confidence >= EMOTION_MIN_CONFIDENCE:
                        emotion_events.append({
                            "emotion":       emotion,
                            "confidence":    confidence,
                            "evidence_type": "facial_expression",
                            "timestamp":     timestamp,
                            "notes":         None,
                        })

    pose.close()
    face_mesh.close()

    logger.info(f"[Stage 6] Nonverbal events: {len(nonverbal_events)}, "
                f"Emotion events: {len(emotion_events)}.")
    return nonverbal_events, emotion_events


def _infer_emotion_from_landmarks(face_landmarks) -> tuple[str, float]:
    """
    Coarse emotion inference from face mesh landmark geometry.
    Uses brow raise (confusion/surprise) and mouth openness (engaged/excited).
    Conservative: returns 'neutral' unless signal is clear.

    Returns (emotion_label, confidence).
    """
    lm = face_landmarks.landmark

    # Brow raise heuristic: landmark 70 (left brow) vs 159 (left upper eyelid)
    try:
        brow_y   = lm[70].y
        eyelid_y = lm[159].y
        brow_raise = eyelid_y - brow_y  # positive = brow above eyelid

        mouth_top    = lm[13].y
        mouth_bottom = lm[14].y
        mouth_open   = mouth_bottom - mouth_top

        if brow_raise > 0.04:
            return "confused", 0.55
        if mouth_open > 0.06:
            return "engaged", 0.55
    except (IndexError, AttributeError):
        pass

    return "neutral", 1.0


# ---------------------------------------------------------------------------
# Stage 7 — Detect screenshare behavior
# ---------------------------------------------------------------------------

def detect_screenshare(ingest: dict, metadata: dict, logger: logging.Logger) -> list[dict]:
    """
    Detect screenshare segments by analyzing the upper panel of sampled frames.
    Uses pixel brightness variance to distinguish active content from black/blank panels.
    OCR identifies platform name from visible logos/text.

    Returns a list of ScreenshareSegment dicts conforming to schema_transcript.json.
    """
    if not ENABLE_SCREENSHARE:
        logger.warning("[Stage 7] Screenshare detection disabled.")
        return []

    logger.info("[Stage 7] Detecting screenshare segments.")

    import numpy as np

    video_path = ingest["video_path"]
    video_id   = ingest["video_id"]
    width  = metadata.get("resolution_width", 1280)
    height = metadata.get("resolution_height", 990)
    fps    = metadata.get("frame_rate") or 24.0

    # Screenshare panel occupies roughly top 72% of frame
    panel_bottom = int(height * 0.72)

    # Extract frames via ffmpeg (avoids OpenCV VP8/WebM read issues)
    sampled_frames = _extract_frames_ffmpeg(video_path, video_id, fps, FRAME_SAMPLE_RATE, logger)
    if not sampled_frames:
        logger.error("[Stage 7] Frame extraction failed — screenshare detection skipped.")
        return []

    segments      = []
    in_share      = False
    share_start   = None
    segment_idx   = 0
    platform_seen = None

    BRIGHTNESS_THRESHOLD = 15.0  # mean pixel value below this = blank/black panel

    for _, timestamp, frame in sampled_frames:
        panel = frame[:panel_bottom, :, :]

        # Exclude the green bar on the right edge (rightmost 5% of frame)
        panel = panel[:, : int(width * 0.93), :]

        mean_brightness = np.mean(panel)
        is_active = mean_brightness > BRIGHTNESS_THRESHOLD

        if is_active and not in_share:
            in_share    = True
            share_start = timestamp
            platform_seen = _identify_platform(panel, logger)
            logger.info(f"[Stage 7] Screenshare start at {timestamp:.1f}s "
                        f"(platform: {platform_seen})")

        elif not is_active and in_share:
            segment_idx += 1
            segments.append({
                "segment_id":    f"ss_{segment_idx:03d}",
                "start_time":    share_start,
                "end_time":      timestamp,
                "content_type":  "learning_platform" if platform_seen else "unknown",
                "platform_name": platform_seen,
                "shared_by":     None,
                "confidence":    0.7,
                "notes":         None,
            })
            logger.info(f"[Stage 7] Screenshare stop at {timestamp:.1f}s")
            in_share = False

    # Close open segment if video ends while sharing
    if in_share:
        segment_idx += 1
        segments.append({
            "segment_id":    f"ss_{segment_idx:03d}",
            "start_time":    share_start,
            "end_time":      None,
            "content_type":  "learning_platform" if platform_seen else "unknown",
            "platform_name": platform_seen,
            "shared_by":     None,
            "confidence":    0.7,
            "notes":         "Screenshare still active at end of video.",
        })

    logger.info(f"[Stage 7] Detected {len(segments)} screenshare segment(s).")
    return segments


def _identify_platform(panel, logger: logging.Logger) -> str | None:
    """
    Attempt to identify the platform being shared via OCR on the panel.
    Returns a lowercase platform name string or None.
    """
    if not ENABLE_OCR:
        return None
    try:
        import pytesseract
        import cv2
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config="--psm 11").lower()
        if "saga" in text:
            return "saga"
        if "desmos" in text:
            return "desmos"
        if "google" in text:
            return "google"
        if "docs" in text or "slides" in text:
            return "google_docs"
    except Exception as e:
        logger.warning(f"[Stage 7] OCR platform detection failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Audio emotion inference helper
# ---------------------------------------------------------------------------

def _infer_emotions_from_audio(
    wav_path: Path,
    asr_segments: list[dict],
    logger: logging.Logger,
) -> dict[int, dict]:
    """
    Infer coarse emotional states from prosodic features (RMS energy, pitch
    variation, speech rate) computed with librosa.

    Conservative vocabulary: only 'engaged' and 'confused' are emitted,
    and only when the acoustic signal is clear. Evidence type is 'vocal_tone'.

    Thresholds calibrated on the two session videos:
      - engaged:  rms > 0.025 AND speech_rate >= 2.5 wps
      - confused: rms < 0.012 AND speech_rate < 1.5 wps AND len(words) > 1

    Returns {segment_index: emotion_event_dict}.
    """
    if not ENABLE_EMOTION:
        return {}

    try:
        import librosa
        import numpy as np

        audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)

        results: dict[int, dict] = {}
        for i, seg in enumerate(asr_segments):
            start_s = int(seg["start"] * sr)
            end_s   = int(seg["end"]   * sr)
            chunk   = audio[start_s:end_s]
            if len(chunk) < int(0.3 * sr):
                continue

            rms   = float(np.sqrt(np.mean(chunk ** 2)))
            dur   = seg["end"] - seg["start"]
            words = seg.get("text", "").split()
            rate  = len(words) / dur if dur > 0 else 0.0

            emotion    = "neutral"
            confidence = 0.0
            notes_str  = f"rms={rms:.4f}, rate={rate:.1f}wps"

            if rms > 0.025 and rate >= 2.5:
                emotion    = "engaged"
                confidence = min(0.65, 0.50 + (rms - 0.025) * 3)
            elif rms < 0.012 and rate < 1.5 and len(words) > 1:
                emotion    = "confused"
                confidence = 0.50

            if emotion != "neutral" and confidence >= EMOTION_MIN_CONFIDENCE:
                results[i] = {
                    "emotion":       emotion,
                    "confidence":    round(confidence, 2),
                    "evidence_type": "vocal_tone",
                    "timestamp":     seg["start"],
                    "notes":         notes_str,
                }

        n = len(results)
        logger.info(
            f"[Stage 8] Audio emotion inference: {n}/{len(asr_segments)} segments "
            f"with non-neutral emotion."
        )
        return results

    except Exception as e:
        logger.warning(f"[Stage 8] Audio emotion inference failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Stage 8 — Merge signals
# ---------------------------------------------------------------------------

def merge_signals(
    ingest: dict,
    asr_segments: list[dict],
    diarization_turns: list[dict],
    visual: dict,
    screenshare_segments: list[dict],
    logger: logging.Logger,
    wav_path: Path | None = None,
) -> dict:
    """
    Combine ASR segments, diarization, nonverbal events, emotions, and screenshare
    into the final transcript artifact conforming to schema_transcript.json.

    Speaker assignment: for each ASR segment, find the diarization turn with
    maximum overlap and assign that speaker label.
    """
    logger.info("[Stage 8] Merging signals.")

    video_id     = ingest["video_id"]
    warnings     = list(ingest["warnings"])
    participants = visual.get("participants", [])

    # Build speaker_id → role map from participants
    role_map: dict[str, tuple[str, str]] = {}
    for p in participants:
        role_map[p["participant_id"]] = (p["role"], p["role_confidence"])

    # Index nonverbal and emotion events by approximate second for fast lookup
    nonverbal_index: dict[int, list] = {}
    for ev in visual.get("nonverbal_events", []):
        bucket = int(ev["timestamp"])
        nonverbal_index.setdefault(bucket, []).append(ev)

    # Use visual emotion events if available; fall back to audio-based inference
    visual_emotions = visual.get("emotion_events", [])
    if not visual_emotions and wav_path is not None:
        audio_emotions = _infer_emotions_from_audio(wav_path, asr_segments, logger)
    else:
        audio_emotions = {}

    emotion_index: dict[int, list] = {}
    for ev in visual_emotions:
        bucket = int(ev["timestamp"])
        emotion_index.setdefault(bucket, []).append(ev)

    # Build utterances
    utterances = []
    for i, seg in enumerate(asr_segments):
        start = seg["start"]
        end   = seg["end"]
        uid   = f"utt_{i:04d}"

        # Assign speaker via max-overlap diarization
        speaker_id = _assign_speaker(start, end, diarization_turns)
        role, role_conf = role_map.get(speaker_id, ("unknown", "low"))

        # Collect nonverbal events within this utterance window
        nonverbal = [
            ev for bucket in range(int(start), int(end) + 1)
            for ev in nonverbal_index.get(bucket, [])
            if start <= ev["timestamp"] <= end
        ]

        # Collect emotion events: visual index first, then audio fallback
        emotions = [
            {
                "emotion":       ev["emotion"],
                "confidence":    ev["confidence"],
                "evidence_type": ev["evidence_type"],
                "notes":         ev.get("notes"),
            }
            for bucket in range(int(start), int(end) + 1)
            for ev in emotion_index.get(bucket, [])
            if start <= ev["timestamp"] <= end
        ]
        if not emotions and i in audio_emotions:
            ev = audio_emotions[i]
            emotions = [{
                "emotion":       ev["emotion"],
                "confidence":    ev["confidence"],
                "evidence_type": ev["evidence_type"],
                "notes":         ev.get("notes"),
            }]

        # Collect screenshare events that fall within this utterance
        ss_events = []
        for seg_ss in screenshare_segments:
            ss_start = seg_ss["start_time"]
            ss_end   = seg_ss.get("end_time") or float("inf")
            if ss_start <= end and ss_end >= start:
                if ss_start >= start:
                    ss_events.append({
                        "event_type":    "share_start",
                        "timestamp":     ss_start,
                        "shared_by":     seg_ss.get("shared_by"),
                        "content_type":  seg_ss.get("content_type"),
                        "platform_name": seg_ss.get("platform_name"),
                        "confidence":    seg_ss.get("confidence", 0.7),
                    })
                if seg_ss.get("end_time") and seg_ss["end_time"] <= end:
                    ss_events.append({
                        "event_type":    "share_stop",
                        "timestamp":     seg_ss["end_time"],
                        "shared_by":     None,
                        "content_type":  None,
                        "platform_name": None,
                        "confidence":    seg_ss.get("confidence", 0.7),
                    })

        utterances.append({
            "utterance_id":           uid,
            "speaker_id":             speaker_id,
            "speaker_display_name":   None,  # enriched below
            "speaker_role":           role,
            "speaker_role_confidence": role_conf,
            "start_time":             start,
            "end_time":               end,
            "duration_seconds":       round(end - start, 3),
            "text":                   seg["text"],
            "asr_confidence":         seg.get("avg_logprob"),
            "nonverbal_events":       nonverbal,
            "inferred_emotions":      emotions,
            "screenshare_events":     ss_events,
            "overlaps_with":          [],  # filled in next pass
        })

    # Enrich display names
    pid_to_name = {p["participant_id"]: p["display_name"] for p in participants}
    for u in utterances:
        u["speaker_display_name"] = pid_to_name.get(u["speaker_id"])

    # Detect overlapping utterances
    for i, u in enumerate(utterances):
        for j, v in enumerate(utterances):
            if i == j:
                continue
            if u["start_time"] < v["end_time"] and u["end_time"] > v["start_time"]:
                if v["utterance_id"] not in u["overlaps_with"]:
                    u["overlaps_with"].append(v["utterance_id"])

    logger.info(f"[Stage 8] Merged {len(utterances)} utterances.")

    return {
        "video_id":           video_id,
        "pipeline_version":   PIPELINE_VERSION,
        "processed_at":       datetime.now(timezone.utc).isoformat(),
        "utterances":         utterances,
        "screenshare_segments": screenshare_segments,
        "warnings":           warnings,
    }


def _assign_speaker(start: float, end: float, turns: list[dict]) -> str:
    """
    Assign a speaker to the [start, end] interval by finding the diarization
    turn with maximum overlap. Returns 'Speaker_unknown' if no turns available.
    """
    if not turns:
        return "Speaker_unknown"

    best_speaker = "Speaker_unknown"
    best_overlap = 0.0

    for turn in turns:
        overlap = min(end, turn["end"]) - max(start, turn["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]

    return best_speaker


# ---------------------------------------------------------------------------
# Stage 9 — Export artifacts
# ---------------------------------------------------------------------------

def export_artifacts(
    metadata: dict,
    transcript: dict,
    video_id: str,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    """
    Write metadata.json and transcript.json to OUTPUT_DIR.

    Returns (metadata_path, transcript_path).
    """
    logger.info("[Stage 9] Exporting artifacts.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path   = OUTPUT_DIR / f"{video_id}_metadata.json"
    transcript_path = OUTPUT_DIR / f"{video_id}_transcript.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"[Stage 9] Metadata written to {metadata_path}")

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    logger.info(f"[Stage 9] Transcript written to {transcript_path}")

    return metadata_path, transcript_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tutoring session video processing pipeline."
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input .webm video file.",
    )
    parser.add_argument(
        "--skip-visual",
        action="store_true",
        help="Skip visual analysis (Stage 6). Useful for fast audio-only runs.",
    )
    args = parser.parse_args()

    video_path = args.video
    video_id   = video_path.stem.split("_")[0]
    logger     = setup_logging(video_id)

    logger.info(f"=== Pipeline start: {video_id} ===")
    logger.info(f"Device: {DEVICE} | Whisper: {WHISPER_MODEL} | "
                f"Diarization: {ENABLE_DIARIZATION} | Visual: {not args.skip_visual}")

    wav_path = None
    try:
        # Stage 1
        ingest = ingest_video(video_path, logger)

        # Stage 2
        metadata = extract_metadata(ingest, logger)

        # Stage 3
        wav_path = extract_audio(ingest, logger)

        # Stage 4
        asr_segments = transcribe_audio(wav_path, logger)

        # Stage 5
        diarization_turns = diarize_speakers(wav_path, logger)

        # Stage 6
        if args.skip_visual:
            logger.warning("[Stage 6] Skipped (--skip-visual).")
            visual = {
                "layout": metadata["composite_layout"],
                "participants": [],
                "nonverbal_events": [],
                "emotion_events": [],
                "warnings": ["Visual analysis skipped."],
            }
        else:
            visual = analyze_visual(ingest, metadata, logger)
            metadata["composite_layout"] = visual["layout"]
            metadata["participants"]     = visual["participants"]
            metadata["warnings"].extend(visual.get("warnings", []))

        # Stage 7
        screenshare_segments = detect_screenshare(ingest, metadata, logger)
        metadata["screenshare_visible"] = len(screenshare_segments) > 0
        if screenshare_segments:
            platforms = [s["platform_name"] for s in screenshare_segments if s["platform_name"]]
            metadata["screenshare_platform"] = platforms[0] if platforms else "unknown"

        # Stage 8
        transcript = merge_signals(
            ingest, asr_segments, diarization_turns, visual,
            screenshare_segments, logger, wav_path=wav_path
        )

        # Stage 9
        meta_path, trans_path = export_artifacts(metadata, transcript, video_id, logger)

        logger.info(f"=== Pipeline complete: {video_id} ===")
        logger.info(f"    Metadata:   {meta_path}")
        logger.info(f"    Transcript: {trans_path}")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

    finally:
        if not SAVE_INTERMEDIATE and wav_path is not None and wav_path.exists():
            wav_path.unlink()
            logger.info(f"[Cleanup] Removed intermediate WAV: {wav_path}")


if __name__ == "__main__":
    main()
