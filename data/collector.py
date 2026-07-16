# =============================================================================
# data/collector.py
# AirSign — Gesture Landmark Data Collection Tool
# -----------------------------------------------------------------------------
# CLI script to record labeled gesture samples as landmark coordinate CSV rows.
# Operator runs this script, performs a gesture repeatedly in front of the
# webcam, and presses SPACE to capture a sample. No model training occurs
# here — this module produces raw training data only.
#
# Output format:
#   data/landmarks/<GESTURE_LABEL>.csv
#   One row per sample. No header row. Columns are the 22 FEATURE_KEYS
#   values in their canonical order (imported directly from
#   feature_extractor.FEATURE_KEYS — never hardcoded), followed by the
#   gesture label string as the 23rd column.
#
# Design contract:
#   - Feature column order is derived exclusively from FEATURE_KEYS.
#     If FEATURE_KEYS changes, this script automatically adapts — no
#     manual column re-mapping required.
#   - Only CLASSIFIABLE_GESTURES may be recorded (excludes NONE/UNKNOWN,
#     which are inference-time states, not collectible poses).
#   - CSV append is atomic per row: each captured sample is flushed to disk
#     immediately so a crash mid-session does not lose prior samples.
#
# Usage::
#
#   python data/collector.py OPEN_PALM 200
#   python data/collector.py PINCH 200
#   python data/collector.py FIST 200
#   python data/collector.py POINT 200
#   python data/collector.py TWO_FINGER 200
#   python data/collector.py THUMBS_UP 200
#
# Controls while running:
#   SPACE — capture one sample of the current hand pose
#   q     — quit early (partial sessions are still saved)
# =============================================================================

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp

# Ensure project root is importable when invoked as `python data/collector.py`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.gesture.feature_extractor import FEATURE_KEYS, extract_features
from shared.constants import FRAME_HEIGHT, FRAME_WIDTH
from shared.gesture_labels import CLASSIFIABLE_GESTURES

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR: str = os.path.join(
    os.path.dirname(__file__), "landmarks"
)

# NOTE: MediaPipe's `solutions` submodule (mp.solutions.hands,
# mp.solutions.drawing_utils) is accessed lazily inside collect_session()
# rather than bound at import time. This mirrors the lazy-init convention
# used by InferenceThread ("MediaPipe must init in the thread that uses
# it") and additionally protects this module from import-time failures on
# MediaPipe distributions/versions where the legacy `solutions` API is not
# attached to the top-level package until explicitly touched.

# Key codes
_KEY_SPACE: int = 32
_KEY_QUIT: int = ord("q")


# =============================================================================
# CSV persistence
# =============================================================================

def _csv_path(gesture_label: str, data_dir: str) -> str:
    """
    Return the absolute CSV path for a given gesture label.

    Args:
        gesture_label: One of CLASSIFIABLE_GESTURES.
        data_dir:      Directory in which per-gesture CSV files are stored.

    Returns:
        Absolute path string to ``<data_dir>/<gesture_label>.csv``.
    """
    return os.path.join(data_dir, f"{gesture_label}.csv")


def append_sample(gesture_label: str, features: dict, data_dir: str = DEFAULT_DATA_DIR) -> None:
    """
    Append one labeled feature sample to the gesture's CSV file.

    Column order is derived from ``FEATURE_KEYS`` — the single source of
    truth for feature layout shared with ``classifier.py`` and ``trainer.py``.
    The label is written as the final column. No header row is written;
    column order is implicit and must match ``FEATURE_KEYS`` on every read.

    Args:
        gesture_label: One of CLASSIFIABLE_GESTURES. Not re-validated here —
                       callers (``collect_session``) are responsible for
                       validating before invoking this function repeatedly.
        features:      Output of ``extract_features()``. Must contain every
                       key in ``FEATURE_KEYS``.
        data_dir:      Directory in which to write ``<gesture_label>.csv``.
                       Created if it does not already exist.

    Raises:
        KeyError: If ``features`` is missing any key required by
                  ``FEATURE_KEYS``.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = _csv_path(gesture_label, data_dir)

    row = [features[key] for key in FEATURE_KEYS]
    row.append(gesture_label)

    # Append mode with immediate flush — each sample survives a crash.
    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def count_existing_samples(gesture_label: str, data_dir: str = DEFAULT_DATA_DIR) -> int:
    """
    Return the number of samples already recorded for a gesture label.

    Used to resume a partially-completed collection session and to display
    an accurate running count that accounts for prior runs.

    Args:
        gesture_label: One of CLASSIFIABLE_GESTURES.
        data_dir:      Directory containing the per-gesture CSV files.

    Returns:
        Number of rows in the existing CSV file, or 0 if the file does not
        yet exist.
    """
    path = _csv_path(gesture_label, data_dir)
    if not os.path.exists(path):
        return 0
    with open(path, mode="r", newline="") as f:
        return sum(1 for _ in csv.reader(f))


# =============================================================================
# Interactive collection session
# =============================================================================

def collect_session(
    gesture_label: str,
    num_samples: int = 200,
    data_dir: str = DEFAULT_DATA_DIR,
    camera_index: int = 0,
) -> int:
    """
    Run an interactive webcam session to collect labeled gesture samples.

    Opens the webcam, runs MediaPipe Hands on each frame, and displays a
    live annotated feed with the current sample count overlaid. The operator
    performs the target gesture and presses SPACE to capture a sample at
    that instant. Samples are appended to ``<data_dir>/<gesture_label>.csv``
    immediately (one fsync'd write per capture).

    The session ends when either ``num_samples`` new samples have been
    captured in this run, or the operator presses 'q' to quit early — in
    which case any samples already captured are preserved.

    Args:
        gesture_label: Must be one of ``CLASSIFIABLE_GESTURES``
                       (``OPEN_PALM``, ``FIST``, ``PINCH``, ``POINT``,
                       ``TWO_FINGER``, ``THUMBS_UP``). ``NONE`` and
                       ``UNKNOWN`` are inference-time states and cannot
                       be collected as training poses.
        num_samples:   Number of NEW samples to capture in this session.
                       Default 200. Existing samples from prior runs are
                       not counted toward this total but are reported.
        data_dir:      Directory to write the CSV file into. Defaults to
                       ``data/landmarks`` relative to this module.
        camera_index:  OpenCV camera device index. Default 0 (first camera).

    Returns:
        Total number of samples now present for this gesture label
        (existing + newly captured).

    Raises:
        ValueError: If ``gesture_label`` is not a member of
                    ``CLASSIFIABLE_GESTURES``.
        RuntimeError: If the webcam at ``camera_index`` cannot be opened.
    """
    if gesture_label not in CLASSIFIABLE_GESTURES:
        raise ValueError(
            f"gesture_label must be one of {CLASSIFIABLE_GESTURES}, "
            f"got {gesture_label!r}. NONE and UNKNOWN are inference-time "
            f"states and cannot be collected as training data."
        )
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples!r}.")

    # Lazy MediaPipe solutions access — see module-level NOTE above.
    mp_hands_module = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    existing_count = count_existing_samples(gesture_label, data_dir)
    captured_this_session = 0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {camera_index}. "
            f"Check that the webcam is connected and not in use by "
            f"another application."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    window_name = f"AirSign Data Collector — {gesture_label}"

    try:
        with mp_hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0,
        ) as hands:
            print(
                f"Collecting '{gesture_label}': "
                f"{existing_count} existing samples, "
                f"target {num_samples} new samples this session."
            )
            print("Press SPACE to capture a sample. Press 'q' to quit early.")

            latest_features: Optional[dict] = None

            while captured_this_session < num_samples:
                ok, frame = cap.read()
                if not ok:
                    print("Warning: failed to read frame from camera, retrying.")
                    continue

                frame = cv2.flip(frame, 1)  # Mirror for natural interaction
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                latest_features = None
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = [
                        (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
                    ]
                    latest_features = extract_features(landmarks)

                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands_module.HAND_CONNECTIONS
                    )

                overlay_text = (
                    f"[{captured_this_session}/{num_samples}] "
                    f"{gesture_label} samples collected"
                )
                cv2.putText(
                    frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2,
                )
                hand_status = "Hand detected" if latest_features else "No hand"
                cv2.putText(
                    frame, hand_status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 0) if latest_features else (0, 0, 200), 2,
                )

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == _KEY_QUIT:
                    print("Quitting early. Partial session samples are saved.")
                    break

                if key == _KEY_SPACE:
                    if latest_features is None:
                        print("No hand detected — sample not captured.")
                        continue
                    append_sample(gesture_label, latest_features, data_dir)
                    captured_this_session += 1
                    print(
                        f"[{captured_this_session}/{num_samples}] "
                        f"{gesture_label} samples collected"
                    )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    total = existing_count + captured_this_session
    print(
        f"Session complete. Captured {captured_this_session} new samples. "
        f"Total for '{gesture_label}': {total}."
    )
    return total


# =============================================================================
# CLI entry point
# =============================================================================

def _print_usage() -> None:
    print(
        "Usage: python data/collector.py <GESTURE_LABEL> [NUM_SAMPLES]\n"
        f"  GESTURE_LABEL must be one of: {CLASSIFIABLE_GESTURES}\n"
        "  NUM_SAMPLES defaults to 200.\n\n"
        "Example:\n"
        "  python data/collector.py PINCH 200"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    label_arg = sys.argv[1]
    count_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    if label_arg not in CLASSIFIABLE_GESTURES:
        print(f"Error: '{label_arg}' is not a valid gesture label.\n")
        _print_usage()
        sys.exit(1)

    collect_session(label_arg, count_arg)