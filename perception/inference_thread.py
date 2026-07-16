# =============================================================================
# perception/inference_thread.py
# AirSign — Inference Thread
# -----------------------------------------------------------------------------
# The core perception pipeline. Consumes raw frames from frame_queue, runs
# MediaPipe Hands, then chains the full Dev A pipeline:
#
#   raw landmarks → LandmarkSmoother → feature_extractor → GestureClassifier
#   → GestureFSM → RESULT_SCHEMA dict → result_queue
#
# Design contract:
#   - MediaPipe is lazily initialised inside run() — never at module or
#     __init__ scope. MediaPipe's Hands solution must be constructed in the
#     thread that uses it; constructing it elsewhere (or at import time) is
#     both a threading hazard and, on some MediaPipe distributions, an
#     import-order hazard since not every build attaches `solutions` to the
#     top-level package until first touched.
#   - Never imports PyQt6, pynput, or any Dev B module — this file's only
#     job is to produce the RESULT_SCHEMA dict; OS injection and UI
#     rendering are strictly out of scope here.
#   - Never blocks Dev B: if result_queue is full, the NEW result is
#     dropped (not the queued one) — ActionThread may be mid-read on the
#     queued result and must never see it mutated or removed out from
#     under it.
#   - screeninfo is an optional dependency from Dev B's requirements. It is
#     imported defensively inside a try/except with a hardcoded fallback
#     resolution, so this module remains importable and functional even in
#     a Dev A-only environment where screeninfo is not installed.
# =============================================================================

from __future__ import annotations

import threading
import time
from collections import deque
from queue import Empty, Full, Queue
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from perception.filtering.smoother import LandmarkSmoother
from perception.gesture.classifier import GestureClassifier
from perception.gesture.feature_extractor import extract_features
from perception.gesture.fsm import GestureFSM
from shared.constants import (
    DEFAULT_ACTIVE_ZONE,
    DEFAULT_FILTER_BETA,
    DEFAULT_FILTER_DCUTOFF,
    DEFAULT_FILTER_MINCUTOFF,
    DEFAULT_PINCH_TOLERANCE,
    DEFAULT_SENSITIVITY,
    FPS_ROLLING_WINDOW,
    QUEUE_GET_TIMEOUT,
    RESULT_QUEUE_MAXSIZE,
    TARGET_FPS,
)

# ---------------------------------------------------------------------------
# Fallback screen resolution used when screeninfo is unavailable or fails
# to enumerate a monitor. 1920x1080 is a safe, common default.
# ---------------------------------------------------------------------------

_FALLBACK_SCREEN_WIDTH: int = 1920
_FALLBACK_SCREEN_HEIGHT: int = 1080

# MediaPipe Hands tuning — fixed, not user-configurable via config.json.
_MP_MIN_DETECTION_CONFIDENCE: float = 0.7
_MP_MIN_TRACKING_CONFIDENCE: float = 0.5
_MP_MAX_NUM_HANDS: int = 1
_MP_MODEL_COMPLEXITY: int = 0  # Lightweight model — mandatory for real-time CPU/GPU balance.

FeatureDict = Dict[str, float]
ResultDict = Dict[str, Any]


def _resolve_screen_size() -> Tuple[int, int]:
    """
    Determine the primary monitor's resolution for cursor coordinate mapping.

    Attempts to use ``screeninfo`` (a Dev B dependency) if it is installed
    and can successfully enumerate at least one monitor. Falls back to a
    hardcoded 1920x1080 resolution otherwise — this keeps
    ``InferenceThread`` fully functional in a Dev A-only development
    environment where ``screeninfo`` may not be installed.

    Returns:
        Tuple of ``(width, height)`` in pixels.
    """
    try:
        import screeninfo  # Optional dependency — imported lazily and defensively.

        monitor = screeninfo.get_monitors()[0]
        return int(monitor.width), int(monitor.height)
    except Exception:
        return _FALLBACK_SCREEN_WIDTH, _FALLBACK_SCREEN_HEIGHT


class InferenceThread(threading.Thread):
    """
    Core perception processing thread.

    Consumes raw BGR frames from ``frame_queue``, runs MediaPipe Hands,
    smooths landmarks, extracts features, classifies the gesture, updates
    the temporal FSM, and pushes one complete ``RESULT_SCHEMA`` dict per
    processed frame onto ``result_queue`` for ``ActionThread`` (Dev B) to
    consume.

    Attributes:
        frame_id (int): Monotonically increasing counter of frames processed
            since thread construction. Never resets during a session.
    """

    def __init__(
        self,
        frame_queue: Queue,
        result_queue: Queue,
        stop_event: threading.Event,
        config: Optional[dict] = None,
    ) -> None:
        """
        Initialise the inference thread. Does not touch MediaPipe here —
        that happens lazily in ``run()``.

        Args:
            frame_queue:  Shared queue this thread consumes raw BGR frames
                          from, as produced by ``CaptureThread``.
            result_queue: Shared queue this thread pushes completed
                          ``RESULT_SCHEMA`` dicts onto, for ``ActionThread``
                          to consume. Should be constructed with
                          ``maxsize=RESULT_QUEUE_MAXSIZE``.
            stop_event:   Shared shutdown signal. When set, the processing
                          loop exits at the start of its next iteration.
            config:       Optional runtime configuration dict, matching the
                          ``config.json`` schema. Relevant sections read:
                          ``filter`` (mincutoff, beta, dcutoff),
                          ``tracking`` (sensitivity, pinch_tolerance,
                          active_zone). Missing keys fall back to the
                          corresponding ``DEFAULT_*`` constants. If omitted
                          entirely, all defaults are used.
        """
        super().__init__(daemon=True, name="InferenceThread")

        self._frame_queue: Queue = frame_queue
        self._result_queue: Queue = result_queue
        self._stop_event: threading.Event = stop_event
        self._config: dict = config or {}

        # ── MediaPipe handles — lazily created inside run() ────────────────
        self._mp_hands: Optional[Any] = None
        self._mp_hands_module: Optional[Any] = None
        self._mp_drawing: Optional[Any] = None

        # ── Config-derived tuning values, resolved once at construction ────
        filter_cfg = self._config.get("filter", {})
        tracking_cfg = self._config.get("tracking", {})

        self._mincutoff: float = filter_cfg.get("mincutoff", DEFAULT_FILTER_MINCUTOFF)
        self._beta: float = filter_cfg.get("beta", DEFAULT_FILTER_BETA)
        self._dcutoff: float = filter_cfg.get("dcutoff", DEFAULT_FILTER_DCUTOFF)

        self._sensitivity: float = tracking_cfg.get("sensitivity", DEFAULT_SENSITIVITY)
        self._pinch_tolerance: float = tracking_cfg.get(
            "pinch_tolerance", DEFAULT_PINCH_TOLERANCE
        )
        self._active_zone: dict = tracking_cfg.get("active_zone", DEFAULT_ACTIVE_ZONE)

        # ── Perception pipeline components ─────────────────────────────────
        # These are safe to construct at __init__ time — none of them touch
        # MediaPipe, a camera, or any OS resource.
        self._smoother: LandmarkSmoother = LandmarkSmoother(
            freq=TARGET_FPS,
            mincutoff=self._mincutoff,
            beta=self._beta,
            dcutoff=self._dcutoff,
        )
        self._classifier: GestureClassifier = GestureClassifier(use_mlp=True)
        self._fsm: GestureFSM = GestureFSM()

        # ── Per-session counters and rolling FPS tracking ───────────────────
        self.frame_id: int = 0
        self.current_fps: float = 0.0
        self._frame_timestamps: deque = deque(maxlen=FPS_ROLLING_WINDOW)

        # ── Screen resolution for cursor mapping ────────────────────────────
        self._screen_w, self._screen_h = _resolve_screen_size()

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Main processing loop. Lazily initialises MediaPipe Hands, then
        repeatedly consumes frames from ``frame_queue``, runs the full
        perception pipeline, and pushes results onto ``result_queue`` until
        ``stop_event`` is set.

        MediaPipe's ``Hands`` context manager is entered exactly once for
        the lifetime of this thread and exited cleanly on shutdown via the
        ``with`` block, ensuring its internal resources are released.
        """
        # Import and construct MediaPipe's Hands solution here, inside the
        # thread that will actually use it — never at module or __init__ scope.
        import mediapipe as mp

        self._mp_hands_module = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils

        with self._mp_hands_module.Hands(
            static_image_mode=False,
            max_num_hands=_MP_MAX_NUM_HANDS,
            min_detection_confidence=_MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=_MP_MIN_TRACKING_CONFIDENCE,
            model_complexity=_MP_MODEL_COMPLEXITY,
        ) as hands:
            self._mp_hands = hands

            while not self._stop_event.is_set():
                try:
                    frame = self._frame_queue.get(timeout=QUEUE_GET_TIMEOUT)
                except Empty:
                    continue

                result = self._process_frame(frame)
                self._push_result(result)
                self.frame_id += 1

        self._mp_hands = None

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> ResultDict:
        """
        Run one frame through the complete perception pipeline.

        Args:
            frame: Raw BGR frame from ``frame_queue``, as produced by
                   ``CaptureThread``.

        Returns:
            A fully-populated ``RESULT_SCHEMA`` dict for this frame.
        """
        timestamp = time.monotonic()
        self._record_frame_timestamp(timestamp)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = self._mp_hands.process(frame_rgb)

        landmarks_raw: Optional[list] = None
        landmarks_filtered: Optional[list] = None
        cursor_screen_pos: Optional[Tuple[int, int]] = None
        static_gesture: Optional[str] = None
        gesture_confidence: float = 0.0
        features: FeatureDict = {}
        annotated_frame = frame

        hand_landmarks_obj = None
        if mp_results.multi_hand_landmarks:
            hand_landmarks_obj = mp_results.multi_hand_landmarks[0]

        if hand_landmarks_obj is not None:
            landmarks_raw = [
                (lm.x, lm.y, lm.z) for lm in hand_landmarks_obj.landmark
            ]

            landmarks_filtered = self._smoother.smooth(
                landmarks_raw, timestamp=timestamp
            )

            features = extract_features(landmarks_filtered)

            static_gesture, gesture_confidence = self._classifier.classify(features)

            cursor_screen_pos = self._smoother.get_cursor_position(
                landmarks_filtered,
                sensitivity=self._sensitivity,
                screen_w=self._screen_w,
                screen_h=self._screen_h,
                active_zone=self._active_zone,
            )

            annotated_frame = frame.copy()
            self._mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks_obj,
                self._mp_hands_module.HAND_CONNECTIONS,
            )

            fsm_gesture = static_gesture if static_gesture is not None else "UNKNOWN"
        else:
            # No hand detected this frame — feed "NONE" to the FSM so its
            # HAND_LOST counter advances, and reset smoother state so a
            # freshly-reappearing hand does not inherit stale filter memory.
            self._smoother.reset()
            fsm_gesture = "NONE"

        fsm_event = self._fsm.update(
            fsm_gesture, gesture_confidence, features, timestamp
        )
        fsm_state = self._fsm.get_state()

        return self._build_result(
            timestamp=timestamp,
            annotated_frame=annotated_frame,
            landmarks_raw=landmarks_raw,
            landmarks_filtered=landmarks_filtered,
            cursor_screen_pos=cursor_screen_pos,
            static_gesture=static_gesture,
            gesture_confidence=gesture_confidence,
            fsm_state=fsm_state,
            fsm_event=fsm_event,
            hand_detected=hand_landmarks_obj is not None,
        )

    def _build_result(
        self,
        timestamp: float,
        annotated_frame: np.ndarray,
        landmarks_raw: Optional[list],
        landmarks_filtered: Optional[list],
        cursor_screen_pos: Optional[Tuple[int, int]],
        static_gesture: Optional[str],
        gesture_confidence: float,
        fsm_state: str,
        fsm_event: Optional[str],
        hand_detected: bool,
    ) -> ResultDict:
        """
        Assemble the complete ``RESULT_SCHEMA`` dict for one processed frame.

        Args:
            timestamp:           Monotonic timestamp when inference completed
                                 for this frame.
            annotated_frame:     BGR frame with landmark skeleton drawn on it
                                 (or the original frame, unmodified, when no
                                 hand was detected).
            landmarks_raw:       21-item list of raw (pre-filter) landmark
                                 tuples, or ``None`` if no hand detected.
            landmarks_filtered:  21-item list of smoothed landmark tuples, or
                                 ``None`` if no hand detected.
            cursor_screen_pos:   Absolute screen pixel coordinates for the
                                 cursor, or ``None`` if no hand detected.
            static_gesture:      Per-frame classified gesture label, or
                                 ``None`` if no hand detected.
            gesture_confidence:  Classifier confidence for ``static_gesture``.
            fsm_state:           Current FSM state after this frame's update.
            fsm_event:           FSM event fired this frame, or ``None``.
            hand_detected:       Whether MediaPipe detected a hand this frame.

        Returns:
            Dict matching the ``RESULT_SCHEMA`` contract exactly — every key
            required by Dev B's ``ActionThread`` is present.
        """
        graph_data = {
            "raw_x": None,
            "raw_y": None,
            "filtered_x": None,
            "filtered_y": None,
        }
        if landmarks_raw is not None and landmarks_filtered is not None:
            from shared.constants import CURSOR_LANDMARK_INDEX

            graph_data["raw_x"] = landmarks_raw[CURSOR_LANDMARK_INDEX][0]
            graph_data["raw_y"] = landmarks_raw[CURSOR_LANDMARK_INDEX][1]
            graph_data["filtered_x"] = landmarks_filtered[CURSOR_LANDMARK_INDEX][0]
            graph_data["filtered_y"] = landmarks_filtered[CURSOR_LANDMARK_INDEX][1]

        return {
            "timestamp": timestamp,
            "frame_id": self.frame_id,
            "annotated_frame": annotated_frame,
            "landmarks_raw": landmarks_raw,
            "landmarks_filtered": landmarks_filtered,
            "cursor_screen_pos": cursor_screen_pos,
            "static_gesture": static_gesture,
            "gesture_confidence": gesture_confidence,
            "fsm_state": fsm_state,
            "fsm_event": fsm_event,
            "graph_data": graph_data,
            "inference_fps": self.current_fps,
            "hand_detected": hand_detected,
        }

    def _push_result(self, result: ResultDict) -> None:
        """
        Push a completed result dict onto ``result_queue``.

        If the queue is already full, the NEW result is dropped — the
        queued result is left untouched, since ``ActionThread`` may be
        mid-read on it. This is the mirror-image policy of
        ``CaptureThread``, which drops the OLD frame instead; the two
        threads intentionally have opposite drop policies because they sit
        on opposite ends of their respective queues relative to the
        consumer that must never see a half-consumed item vanish.

        Args:
            result: Completed ``RESULT_SCHEMA`` dict for the current frame.
        """
        try:
            self._result_queue.put_nowait(result)
        except Full:
            pass

    def _record_frame_timestamp(self, timestamp: float) -> None:
        """
        Record a frame timestamp and recompute the rolling inference FPS.

        Args:
            timestamp: Monotonic timestamp for the current frame.
        """
        self._frame_timestamps.append(timestamp)

        if len(self._frame_timestamps) >= 2:
            elapsed = self._frame_timestamps[-1] - self._frame_timestamps[0]
            frame_count = len(self._frame_timestamps) - 1
            if elapsed > 0:
                self.current_fps = frame_count / elapsed

    # ------------------------------------------------------------------
    # Runtime parameter updates (called from settings panel via Dev B)
    # ------------------------------------------------------------------

    def update_filter_params(
        self, mincutoff: Optional[float] = None, beta: Optional[float] = None
    ) -> None:
        """
        Hot-swap the 1€ filter's mincutoff and/or beta parameters at runtime.

        Forwards directly to ``LandmarkSmoother.update_params()``, which
        propagates the change to every active filter in its pool without
        discontinuity in the smoothed output. Called when the user adjusts
        the corresponding sliders in Dev B's settings panel.

        Args:
            mincutoff: New minimum cutoff frequency in Hz, or ``None`` to
                       leave unchanged.
            beta:      New speed coefficient, or ``None`` to leave unchanged.
        """
        self._smoother.update_params(mincutoff=mincutoff, beta=beta)
        if mincutoff is not None:
            self._mincutoff = mincutoff
        if beta is not None:
            self._beta = beta

    def get_classifier_backend(self) -> str:
        """
        Return which gesture classifier backend is currently active.

        Returns:
            ``"mlp"`` if the trained MLP model is loaded and active,
            ``"rule_based"`` if operating on the deterministic fallback.
            Mirrors ``GestureClassifier.backend_type`` for display in Dev
            B's status bar.
        """
        return self._classifier.backend_type

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"InferenceThread("
            f"frame_id={self.frame_id}, "
            f"fsm_state={self._fsm.get_state()!r}, "
            f"classifier_backend={self._classifier.backend_type!r}, "
            f"fps={self.current_fps:.1f}"
            f")"
        )