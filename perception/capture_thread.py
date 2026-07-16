# =============================================================================
# perception/capture_thread.py
# AirSign — Camera Capture Thread
# -----------------------------------------------------------------------------
# Dedicates one thread to polling the webcam via cv2.VideoCapture. This
# thread's sole responsibility is reading frames as fast as the camera
# produces them and pushing them onto frame_queue — it performs no
# MediaPipe inference, no feature extraction, and no gesture logic.
#
# Design contract:
#   - Never blocks on a full queue. If frame_queue is full, the OLDEST
#     buffered frame is discarded to make room for the newest one — this
#     thread always prioritises recency over completeness.
#   - daemon=True so the thread does not prevent process exit.
#   - stop_event is the sole cross-thread shutdown signal; this thread
#     checks it every loop iteration and exits promptly when set.
#   - current_fps is exposed for the status bar / RESULT_SCHEMA's
#     "inference_fps" field is populated by InferenceThread separately;
#     this thread's own FPS is a distinct, capture-side measurement.
# =============================================================================

from __future__ import annotations

import threading
import time
from collections import deque
from queue import Full, Queue
from typing import Optional

import cv2
import numpy as np

from shared.constants import (
    CAPTURE_QUEUE_MAXSIZE,
    DEFAULT_CAMERA_INDEX,
    FPS_ROLLING_WINDOW,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    TARGET_FPS,
)


class CaptureThread(threading.Thread):
    """
    Dedicated webcam polling thread.

    Reads frames from a ``cv2.VideoCapture`` device as fast as the camera
    delivers them and pushes each frame onto ``frame_queue``. Runs
    independently of ``InferenceThread`` so a slow inference pass never
    stalls frame acquisition — frames are simply dropped rather than
    allowed to queue up.

    Attributes:
        current_fps (float): Rolling average frames-per-second of the
            capture loop itself (not inference throughput).
    """

    def __init__(
        self,
        frame_queue: Queue,
        stop_event: threading.Event,
        device_index: int = DEFAULT_CAMERA_INDEX,
    ) -> None:
        """
        Initialise the capture thread. Does not open the camera yet — that
        happens in ``run()``, once the thread actually starts, so
        construction is cheap and side-effect-free.

        Args:
            frame_queue:  Shared queue this thread pushes raw BGR frames
                          onto. Should be constructed with
                          ``maxsize=CAPTURE_QUEUE_MAXSIZE``.
            stop_event:   Shared shutdown signal. When set, the capture
                          loop exits at the start of its next iteration.
            device_index: OpenCV camera device index. Defaults to
                          ``DEFAULT_CAMERA_INDEX`` (0 — first camera).
        """
        super().__init__(daemon=True, name="CaptureThread")

        self._frame_queue: Queue = frame_queue
        self._stop_event: threading.Event = stop_event
        self._device_index: int = device_index

        self.cap: Optional[cv2.VideoCapture] = None
        self.current_fps: float = 0.0

        # Rolling window of per-frame timestamps for FPS computation.
        self._frame_timestamps: deque = deque(maxlen=FPS_ROLLING_WINDOW)

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Main capture loop. Opens the camera, then repeatedly reads frames
        and pushes them onto ``frame_queue`` until ``stop_event`` is set.

        On any frame read failure, logs a warning and retries on the next
        iteration rather than terminating the thread — a single dropped
        frame from a flaky USB camera should not end the session.

        The camera is guaranteed to be released via ``release()`` in a
        ``finally`` block, even if the loop exits due to an unexpected
        exception.
        """
        try:
            self.cap = cv2.VideoCapture(self._device_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

            if not self.cap.isOpened():
                print(
                    f"CaptureThread: could not open camera at index "
                    f"{self._device_index}. Thread exiting."
                )
                return

            while not self._stop_event.is_set():
                ok, frame = self.cap.read()

                if not ok or frame is None:
                    # Transient read failure — retry next iteration rather
                    # than terminating the thread.
                    continue

                self._push_frame(frame)
                self._record_frame_timestamp()

        finally:
            self.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _push_frame(self, frame: np.ndarray) -> None:
        """
        Push a frame onto ``frame_queue``, dropping the oldest buffered
        frame if the queue is already full.

        This is the core "prioritise recency" policy: ``InferenceThread``
        should always see the most recent frame available, never a stale
        backlog. Frame drops here are silent and expected under normal
        operation whenever inference is the throughput bottleneck.

        Args:
            frame: Raw BGR frame as returned by ``cv2.VideoCapture.read()``.
        """
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except Exception:
                # Queue emptied concurrently between full() and get_nowait() —
                # harmless race, proceed to put the new frame regardless.
                pass

        try:
            self._frame_queue.put_nowait(frame)
        except Full:
            # Extremely unlikely given the drop above, but never block.
            pass

    def _record_frame_timestamp(self) -> None:
        """
        Record the current monotonic time and recompute ``current_fps``
        as a rolling average over the last ``FPS_ROLLING_WINDOW`` frames.
        """
        now = time.monotonic()
        self._frame_timestamps.append(now)

        if len(self._frame_timestamps) >= 2:
            elapsed = self._frame_timestamps[-1] - self._frame_timestamps[0]
            frame_count = len(self._frame_timestamps) - 1
            if elapsed > 0:
                self.current_fps = frame_count / elapsed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """
        Signal the capture loop to exit.

        This does not directly stop the OS-level camera read call in
        progress (blocking I/O cannot be interrupted mid-call); the loop
        checks ``stop_event`` once per iteration, so the thread exits
        promptly after the current frame read completes. Callers should
        set the shared ``stop_event`` (typically via this method) and then
        ``join()`` the thread to wait for full shutdown.
        """
        self._stop_event.set()

    def release(self) -> None:
        """
        Release the underlying camera device, if open.

        Safe to call multiple times or when the camera was never
        successfully opened. Always called from ``run()``'s ``finally``
        block; exposed publicly as well so callers can force an early
        release if needed.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None