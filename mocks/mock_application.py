# =============================================================================
# mocks/mock_application.py
# AirSign — Standalone Perception Pipeline Validator (Phase 7)
# -----------------------------------------------------------------------------
# NOT a production file. This is Dev A's standalone validation utility for
# running the complete perception pipeline (CaptureThread + InferenceThread)
# end-to-end against a real webcam, with no Dev B code involved at all.
#
# It plays the role Dev B's ActionThread will eventually play — consuming
# result_queue — but does nothing with the results except validate their
# schema and print/display them. This is how Dev A confirms the entire
# perception stack (capture -> MediaPipe -> smoothing -> features ->
# classification -> FSM -> RESULT_SCHEMA) works correctly BEFORE handing
# off to Dev B for integration.
#
# Usage::
#
#   python mocks/mock_application.py
#   python mocks/mock_application.py --camera 1
#
# Controls while running:
#   q — quit
#
# Exit behavior:
#   Pressing 'q' sets the shared stop_event, both threads are joined with a
#   timeout, and the OpenCV preview window is destroyed before the process
#   exits. Ctrl+C is also handled for a clean shutdown.
# =============================================================================

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from typing import Any, Dict

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.capture_thread import CaptureThread
from perception.inference_thread import InferenceThread
from shared.constants import CAPTURE_QUEUE_MAXSIZE, RESULT_QUEUE_MAXSIZE
from shared.fsm_states import FSM_STATE_SET
from shared.gesture_labels import GESTURE_LABEL_SET

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_WINDOW_NAME: str = "AirSign — mock_application.py (Dev A pipeline validator)"
_SUMMARY_EVERY_N_FRAMES: int = 30
_RESULT_GET_TIMEOUT: float = 0.1
_JOIN_TIMEOUT: float = 3.0

# Complete RESULT_SCHEMA key set, per perception/inference_thread.py's
# _build_result(). Used for the schema validator below.
_REQUIRED_RESULT_KEYS: tuple = (
    "timestamp",
    "frame_id",
    "annotated_frame",
    "landmarks_raw",
    "landmarks_filtered",
    "cursor_screen_pos",
    "static_gesture",
    "gesture_confidence",
    "fsm_state",
    "fsm_event",
    "graph_data",
    "inference_fps",
    "hand_detected",
)

_REQUIRED_GRAPH_DATA_KEYS: tuple = ("raw_x", "raw_y", "filtered_x", "filtered_y")


# =============================================================================
# Schema validation
# =============================================================================

def validate_result_schema(result: Dict[str, Any], frame_count: int) -> None:
    """
    Validate that a RESULT_SCHEMA dict conforms to the contract documented
    in perception/inference_thread.py's _build_result().

    Raises loudly (AssertionError) on any violation — this script is a
    validator, not a production consumer, so failing fast and visibly is
    the correct behavior rather than silently tolerating malformed output.

    Args:
        result:      The dict popped from result_queue.
        frame_count: Current frame counter, used only for error messages.

    Raises:
        AssertionError: On any schema violation, with a message identifying
                        exactly which field or constraint failed and at
                        which frame.
    """
    missing_keys = [k for k in _REQUIRED_RESULT_KEYS if k not in result]
    assert not missing_keys, (
        f"[frame {frame_count}] SCHEMA VIOLATION: missing keys {missing_keys}"
    )

    assert isinstance(result["timestamp"], float), (
        f"[frame {frame_count}] SCHEMA VIOLATION: timestamp is not a float"
    )
    assert isinstance(result["frame_id"], int), (
        f"[frame {frame_count}] SCHEMA VIOLATION: frame_id is not an int"
    )

    frame = result["annotated_frame"]
    assert frame is not None and hasattr(frame, "shape"), (
        f"[frame {frame_count}] SCHEMA VIOLATION: annotated_frame is not "
        f"an array-like object"
    )
    assert len(frame.shape) == 3 and frame.shape[2] == 3, (
        f"[frame {frame_count}] SCHEMA VIOLATION: annotated_frame shape "
        f"{frame.shape} is not (H, W, 3)"
    )

    hand_detected = result["hand_detected"]
    assert isinstance(hand_detected, bool), (
        f"[frame {frame_count}] SCHEMA VIOLATION: hand_detected is not a bool"
    )

    if not hand_detected:
        assert result["landmarks_raw"] is None, (
            f"[frame {frame_count}] SCHEMA VIOLATION: landmarks_raw should "
            f"be None when hand_detected is False"
        )
        assert result["landmarks_filtered"] is None, (
            f"[frame {frame_count}] SCHEMA VIOLATION: landmarks_filtered "
            f"should be None when hand_detected is False"
        )
        assert result["cursor_screen_pos"] is None, (
            f"[frame {frame_count}] SCHEMA VIOLATION: cursor_screen_pos "
            f"should be None when hand_detected is False"
        )
    else:
        assert (
            result["landmarks_raw"] is not None
            and len(result["landmarks_raw"]) == 21
        ), (
            f"[frame {frame_count}] SCHEMA VIOLATION: landmarks_raw must "
            f"have 21 entries when hand_detected is True"
        )
        assert (
            result["landmarks_filtered"] is not None
            and len(result["landmarks_filtered"]) == 21
        ), (
            f"[frame {frame_count}] SCHEMA VIOLATION: landmarks_filtered "
            f"must have 21 entries when hand_detected is True"
        )
        assert result["cursor_screen_pos"] is not None, (
            f"[frame {frame_count}] SCHEMA VIOLATION: cursor_screen_pos "
            f"must not be None when hand_detected is True"
        )
        cx, cy = result["cursor_screen_pos"]
        assert isinstance(cx, int) and isinstance(cy, int), (
            f"[frame {frame_count}] SCHEMA VIOLATION: cursor_screen_pos "
            f"components must be plain ints, got ({type(cx)}, {type(cy)})"
        )

    static_gesture = result["static_gesture"]
    assert static_gesture is None or static_gesture in GESTURE_LABEL_SET, (
        f"[frame {frame_count}] SCHEMA VIOLATION: static_gesture "
        f"{static_gesture!r} is not a member of GESTURE_LABELS"
    )

    confidence = result["gesture_confidence"]
    assert isinstance(confidence, float) and 0.0 <= confidence <= 1.0, (
        f"[frame {frame_count}] SCHEMA VIOLATION: gesture_confidence "
        f"{confidence!r} is not a float in [0.0, 1.0]"
    )

    fsm_state = result["fsm_state"]
    assert fsm_state in FSM_STATE_SET, (
        f"[frame {frame_count}] SCHEMA VIOLATION: fsm_state {fsm_state!r} "
        f"is not a member of FSM_STATES"
    )

    fsm_event = result["fsm_event"]
    from shared.fsm_states import FSM_EVENT_SET

    assert fsm_event is None or fsm_event in FSM_EVENT_SET, (
        f"[frame {frame_count}] SCHEMA VIOLATION: fsm_event {fsm_event!r} "
        f"is not a member of FSM_EVENTS"
    )

    graph_data = result["graph_data"]
    missing_graph_keys = [
        k for k in _REQUIRED_GRAPH_DATA_KEYS if k not in graph_data
    ]
    assert not missing_graph_keys, (
        f"[frame {frame_count}] SCHEMA VIOLATION: graph_data missing keys "
        f"{missing_graph_keys}"
    )
    if hand_detected:
        for key in _REQUIRED_GRAPH_DATA_KEYS:
            assert graph_data[key] is not None, (
                f"[frame {frame_count}] SCHEMA VIOLATION: graph_data[{key!r}] "
                f"is None while hand_detected is True"
            )

    assert isinstance(result["inference_fps"], float), (
        f"[frame {frame_count}] SCHEMA VIOLATION: inference_fps is not a float"
    )


# =============================================================================
# Summary printing
# =============================================================================

def print_summary(result: Dict[str, Any], frame_count: int) -> None:
    """
    Print a concise, human-readable summary of one RESULT_SCHEMA dict.

    Excludes the annotated_frame array (far too large to print usefully)
    and prints every other field so Dev A can eyeball correctness while
    the pipeline runs.

    Args:
        result:      Validated RESULT_SCHEMA dict.
        frame_count: Current frame counter, used in the summary header.
    """
    print(f"\n── Frame {frame_count} (frame_id={result['frame_id']}) ──")
    print(f"  hand_detected      : {result['hand_detected']}")
    print(f"  static_gesture     : {result['static_gesture']}")
    print(f"  gesture_confidence : {result['gesture_confidence']:.3f}")
    print(f"  fsm_state          : {result['fsm_state']}")
    print(f"  fsm_event          : {result['fsm_event']}")
    print(f"  cursor_screen_pos  : {result['cursor_screen_pos']}")
    print(f"  graph_data         : {result['graph_data']}")
    print(f"  inference_fps      : {result['inference_fps']:.1f}")


# =============================================================================
# Main entry point
# =============================================================================

def run(camera_index: int = 0) -> None:
    """
    Wire up CaptureThread + InferenceThread, consume result_queue, validate
    every result against RESULT_SCHEMA, print a summary every
    _SUMMARY_EVERY_N_FRAMES frames, and display the live annotated feed
    until the user presses 'q' or interrupts with Ctrl+C.

    Args:
        camera_index: OpenCV camera device index to open.
    """
    frame_queue: queue.Queue = queue.Queue(maxsize=CAPTURE_QUEUE_MAXSIZE)
    result_queue: queue.Queue = queue.Queue(maxsize=RESULT_QUEUE_MAXSIZE)
    stop_event = threading.Event()

    capture = CaptureThread(frame_queue, stop_event, device_index=camera_index)
    inference = InferenceThread(frame_queue, result_queue, stop_event, config={})

    print(f"Starting CaptureThread (camera index {camera_index})...")
    capture.start()
    print("Starting InferenceThread...")
    inference.start()
    print(f"Classifier backend: {inference.get_classifier_backend()}")
    print("Press 'q' in the preview window to quit.\n")

    frame_count = 0

    try:
        while True:
            try:
                result = result_queue.get(timeout=_RESULT_GET_TIMEOUT)
            except queue.Empty:
                # No new result yet — check whether the capture thread has
                # died (e.g. camera never opened) so we don't spin forever.
                if not capture.is_alive() and not inference.is_alive():
                    print(
                        "Both threads have exited. Check that a camera is "
                        "connected and accessible at the given index."
                    )
                    break
                # Allow 'q' to be checked even while waiting for frames.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quit requested.")
                    break
                continue

            frame_count += 1
            validate_result_schema(result, frame_count)

            if (frame_count % _SUMMARY_EVERY_N_FRAMES == 0 or result["fsm_event"] is not None):
                print_summary(result, frame_count)

            cv2.imshow(_WINDOW_NAME, result["annotated_frame"])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit requested.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C.")

    finally:
        print("\nShutting down...")
        stop_event.set()
        capture.join(timeout=_JOIN_TIMEOUT)
        inference.join(timeout=_JOIN_TIMEOUT)
        cv2.destroyAllWindows()

        if capture.is_alive() or inference.is_alive():
            print(
                "Warning: one or more threads did not exit within the join "
                "timeout. This may indicate a blocked queue operation or a "
                "MediaPipe shutdown delay."
            )

        print(f"Processed {frame_count} frames total.")
        print(f"Final classifier backend: {inference.get_classifier_backend()}")
        print("Schema validation passed on every processed frame.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AirSign standalone perception pipeline validator "
                    "(Dev A, Phase 7)."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0).",
    )
    args = parser.parse_args()

    run(camera_index=args.camera)