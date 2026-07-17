"""Qt signal interface for the AirSign Developer B UI layer.

This module defines :class:`AppSignals`, the single ``QObject`` that
declares every ``pyqtSignal`` used to communicate from
``ActionThread`` (and any other background thread) to the PyQt6 UI.
No other module in the codebase may declare its own ``pyqtSignal``
attributes; all cross-thread notification is routed through an
``AppSignals`` instance passed by reference to whatever component
needs to emit or connect to it.
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class AppSignals(QObject):
    """Central collection of Qt signals shared across the AirSign UI.

    A single instance of this class is constructed once and injected
    into every component that needs to emit or receive cross-thread
    notifications (for example, ``ActionThread`` emits on these
    signals, and ``MainWindow`` connects its widgets' slots to them).
    This class contains no business logic and no helper methods; it is
    purely a declarative signal interface.

    Signals:
        frame_ready: Emitted once per processed camera frame, carrying
            the annotated frame to display in the video panel.

            Args:
                frame (np.ndarray): The annotated BGR frame, shape
                    ``(height, width, 3)``, dtype ``uint8``.

        graph_update: Emitted once per processed frame, carrying the
            raw and 1€-filtered cursor coordinates for the smoothing
            graph.

            Args:
                raw_x (float): Raw (unfiltered) cursor x-coordinate.
                raw_y (float): Raw (unfiltered) cursor y-coordinate.
                filtered_x (float): 1€-filtered cursor x-coordinate.
                filtered_y (float): 1€-filtered cursor y-coordinate.

        gesture_event: Emitted only on frames where an FSM event
            fired, carrying the event and the gesture context that
            produced it.

            Args:
                fsm_event (str): The FSM event string that fired, a
                    member of ``shared.fsm_states.FSM_EVENTS``.
                static_gesture (str): The classified gesture label
                    active on this frame, a member of
                    ``shared.gesture_labels.GESTURE_LABELS``.
                gesture_confidence (float): Classifier confidence for
                    ``static_gesture``, in the range 0.0-1.0.

        state_changed: Emitted only when the FSM state differs from
            the previous frame's state.

            Args:
                fsm_state (str): The new FSM state, a member of
                    ``shared.fsm_states.FSM_STATES``.

        perf_update: Emitted once per processed frame, carrying
            pipeline performance and hand-detection status for the
            status bar.

            Args:
                inference_fps (float): Current measured inference
                    throughput in frames per second.
                hand_detected (bool): Whether a hand was detected in
                    this frame.

        export_complete: Emitted when a PDF session report export
            finishes, carrying the path of the generated file.

            Args:
                file_path (str): Filesystem path of the exported PDF
                    report.
    """

    frame_ready = pyqtSignal(np.ndarray)
    graph_update = pyqtSignal(float, float, float, float)
    gesture_event = pyqtSignal(str, str, float)
    state_changed = pyqtSignal(str)
    perf_update = pyqtSignal(float, bool)
    export_complete = pyqtSignal(str)
