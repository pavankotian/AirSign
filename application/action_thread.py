"""Gesture event consumption thread for the AirSign Developer B track.

This module implements :class:`ActionThread`, the ``QThread`` subclass
responsible for consuming perception results from Dev A's
``result_queue``, forwarding per-frame data to the UI via
``AppSignals``, and dispatching the OS mouse/keyboard action bound to
each fired FSM event. Thread lifecycle (startup, queue polling,
graceful shutdown, safe interruption) is unchanged from the prior
revision; this revision adds the event-processing and dispatch layer
only.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

from PyQt6.QtCore import QThread

from application.config.bindings import get_binding
from shared.constants import QUEUE_GET_TIMEOUT

if TYPE_CHECKING:
    from application.config.settings_manager import SettingsManager
    from application.db.event_logger import EventLogger
    from application.os_integration.coordinate_mapper import CoordinateMapper
    from application.os_integration.keyboard_controller import KeyboardInjector
    from application.os_integration.mouse_controller import MouseInjector
    from ui.signals import AppSignals

logger = logging.getLogger(__name__)


class ActionThread(QThread):
    """Consumes perception results, updates the UI, and dispatches actions.

    This class is a ``QThread`` subclass responsible for pulling result
    dictionaries from a shared ``result_queue`` (populated by Dev A's
    ``InferenceThread``), forwarding per-frame data to the UI through
    ``AppSignals``, moving the OS cursor every frame, and dispatching
    the OS mouse or keyboard action bound to whichever FSM event fired
    on that frame, logging the dispatched action to the session
    database.

    Cursor movement is unconditional: it happens on every frame with a
    non-``None`` ``cursor_screen_pos``, independent of whether an FSM
    event fired. Click and keyboard action dispatch, by contrast, only
    occur when ``fsm_event`` is present and a binding exists for it.

    All Qt signal emission for this thread must go through the
    injected ``app_signals`` object; ``ActionThread`` itself never
    declares its own ``pyqtSignal`` attributes. Per frame,
    ``frame_ready``, ``graph_update``, and ``perf_update`` are always
    emitted; ``state_changed`` is emitted only when ``fsm_state``
    differs from the previous frame; ``gesture_event`` is emitted only
    when ``fsm_event`` is not ``None``.

    ``cursor_screen_pos`` values consumed from a result are already
    filtered, mapped, and clamped by ``InferenceThread``; this class
    never calls ``CoordinateMapper`` and never reads raw landmarks.

    Attributes:
        result_queue: The queue to consume perception result values
            from.
        stop_event: A ``threading.Event`` used to signal the run loop
            to exit.
        settings_manager: The settings manager dependency, injected for
            use by later event-dispatch logic.
        coordinate_mapper: The coordinate mapper dependency, injected
            for potential calibration use elsewhere; not used on the
            per-frame dispatch path.
        mouse_injector: The mouse injector used for per-frame cursor
            movement and dispatched mouse actions.
        keyboard_injector: The keyboard injector used to dispatch bound
            keyboard actions.
        event_logger: The event logger used to record every dispatched
            action.
        app_signals: An optional ``AppSignals`` instance (see
            ``ui/signals.py``) used to notify the UI of state changes.
    """

    _MOUSE_ACTIONS: dict[str, str] = {
        "left_click": "left_click",
        "right_click": "right_click",
        "double_click": "double_click",
        "mouse_drag": "press",
        "mouse_release": "release",
    }

    def __init__(
        self,
        result_queue: "queue.Queue[Any]",
        stop_event: threading.Event,
        settings_manager: "SettingsManager",
        coordinate_mapper: "CoordinateMapper",
        mouse_injector: "MouseInjector",
        keyboard_injector: "KeyboardInjector",
        event_logger: "EventLogger",
        app_signals: Optional["AppSignals"] = None,
        queue_timeout: float = QUEUE_GET_TIMEOUT,
    ) -> None:
        """Initializes the action thread with its dependencies.

        Args:
            result_queue: The queue to consume perception result
                values from. Any object exposing a ``get(timeout=...)``
                method that raises ``queue.Empty`` on timeout is
                accepted.
            stop_event: A ``threading.Event`` used to signal shutdown.
            settings_manager: The settings manager dependency.
            coordinate_mapper: The coordinate mapper dependency.
            mouse_injector: The mouse injector dependency.
            keyboard_injector: The keyboard injector dependency.
            event_logger: The event logger dependency.
            app_signals: An optional ``AppSignals`` instance used for
                UI notification. Defaults to ``None``.
            queue_timeout: Seconds to wait on each
                ``result_queue.get()`` call before checking
                ``stop_event`` again. Defaults to
                :data:`shared.constants.QUEUE_GET_TIMEOUT`.
        """
        super().__init__()

        self._lock = threading.RLock()
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.settings_manager = settings_manager
        self.coordinate_mapper = coordinate_mapper
        self.mouse_injector = mouse_injector
        self.keyboard_injector = keyboard_injector
        self.event_logger = event_logger
        self.app_signals = app_signals
        self._queue_timeout = queue_timeout
        self._running = False
        self._last_fsm_state: Optional[str] = None

        logger.info(
            "ActionThread initialized (queue_timeout=%s)", self._queue_timeout
        )

    def run(self) -> None:
        """Runs the consume loop until ``stop_event`` is set.

        Reads from ``result_queue`` with a timeout on every iteration
        so ``stop_event`` is checked regularly instead of blocking
        indefinitely, which avoids busy-waiting while still supporting
        prompt, graceful interruption. ``queue.Empty`` is expected and
        handled silently; any other unexpected exception raised while
        reading or handling a result is logged and does not terminate
        the loop, so a single bad frame cannot crash the thread.
        """
        with self._lock:
            self._running = True

        logger.info("ActionThread run loop starting")

        while not self.stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=self._queue_timeout)
            except queue.Empty:
                continue
            except Exception:
                logger.exception(
                    "Unexpected error reading from result_queue; continuing"
                )
                continue

            try:
                self._process_result(result)
            except Exception:
                logger.exception(
                    "Unexpected error while handling a result; continuing"
                )

        with self._lock:
            self._running = False

        logger.info("ActionThread run loop terminated")

    def _process_result(self, result: dict[str, Any]) -> None:
        """Processes one perception result: UI updates, then event dispatch.

        Every frame is forwarded to the UI (frame, graph, and
        performance signals) regardless of whether an FSM event fired.
        Cursor movement likewise happens on every frame whenever
        ``cursor_screen_pos`` is available, independent of whether an
        FSM event fired. Click and keyboard action dispatch, by
        contrast, only occur when ``fsm_event`` is present. Each step
        is isolated in its own try/except so a failure in one (for
        example, a UI signal emission failing before the UI is
        connected) never prevents the remaining steps from running.

        Args:
            result: A perception result dictionary conforming to the
                finalized ``RESULT_SCHEMA`` contract. Only the
                ``annotated_frame``, ``graph_data``, ``fsm_state``,
                ``fsm_event``, ``gesture_confidence``,
                ``static_gesture``, ``cursor_screen_pos``,
                ``inference_fps``, and ``hand_detected`` keys are
                used.
        """
        try:
            self._emit_frame(result)
        except Exception:
            logger.exception("Failed to emit frame_ready signal")

        try:
            self._emit_graph(result)
        except Exception:
            logger.exception("Failed to emit graph_update signal")

        try:
            self._emit_perf(result)
        except Exception:
            logger.exception("Failed to emit perf_update signal")

        try:
            self._emit_state_change(result)
        except Exception:
            logger.exception("Failed to emit state_changed signal")

        try:
            self._move_cursor(result)
        except Exception:
            logger.exception("Failed to move cursor")

        try:
            self._emit_gesture_event(result)
        except Exception:
            logger.exception("Failed to emit gesture_event signal")

        try:
            self._dispatch_event(result)
        except Exception:
            logger.exception("Failed to dispatch FSM event")

    def _emit_frame(self, result: dict[str, Any]) -> None:
        """Forwards the annotated frame to the UI via ``AppSignals``.

        Args:
            result: The perception result dictionary for this frame.
                Only the ``annotated_frame`` key is used.
        """
        if self.app_signals is None:
            return
        self.app_signals.frame_ready.emit(result.get("annotated_frame"))

    def _emit_graph(self, result: dict[str, Any]) -> None:
        """Forwards cursor smoothing graph data to the UI via ``AppSignals``.

        ``graph_update`` is emitted with four separate float arguments
        rather than the raw ``graph_data`` dictionary, matching the
        ``AppSignals`` contract.

        Args:
            result: The perception result dictionary for this frame.
                Only the ``graph_data`` key is used, expected to
                contain ``raw_x``, ``raw_y``, ``filtered_x``, and
                ``filtered_y`` numeric values.
        """
        if self.app_signals is None:
            return

        graph_data = result.get("graph_data") or {}
        raw_x = float(graph_data.get("raw_x", 0.0))
        raw_y = float(graph_data.get("raw_y", 0.0))
        filtered_x = float(graph_data.get("filtered_x", 0.0))
        filtered_y = float(graph_data.get("filtered_y", 0.0))

        self.app_signals.graph_update.emit(raw_x, raw_y, filtered_x, filtered_y)

    def _emit_perf(self, result: dict[str, Any]) -> None:
        """Forwards performance and hand-detection status to the UI.

        Args:
            result: The perception result dictionary for this frame.
                Only the ``inference_fps`` and ``hand_detected`` keys
                are used.
        """
        if self.app_signals is None:
            return
        self.app_signals.perf_update.emit(
            result.get("inference_fps"), result.get("hand_detected")
        )

    def _emit_state_change(self, result: dict[str, Any]) -> None:
        """Emits ``state_changed`` only when the FSM state has changed.

        Compares this frame's ``fsm_state`` against the state recorded
        on the previous frame; the signal is emitted only when the two
        differ, so a UI subscriber sees exactly one notification per
        state transition rather than one per frame.

        Args:
            result: The perception result dictionary for this frame.
                Only the ``fsm_state`` key is used.
        """
        fsm_state = result.get("fsm_state")

        with self._lock:
            state_changed = fsm_state != self._last_fsm_state
            self._last_fsm_state = fsm_state

        if not state_changed:
            return

        if self.app_signals is None:
            return

        self.app_signals.state_changed.emit(fsm_state)

    def _move_cursor(self, result: dict[str, Any]) -> None:
        """Moves the OS cursor every frame whenever a position is available.

        Cursor movement is intentionally independent of ``fsm_event``:
        it happens on every frame that reports a non-``None``
        ``cursor_screen_pos``, whether or not an FSM event fired on
        that frame. The position is used directly; ``CoordinateMapper``
        is never called here.

        Args:
            result: The perception result dictionary for this frame.
                Only the ``cursor_screen_pos`` key is used.
        """
        cursor_screen_pos = result.get("cursor_screen_pos")
        if cursor_screen_pos is None:
            return

        cursor_x, cursor_y = cursor_screen_pos
        self.mouse_injector.move(cursor_x, cursor_y)

    def _emit_gesture_event(self, result: dict[str, Any]) -> None:
        """Emits ``gesture_event`` only when an FSM event fired this frame.

        Args:
            result: The perception result dictionary for this frame.
                Only the ``fsm_event``, ``static_gesture``, and
                ``gesture_confidence`` keys are used.
        """
        fsm_event = result.get("fsm_event")
        if fsm_event is None:
            return

        if self.app_signals is None:
            return

        self.app_signals.gesture_event.emit(
            fsm_event,
            result.get("static_gesture"),
            result.get("gesture_confidence"),
        )

    def _dispatch_event(self, result: dict[str, Any]) -> None:
        """Dispatches the OS action bound to this frame's FSM event, if any.

        If ``fsm_event`` is ``None``, this method returns immediately
        and executes no OS action. If a binding exists for the fired
        event, the corresponding mouse or keyboard action is executed.
        Cursor movement is not performed here: it happens unconditionally
        every frame via :meth:`_move_cursor`, independent of whether an
        FSM event fired. ``CoordinateMapper`` is never called and no
        landmark data is read.

        Args:
            result: The perception result dictionary for this frame.
                Only the ``fsm_event`` key is used directly here;
                ``static_gesture``, ``fsm_state``, and
                ``gesture_confidence`` are used by :meth:`_log_event`.
        """
        fsm_event = result.get("fsm_event")
        if fsm_event is None:
            return

        action = get_binding(fsm_event)
        if action is None:
            return

        if action in self._MOUSE_ACTIONS:
            method_name = self._MOUSE_ACTIONS[action]
            getattr(self.mouse_injector, method_name)()
        else:
            self.keyboard_injector.perform_action(action)

        self._log_event(result, action)

    def _log_event(self, result: dict[str, Any], action: str) -> None:
        """Logs a dispatched action to the session database.

        Only called when an action was actually dispatched; frames
        with no fired event or no bound action are never logged.

        Args:
            result: The perception result dictionary for this frame.
                The ``static_gesture``, ``fsm_state``, ``fsm_event``,
                ``gesture_confidence``, and ``inference_fps`` keys are
                used.
            action: The action identifier that was dispatched.
        """
        inference_fps = result.get("inference_fps")
        latency_ms = (1000.0 / inference_fps) if inference_fps else 0.0

        self.event_logger.log_event(
            timestamp=time.time(),
            gesture=result.get("static_gesture"),
            fsm_state=result.get("fsm_state"),
            fsm_event=result.get("fsm_event"),
            action=action,
            confidence=result.get("gesture_confidence"),
            latency_ms=latency_ms,
        )

    def stop(self, timeout_ms: int = 2000) -> None:
        """Signals the run loop to exit and waits for the thread to finish.

        Safe to call more than once. If the thread is not currently
        running (never started, or already finished), this method
        signals shutdown and returns immediately without waiting.

        Args:
            timeout_ms: Maximum milliseconds to wait for the thread to
                finish via ``QThread.wait``. Defaults to 2000.
        """
        logger.info("ActionThread stop requested")
        self.stop_event.set()

        if self.isRunning():
            finished = self.wait(timeout_ms)
            if not finished:
                logger.warning(
                    "ActionThread did not terminate within %s ms", timeout_ms
                )

        with self._lock:
            self._running = False

        logger.info("ActionThread stopped")

    def is_running(self) -> bool:
        """Returns whether the run loop is currently executing.

        Returns:
            True if the run loop has started and has not yet exited,
            False otherwise.
        """
        with self._lock:
            return self._running
