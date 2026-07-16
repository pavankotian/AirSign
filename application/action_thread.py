"""Gesture event consumption thread for the AirSign Developer B track.

This module implements :class:`ActionThread`, the ``QThread`` subclass
responsible for consuming perception results from Dev A's
``result_queue``. This revision covers thread lifecycle only: startup,
dependency injection, queue polling with timeout, graceful shutdown,
and safe interruption. Event-dispatch logic (translating FSM events
into mouse/keyboard actions and session log entries) is added in a
later revision.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING, Any, Optional

from PyQt6.QtCore import QThread

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
    """Consumes perception results and manages its own lifecycle.

    This class is a ``QThread`` subclass responsible for pulling result
    values from a shared ``result_queue`` (populated by Dev A's
    ``InferenceThread``) and, in a later revision, dispatching the
    corresponding OS mouse/keyboard actions and session log entries.
    This revision implements lifecycle management only: consuming the
    queue, timing out safely on an empty queue, and terminating
    cleanly on request.

    All Qt signal emission for this thread must go through the
    injected ``app_signals`` object; ``ActionThread`` itself never
    declares its own ``pyqtSignal`` attributes.

    Attributes:
        result_queue: The queue to consume perception result values
            from.
        stop_event: A ``threading.Event`` used to signal the run loop
            to exit.
        settings_manager: The settings manager dependency, injected for
            use by later event-dispatch logic.
        coordinate_mapper: The coordinate mapper dependency, injected
            for use by later event-dispatch logic.
        mouse_injector: The mouse injector dependency, injected for use
            by later event-dispatch logic.
        keyboard_injector: The keyboard injector dependency, injected
            for use by later event-dispatch logic.
        event_logger: The event logger dependency, injected for use by
            later event-dispatch logic.
        app_signals: An optional ``AppSignals`` instance (see
            ``ui/signals.py``) used to notify the UI of state changes.
    """

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
                self._handle_result(result)
            except Exception:
                logger.exception(
                    "Unexpected error while handling a result; continuing"
                )

        with self._lock:
            self._running = False

        logger.info("ActionThread run loop terminated")

    def _handle_result(self, result: Any) -> None:
        """Processes a single result value consumed from the queue.

        This revision intentionally contains no event-dispatch logic.
        It is the extension point where FSM event handling, cursor
        movement, OS action dispatch, and session logging will be
        added in a later revision.

        Args:
            result: The value consumed from ``result_queue``.
        """
        logger.debug("Result received (dispatch logic not yet implemented)")

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
