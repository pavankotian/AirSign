"""Application composition root for AirSign.

This module owns nothing but wiring: it constructs every application-
wide object exactly once, injects dependencies where required, starts
the background threads in the approved order, launches the Qt event
loop, and tears everything down cleanly on exit. It contains no
gesture, dispatch, or configuration logic of its own -- all of that
lives in the already-frozen Developer A and Developer B modules this
file only assembles.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtWidgets import QApplication

from application.action_thread import ActionThread
from application.config.bindings import load_bindings
from application.config.settings_manager import SettingsManager
from application.db.event_logger import EventLogger
from application.os_integration.coordinate_mapper import CoordinateMapper
from application.os_integration.keyboard_controller import KeyboardInjector
from application.os_integration.mouse_controller import MouseInjector
from perception.capture_thread import CaptureThread
from perception.inference_thread import InferenceThread
from shared.constants import CAPTURE_QUEUE_MAXSIZE, RESULT_QUEUE_MAXSIZE
from tray.tray_app import TrayApplication
from ui.main_window import MainWindow

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

_THREAD_JOIN_TIMEOUT_SECONDS = 2.0
_ACTION_THREAD_STOP_TIMEOUT_MS = 2000


def _configure_logging() -> None:
    """Configures application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    """Composes, wires, and runs the AirSign application.

    Constructs exactly one instance of every application-wide object,
    wires them together per the approved dependency graph, starts the
    perception and dispatch threads, shows the main window, and runs
    the Qt event loop until the application quits.

    Returns:
        The process exit code: the Qt event loop's exit code on a
        normal run, or ``1`` if startup failed before the event loop
        could be entered.
    """
    _configure_logging()

    event_logger: Optional[EventLogger] = None
    tray: Optional[TrayApplication] = None

    try:
        app = QApplication(sys.argv)

        stop_event = threading.Event()
        frame_queue: "queue.Queue[Any]" = queue.Queue(maxsize=CAPTURE_QUEUE_MAXSIZE)
        result_queue: "queue.Queue[dict[str, Any]]" = queue.Queue(
            maxsize=RESULT_QUEUE_MAXSIZE
        )

        settings_manager = SettingsManager(config_path=CONFIG_PATH)
        load_bindings(settings_manager)

        coordinate_mapper = CoordinateMapper(settings_manager)
        mouse_injector = MouseInjector(settings_manager)
        keyboard_injector = KeyboardInjector()
        event_logger = EventLogger()

        window = MainWindow(settings_manager)
        tray = TrayApplication(app, window)

        action_thread = ActionThread(
            result_queue=result_queue,
            stop_event=stop_event,
            settings_manager=settings_manager,
            coordinate_mapper=coordinate_mapper,
            mouse_injector=mouse_injector,
            keyboard_injector=keyboard_injector,
            event_logger=event_logger,
            app_signals=window.app_signals,
        )

        inference_thread = InferenceThread(
            frame_queue=frame_queue,
            result_queue=result_queue,
            stop_event=stop_event,
            config=settings_manager.as_dict(),
        )

        capture_thread = CaptureThread(
            frame_queue=frame_queue,
            stop_event=stop_event,
            device_index=settings_manager.get("camera.device_index", 0),
        )
    except Exception:
        logger.exception("AirSign failed to start")
        if tray is not None:
            tray.shutdown()
        if event_logger is not None:
            event_logger.close()
        return 1

    def _shutdown() -> None:
        """Stops every background thread and releases resources cleanly.

        Threads are stopped in producer-to-consumer order so each
        queue drains naturally rather than racing an upstream producer
        that is still running.
        """
        logger.info("Shutting down AirSign")
        stop_event.set()
        capture_thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
        inference_thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
        action_thread.stop(timeout_ms=_ACTION_THREAD_STOP_TIMEOUT_MS)
        tray.shutdown()
        event_logger.close()
        logger.info("AirSign shutdown complete")

    app.aboutToQuit.connect(_shutdown)

    capture_thread.start()
    inference_thread.start()
    action_thread.start()

    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
