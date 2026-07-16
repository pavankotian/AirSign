"""Main window composition root for the AirSign Developer B UI.

This module implements :class:`MainWindow`, the top-level
``QMainWindow`` that constructs every UI panel, owns the single
``AppSignals`` instance shared with background threads, and wires each
signal to the panel or status bar element responsible for displaying
it. It contains no gesture, threading, or business logic of its own;
it only composes and connects existing, independently frozen widgets.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.gesture_panel import GesturePanel
from ui.settings_panel import SettingsPanel
from ui.signals import AppSignals
from ui.smoothing_graph import SmoothingGraph
from ui.video_panel import VideoPanel

if TYPE_CHECKING:
    from application.config.settings_manager import SettingsManager

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_WIDTH = 1216
_DEFAULT_WINDOW_HEIGHT = 765


class MainWindow(QMainWindow):
    """Top-level window composing and wiring every AirSign UI panel.

    Constructs a single :class:`~ui.signals.AppSignals` instance and
    exposes it via :attr:`app_signals` so that background threads
    (such as ``ActionThread``, constructed elsewhere) can be given a
    reference to emit on. The window lays out a
    :class:`~ui.video_panel.VideoPanel` on the left and, on the right,
    a :class:`~ui.gesture_panel.GesturePanel` above a ``QTabWidget``
    containing a "Graph" tab
    (:class:`~ui.smoothing_graph.SmoothingGraph`) and a "Settings" tab
    (:class:`~ui.settings_panel.SettingsPanel`). A status bar displays
    live FPS, hand-detection status, and the current FSM state.

    No colors are hardcoded anywhere in this class, so an external
    stylesheet (for example, a dark theme ``.qss`` file) can be applied
    to the application without conflicting with inline styling here.

    Attributes:
        app_signals: The shared signal bus every background thread and
            UI panel connects to.
        video_panel: The video display panel.
        gesture_panel: The FSM state and gesture event display panel.
        smoothing_graph: The cursor smoothing comparison graph.
        settings_panel: The configuration editor panel.
    """

    def __init__(
        self,
        settings_manager: "SettingsManager",
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initializes the main window, its panels, and signal wiring.

        Args:
            settings_manager: The settings manager passed through to
                :class:`~ui.settings_panel.SettingsPanel`.
            parent: Optional parent widget.
        """
        super().__init__(parent)

        self.app_signals = AppSignals()

        self.video_panel = VideoPanel()
        self.gesture_panel = GesturePanel()
        self.smoothing_graph = SmoothingGraph()
        self.settings_panel = SettingsPanel(settings_manager)

        self._fps_label = QLabel("FPS: --")
        self._hand_status_label = QLabel("Hand: --")
        self._fsm_state_label = QLabel("State: --")

        self.setWindowTitle("AirSign - Zero-Touch Interface Controller")
        self.resize(_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)

        self._build_layout()
        self._build_status_bar()
        self._connect_signals()

        logger.info("MainWindow initialized")

    def _build_layout(self) -> None:
        """Assembles the video panel, gesture panel, and tab widget."""
        tab_widget = QTabWidget()
        tab_widget.addTab(self.smoothing_graph, "Graph")
        tab_widget.addTab(self.settings_panel, "Settings")

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.gesture_panel)
        right_layout.addWidget(tab_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        central_widget = QWidget()
        central_layout = QHBoxLayout(central_widget)
        central_layout.addWidget(splitter)
        self.setCentralWidget(central_widget)

    def _build_status_bar(self) -> None:
        """Adds the FPS, hand-status, and FSM state labels to the status bar."""
        status_bar = self.statusBar()
        status_bar.addPermanentWidget(self._fsm_state_label)
        status_bar.addPermanentWidget(self._hand_status_label)
        status_bar.addPermanentWidget(self._fps_label)

    def _connect_signals(self) -> None:
        """Connects every ``AppSignals`` signal to its display target."""
        self.app_signals.frame_ready.connect(self.video_panel.update_frame)
        self.app_signals.graph_update.connect(self.smoothing_graph.update_graph)
        self.app_signals.gesture_event.connect(
            self.gesture_panel.update_gesture_event
        )
        self.app_signals.state_changed.connect(self.gesture_panel.update_state)
        self.app_signals.state_changed.connect(self._on_state_changed)
        self.app_signals.perf_update.connect(self._on_perf_update)

    def _on_perf_update(self, inference_fps: float, hand_detected: bool) -> None:
        """Updates the FPS and hand-detection status bar labels.

        Intended to be connected to ``AppSignals.perf_update``.

        Args:
            inference_fps: Current measured inference throughput in
                frames per second.
            hand_detected: Whether a hand was detected in the current
                frame.
        """
        self._fps_label.setText(f"FPS: {inference_fps:.1f}")
        self._hand_status_label.setText(
            "Hand: Detected" if hand_detected else "Hand: Not Detected"
        )

    def _on_state_changed(self, fsm_state: str) -> None:
        """Updates the FSM state status bar label.

        Intended to be connected to ``AppSignals.state_changed``.

        Args:
            fsm_state: The new FSM state string.
        """
        self._fsm_state_label.setText(f"State: {fsm_state}")
