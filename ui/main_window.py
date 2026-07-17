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

_DEFAULT_WINDOW_WIDTH = 1600
_DEFAULT_WINDOW_HEIGHT = 900


class MainWindow(QMainWindow):
    """Top-level window composing and wiring every AirSign UI panel.

    Constructs a single :class:`~ui.signals.AppSignals` instance and
    exposes it via :attr:`app_signals` so that background threads
    (such as ``ActionThread``, constructed elsewhere) can be given a
    reference to emit on.

    The layout is camera-first, in the style of a computer-vision
    debugging tool: a full-width
    :class:`~ui.gesture_panel.GesturePanel` strip (state, gesture,
    confidence, and the recent event log) sits at the top so those
    values are always visible; below it, a horizontal splitter gives
    the :class:`~ui.video_panel.VideoPanel` the majority of the
    width, with the :class:`~ui.smoothing_graph.SmoothingGraph` beside
    it as a secondary, always-visible telemetry panel; a ``QTabWidget``
    holding only the "Settings" tab
    (:class:`~ui.settings_panel.SettingsPanel`) sits at the bottom. A
    status bar displays live FPS, hand-detection status, and the
    current FSM state.

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
        """Assembles the camera-first, CV-tool-style window layout.

        A full-width gesture panel sits on top so state, gesture,
        confidence, and the recent event log are always visible. Below
        it, a horizontal splitter gives the video panel roughly
        60-70% of the width, with the smoothing graph beside it as a
        secondary, always-visible panel. A tab widget holding only the
        "Settings" tab sits at the bottom.

        ``setStretchFactor`` alone only governs how *extra* space is
        redistributed on a later resize; it does not determine the
        splitters' first-shown proportions, which Qt otherwise derives
        from each child widget's size hint. Since ``SmoothingGraph``'s
        pyqtgraph plots and ``SettingsPanel``'s form layout both
        report large size hints, an explicit initial ``setSizes`` call
        (proportional to the window's own default size, not an
        arbitrary constant) is required so the camera feed is
        genuinely dominant from the moment the window first appears;
        stretch factors then keep that same proportion on every
        subsequent resize.
        """
        camera_graph_splitter = QSplitter(Qt.Orientation.Horizontal)
        camera_graph_splitter.addWidget(self.video_panel)
        camera_graph_splitter.addWidget(self.smoothing_graph)
        camera_graph_splitter.setStretchFactor(0, 2)
        camera_graph_splitter.setStretchFactor(1, 1)
        camera_graph_splitter.setChildrenCollapsible(False)
        camera_graph_splitter.setSizes(
            [round(_DEFAULT_WINDOW_WIDTH * 0.72), round(_DEFAULT_WINDOW_WIDTH * 0.28)]
        )

        tab_widget = QTabWidget()
        tab_widget.addTab(self.settings_panel, "Settings")

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(self.gesture_panel)
        main_splitter.addWidget(camera_graph_splitter)
        main_splitter.addWidget(tab_widget)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setSizes(
            [
                round(_DEFAULT_WINDOW_HEIGHT * 0.15),
                round(_DEFAULT_WINDOW_HEIGHT * 0.70),
                round(_DEFAULT_WINDOW_HEIGHT * 0.15),
            ]
        )

        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.addWidget(main_splitter)
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
