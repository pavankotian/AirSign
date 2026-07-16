"""FSM state and gesture event display panel for the AirSign Developer B UI.

This module implements :class:`GesturePanel`, a ``QWidget`` that
displays the current FSM state, the most recently classified gesture,
its confidence, and a scrolling log of recent gesture events. It
contains no gesture-recognition or threading logic; it only renders
values handed to it by ``AppSignals.state_changed`` and
``AppSignals.gesture_event``.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QFormLayout,
    QLabel,
    QListWidget,
    QVBoxLayout,
    QWidget,
)

from shared.constants import ACTION_LOG_MAX_ROWS, GESTURE_FLASH_MS

logger = logging.getLogger(__name__)


class GesturePanel(QWidget):
    """Displays live FSM state, gesture, confidence, and a recent event log.

    The panel exposes two slots intended to be connected directly to
    ``AppSignals``: :meth:`update_state` for ``state_changed`` and
    :meth:`update_gesture_event` for ``gesture_event``. Each fired
    gesture event briefly flashes the state label's background before
    it reverts to the color associated with the current FSM state.

    Attributes:
        STATE_COLORS: Mapping of FSM state string to its display
            background color.
    """

    STATE_COLORS: dict[str, str] = {
        "IDLE": "#808080",
        "HOVER": "#2196F3",
        "PINCH_START": "#FFA500",
        "PINCH_HELD": "#FF0000",
        "SCROLL_MODE": "#800080",
        "SWIPE_PENDING": "#FFFF00",
        "SWIPE_COMMIT": "#008000",
    }

    _DEFAULT_STATE = "IDLE"
    _FLASH_COLOR = "#FFFFFF"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initializes the gesture panel.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)

        self._current_state: str = self._DEFAULT_STATE

        self._state_label = QLabel(self._DEFAULT_STATE)
        self._gesture_label = QLabel("—")
        self._confidence_label = QLabel("—")
        self._event_log = QListWidget()

        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._restore_state_label_style)

        self._build_layout()
        self._apply_state_style(self._current_state)

        logger.info("GesturePanel initialized")

    def _build_layout(self) -> None:
        """Assembles the panel's child widgets into their layout."""
        form_layout = QFormLayout()
        form_layout.addRow("State:", self._state_label)
        form_layout.addRow("Gesture:", self._gesture_label)
        form_layout.addRow("Confidence:", self._confidence_label)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(form_layout)
        main_layout.addWidget(QLabel("Recent Event Log:"))
        main_layout.addWidget(self._event_log)
        self.setLayout(main_layout)

    def update_state(self, fsm_state: str) -> None:
        """Updates the displayed FSM state and its color.

        Intended to be connected directly to
        ``AppSignals.state_changed``.

        Args:
            fsm_state: The new FSM state string.
        """
        self._current_state = fsm_state
        self._state_label.setText(fsm_state)
        self._apply_state_style(fsm_state)
        logger.debug("FSM state updated to %s", fsm_state)

    def update_gesture_event(
        self, fsm_event: str, static_gesture: str, confidence: float
    ) -> None:
        """Updates the gesture/confidence display, logs, and flashes the state.

        Intended to be connected directly to
        ``AppSignals.gesture_event``.

        Args:
            fsm_event: The FSM event string that fired.
            static_gesture: The classified gesture label active when
                the event fired.
            confidence: Classifier confidence for ``static_gesture``,
                in the range 0.0-1.0.
        """
        self._gesture_label.setText(static_gesture)
        self._confidence_label.setText(f"{confidence:.2f}")
        self._append_log_entry(fsm_event, static_gesture, confidence)
        self._flash_state_label()
        logger.debug(
            "Gesture event received: fsm_event=%s static_gesture=%s "
            "confidence=%.2f",
            fsm_event,
            static_gesture,
            confidence,
        )

    def _append_log_entry(
        self, fsm_event: str, static_gesture: str, confidence: float
    ) -> None:
        """Inserts a new event log entry at the top, trimming old entries.

        Args:
            fsm_event: The FSM event string that fired.
            static_gesture: The classified gesture label active when
                the event fired.
            confidence: Classifier confidence for ``static_gesture``.
        """
        entry_text = f"{fsm_event} | gesture={static_gesture} | confidence={confidence:.2f}"
        self._event_log.insertItem(0, entry_text)

        while self._event_log.count() > ACTION_LOG_MAX_ROWS:
            last_row = self._event_log.count() - 1
            self._event_log.takeItem(last_row)

    def _apply_state_style(self, fsm_state: str) -> None:
        """Applies the background color associated with an FSM state.

        Args:
            fsm_state: The FSM state string to color the label for. If
                the state is not recognized, the default color for
                ``IDLE`` is used.
        """
        color = self.STATE_COLORS.get(
            fsm_state, self.STATE_COLORS[self._DEFAULT_STATE]
        )
        self._state_label.setStyleSheet(
            f"background-color: {color}; color: white; padding: 4px; "
            "border-radius: 4px;"
        )

    def _flash_state_label(self) -> None:
        """Briefly flashes the state label background, then restores it.

        The flash duration is
        :data:`shared.constants.GESTURE_FLASH_MS`.
        """
        self._state_label.setStyleSheet(
            f"background-color: {self._FLASH_COLOR}; color: black; "
            "padding: 4px; border-radius: 4px;"
        )
        self._flash_timer.start(GESTURE_FLASH_MS)

    def _restore_state_label_style(self) -> None:
        """Restores the state label to its current state's normal color."""
        self._apply_state_style(self._current_state)
