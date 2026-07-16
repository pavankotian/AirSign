"""Configuration editor panel for the AirSign Developer B UI.

This module implements :class:`SettingsPanel`, a ``QWidget`` that
provides a graphical editor for the application configuration. All
reads and writes go exclusively through the injected
:class:`~application.config.settings_manager.SettingsManager`; this
module never opens or parses ``config.json`` directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from application.config.settings_manager import SettingsManager

logger = logging.getLogger(__name__)


class SettingsPanel(QWidget):
    """Graphical editor for the application configuration.

    All configuration values are loaded from and written back through
    :class:`~application.config.settings_manager.SettingsManager`
    using its dot-notation ``get``/``set`` interface; this panel never
    touches the underlying JSON file. Two buttons control persistence:
    "Save Configuration" writes every edited widget's value into the
    settings manager and persists it to disk, and "Restore Defaults"
    reloads the last-saved configuration from disk and refreshes every
    widget to match.

    Attributes:
        settings_manager: The settings manager this panel reads from
            and writes to.
    """

    def __init__(
        self,
        settings_manager: "SettingsManager",
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initializes the settings panel and loads current values.

        Args:
            settings_manager: The settings manager to read initial
                values from and write edited values to.
            parent: Optional parent widget.
        """
        super().__init__(parent)

        self.settings_manager = settings_manager

        self._sensitivity_spinbox = QDoubleSpinBox()
        self._pinch_tolerance_spinbox = QDoubleSpinBox()
        self._mincutoff_spinbox = QDoubleSpinBox()
        self._beta_spinbox = QDoubleSpinBox()
        self._dcutoff_spinbox = QDoubleSpinBox()
        self._device_index_spinbox = QSpinBox()
        self._target_fps_spinbox = QSpinBox()
        self._bindings_table = QTableWidget()

        self._save_button = QPushButton("Save Configuration")
        self._restore_button = QPushButton("Restore Defaults")

        self._configure_widgets()
        self._build_layout()
        self._connect_signals()
        self._load_values()

        logger.info("SettingsPanel initialized")

    def _configure_widgets(self) -> None:
        """Sets ranges, decimals, and step sizes for every input widget."""
        self._sensitivity_spinbox.setRange(0.0, 5.0)
        self._sensitivity_spinbox.setDecimals(2)
        self._sensitivity_spinbox.setSingleStep(0.05)

        self._pinch_tolerance_spinbox.setRange(0.0, 1.0)
        self._pinch_tolerance_spinbox.setDecimals(2)
        self._pinch_tolerance_spinbox.setSingleStep(0.01)

        self._mincutoff_spinbox.setRange(0.0, 10.0)
        self._mincutoff_spinbox.setDecimals(3)
        self._mincutoff_spinbox.setSingleStep(0.01)

        self._beta_spinbox.setRange(0.0, 1.0)
        self._beta_spinbox.setDecimals(3)
        self._beta_spinbox.setSingleStep(0.001)

        self._dcutoff_spinbox.setRange(0.0, 10.0)
        self._dcutoff_spinbox.setDecimals(3)
        self._dcutoff_spinbox.setSingleStep(0.01)

        self._device_index_spinbox.setRange(0, 16)
        self._target_fps_spinbox.setRange(1, 240)

        self._bindings_table.setColumnCount(2)
        self._bindings_table.setHorizontalHeaderLabels(["FSM Event", "Action"])
        self._bindings_table.horizontalHeader().setStretchLastSection(True)

    def _build_layout(self) -> None:
        """Assembles the panel's group boxes, table, and buttons."""
        tracking_group = QGroupBox("Tracking")
        tracking_layout = QFormLayout()
        tracking_layout.addRow("Sensitivity:", self._sensitivity_spinbox)
        tracking_layout.addRow("Pinch Tolerance:", self._pinch_tolerance_spinbox)
        tracking_group.setLayout(tracking_layout)

        filter_group = QGroupBox("Filter")
        filter_layout = QFormLayout()
        filter_layout.addRow("Min Cutoff:", self._mincutoff_spinbox)
        filter_layout.addRow("Beta:", self._beta_spinbox)
        filter_layout.addRow("D Cutoff:", self._dcutoff_spinbox)
        filter_group.setLayout(filter_layout)

        camera_group = QGroupBox("Camera")
        camera_layout = QFormLayout()
        camera_layout.addRow("Device Index:", self._device_index_spinbox)
        camera_layout.addRow("Target FPS:", self._target_fps_spinbox)
        camera_group.setLayout(camera_layout)

        bindings_group = QGroupBox("Bindings")
        bindings_layout = QVBoxLayout()
        bindings_layout.addWidget(self._bindings_table)
        bindings_group.setLayout(bindings_layout)

        button_row = QHBoxLayout()
        button_row.addWidget(self._save_button)
        button_row.addWidget(self._restore_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(tracking_group)
        main_layout.addWidget(filter_group)
        main_layout.addWidget(camera_group)
        main_layout.addWidget(bindings_group)
        main_layout.addLayout(button_row)
        self.setLayout(main_layout)

    def _connect_signals(self) -> None:
        """Connects button clicks to their handler methods."""
        self._save_button.clicked.connect(self._on_save_clicked)
        self._restore_button.clicked.connect(self._on_restore_clicked)

    def _load_values(self) -> None:
        """Loads current values from the settings manager into all widgets."""
        self._sensitivity_spinbox.setValue(
            float(self.settings_manager.get("tracking.sensitivity", 0.85))
        )
        self._pinch_tolerance_spinbox.setValue(
            float(self.settings_manager.get("tracking.pinch_tolerance", 0.15))
        )
        self._mincutoff_spinbox.setValue(
            float(self.settings_manager.get("filter.mincutoff", 1.0))
        )
        self._beta_spinbox.setValue(
            float(self.settings_manager.get("filter.beta", 0.007))
        )
        self._dcutoff_spinbox.setValue(
            float(self.settings_manager.get("filter.dcutoff", 1.0))
        )
        self._device_index_spinbox.setValue(
            int(self.settings_manager.get("camera.device_index", 0))
        )
        self._target_fps_spinbox.setValue(
            int(self.settings_manager.get("camera.target_fps", 30))
        )
        self._load_bindings_table()

        logger.debug("Settings panel widgets loaded from settings manager")

    def _load_bindings_table(self) -> None:
        """Populates the bindings table from the current configuration.

        Rows are created dynamically to match whatever entries are
        present under the ``bindings`` configuration section; no row
        is hardcoded.
        """
        bindings: dict[str, Any] = self.settings_manager.get("bindings", {}) or {}

        self._bindings_table.setRowCount(0)
        self._bindings_table.setRowCount(len(bindings))

        for row_index, (fsm_event, action) in enumerate(bindings.items()):
            event_item = QTableWidgetItem(str(fsm_event))
            event_item.setFlags(event_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            action_item = QTableWidgetItem("" if action is None else str(action))

            self._bindings_table.setItem(row_index, 0, event_item)
            self._bindings_table.setItem(row_index, 1, action_item)

    def _on_save_clicked(self) -> None:
        """Writes every widget's current value to the settings manager.

        All tracking, filter, and camera spinbox values, along with
        every row of the bindings table, are written via
        ``settings_manager.set`` before the configuration is persisted
        to disk with ``settings_manager.save``.
        """
        self.settings_manager.set(
            "tracking.sensitivity", self._sensitivity_spinbox.value()
        )
        self.settings_manager.set(
            "tracking.pinch_tolerance", self._pinch_tolerance_spinbox.value()
        )
        self.settings_manager.set(
            "filter.mincutoff", self._mincutoff_spinbox.value()
        )
        self.settings_manager.set("filter.beta", self._beta_spinbox.value())
        self.settings_manager.set(
            "filter.dcutoff", self._dcutoff_spinbox.value()
        )
        self.settings_manager.set(
            "camera.device_index", self._device_index_spinbox.value()
        )
        self.settings_manager.set(
            "camera.target_fps", self._target_fps_spinbox.value()
        )

        for row_index in range(self._bindings_table.rowCount()):
            event_item = self._bindings_table.item(row_index, 0)
            action_item = self._bindings_table.item(row_index, 1)
            if event_item is None or action_item is None:
                continue
            fsm_event = event_item.text()
            action = action_item.text()
            self.settings_manager.set(f"bindings.{fsm_event}", action)

        self.settings_manager.save()
        logger.info("Configuration saved via SettingsPanel")

    def _on_restore_clicked(self) -> None:
        """Reloads the last-saved configuration from disk and refreshes widgets.

        This discards any unsaved edits in the panel by reloading the
        settings manager's cache from the on-disk configuration file,
        then re-populating every widget from the reloaded values.
        """
        self.settings_manager.reload()
        self._load_values()
        logger.info("Configuration restored via SettingsPanel")
