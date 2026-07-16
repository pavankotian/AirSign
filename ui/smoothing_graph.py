"""Live cursor smoothing comparison graph for the AirSign Developer B UI.

This module implements :class:`SmoothingGraph`, a ``QWidget`` that
displays a rolling, real-time comparison of raw versus 1€-filtered
cursor coordinates, split into a top plot for the X axis and a bottom
plot for the Y axis. It contains no gesture, threading, or business
logic; it only maintains bounded sample history and redraws existing
plot curves.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class SmoothingGraph(QWidget):
    """Displays rolling raw-vs-filtered cursor smoothing plots.

    Two stacked ``pyqtgraph.PlotWidget`` instances are created once at
    construction time and never recreated: the top plot compares raw
    and filtered X coordinates, and the bottom plot compares raw and
    filtered Y coordinates. Each update reuses the existing plot
    curves via ``setData`` rather than creating new curves or plot
    widgets, keeping redraws lightweight for smooth real-time display.

    Attributes:
        window_size: Maximum number of samples retained per history,
            controlling both the rolling window length and how far
            back each plot scrolls.
        raw_x_history: Rolling history of raw cursor X coordinates.
        raw_y_history: Rolling history of raw cursor Y coordinates.
        filtered_x_history: Rolling history of 1€-filtered cursor X
            coordinates.
        filtered_y_history: Rolling history of 1€-filtered cursor Y
            coordinates.
    """

    def __init__(
        self, window_size: int = 150, parent: Optional[QWidget] = None
    ) -> None:
        """Initializes the smoothing graph.

        Args:
            window_size: Maximum number of samples retained per
                history and shown on screen at once. Defaults to 150.
            parent: Optional parent widget.
        """
        super().__init__(parent)

        self.window_size = window_size

        self._sample_index: int = 0
        self._sample_history: deque[int] = deque(maxlen=window_size)
        self.raw_x_history: deque[float] = deque(maxlen=window_size)
        self.raw_y_history: deque[float] = deque(maxlen=window_size)
        self.filtered_x_history: deque[float] = deque(maxlen=window_size)
        self.filtered_y_history: deque[float] = deque(maxlen=window_size)

        self._x_plot_widget = pg.PlotWidget()
        self._y_plot_widget = pg.PlotWidget()

        raw_x_curve, filtered_x_curve = self._configure_plot(
            self._x_plot_widget,
            title="Cursor X: Raw vs. Filtered",
            y_label="X Position",
            raw_name="Raw X",
            filtered_name="Filtered X",
        )
        self._raw_x_curve = raw_x_curve
        self._filtered_x_curve = filtered_x_curve

        raw_y_curve, filtered_y_curve = self._configure_plot(
            self._y_plot_widget,
            title="Cursor Y: Raw vs. Filtered",
            y_label="Y Position",
            raw_name="Raw Y",
            filtered_name="Filtered Y",
        )
        self._raw_y_curve = raw_y_curve
        self._filtered_y_curve = filtered_y_curve

        layout = QVBoxLayout(self)
        layout.addWidget(self._x_plot_widget)
        layout.addWidget(self._y_plot_widget)
        self.setLayout(layout)

        logger.info(
            "SmoothingGraph initialized (window_size=%d)", self.window_size
        )

    @staticmethod
    def _configure_plot(
        plot_widget: "pg.PlotWidget",
        title: str,
        y_label: str,
        raw_name: str,
        filtered_name: str,
    ) -> tuple["pg.PlotDataItem", "pg.PlotDataItem"]:
        """Configures a plot widget's axes, grid, legend, and curves.

        Args:
            plot_widget: The plot widget to configure.
            title: Title displayed above the plot.
            y_label: Label for the vertical axis.
            raw_name: Legend label for the raw-value curve.
            filtered_name: Legend label for the filtered-value curve.

        Returns:
            A ``(raw_curve, filtered_curve)`` tuple of the created
            ``PlotDataItem`` curves, to be updated in place on every
            frame via ``setData``.
        """
        plot_item = plot_widget.getPlotItem()
        plot_item.setTitle(title)
        plot_item.setLabel("left", y_label)
        plot_item.setLabel("bottom", "Sample")
        plot_item.showGrid(x=True, y=True, alpha=0.3)
        plot_item.addLegend()

        raw_curve = plot_widget.plot(
            pen=pg.mkPen(color=(150, 150, 150), width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
            name=raw_name,
        )
        filtered_curve = plot_widget.plot(
            pen=pg.mkPen(color=(0, 204, 255), width=2),
            name=filtered_name,
        )

        return raw_curve, filtered_curve

    def update_graph(
        self,
        raw_x: Optional[float],
        raw_y: Optional[float],
        filtered_x: Optional[float],
        filtered_y: Optional[float],
    ) -> None:
        """Appends one sample and redraws all four curves.

        Intended to be connected directly to
        ``AppSignals.graph_update``. If any of the four values is
        ``None``, the sample is ignored and no curves are redrawn.

        Args:
            raw_x: Raw (unfiltered) cursor X coordinate.
            raw_y: Raw (unfiltered) cursor Y coordinate.
            filtered_x: 1€-filtered cursor X coordinate.
            filtered_y: 1€-filtered cursor Y coordinate.
        """
        if raw_x is None or raw_y is None or filtered_x is None or filtered_y is None:
            logger.debug("Ignoring graph update containing a None value")
            return

        self._sample_index += 1
        self._sample_history.append(self._sample_index)
        self.raw_x_history.append(raw_x)
        self.raw_y_history.append(raw_y)
        self.filtered_x_history.append(filtered_x)
        self.filtered_y_history.append(filtered_y)

        self._redraw_curves()

    def _redraw_curves(self) -> None:
        """Updates all four existing curves in place with current history.

        No new ``PlotDataItem`` or ``PlotWidget`` instances are
        created; only ``setData`` is called on the curves created in
        :meth:`__init__`. Because each history is a bounded deque, the
        x-axis sample range shrinks and grows with the window,
        producing a continuously auto-scrolling display.
        """
        sample_values = list(self._sample_history)

        self._raw_x_curve.setData(sample_values, list(self.raw_x_history))
        self._filtered_x_curve.setData(sample_values, list(self.filtered_x_history))
        self._raw_y_curve.setData(sample_values, list(self.raw_y_history))
        self._filtered_y_curve.setData(sample_values, list(self.filtered_y_history))
