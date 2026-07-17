"""Annotated camera frame display widget for the AirSign Developer B UI.

This module implements :class:`VideoPanel`, a ``QLabel`` subclass that
displays the annotated BGR frames delivered by
``AppSignals.frame_ready``. It performs only the display-side BGR to
RGB channel conversion and Qt image/pixmap construction; it never
redraws landmarks, never invokes MediaPipe, and contains no gesture or
threading logic.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QResizeEvent
from PyQt6.QtWidgets import QLabel, QWidget

logger = logging.getLogger(__name__)


class VideoPanel(QLabel):
    """Displays annotated camera frames with aspect-ratio-preserving scaling.

    Frames are received as NumPy BGR arrays (the format produced by
    OpenCV capture and MediaPipe drawing utilities upstream), converted
    to RGB, wrapped in a ``QImage``, and rendered as a ``QPixmap``
    scaled to fit the current widget size while preserving the
    original aspect ratio. The most recently received full-resolution
    pixmap is retained so the displayed image can be rescaled cleanly
    whenever the widget is resized.

    Attributes:
        placeholder_text: The text shown in place of an image when no
            frame has been received, or the most recent frame was
            ``None``.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        placeholder_text: str = "No Signal",
    ) -> None:
        """Initializes the video panel.

        Args:
            parent: Optional parent widget.
            placeholder_text: Text displayed when there is no frame to
                show. Defaults to ``"No Signal"``.
        """
        super().__init__(parent)

        self.placeholder_text = placeholder_text
        self._current_pixmap: Optional[QPixmap] = None

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)
        self.setText(self.placeholder_text)

        logger.info("VideoPanel initialized")

    def update_frame(self, frame: Optional[np.ndarray]) -> None:
        """Displays a new annotated frame, or clears the display if None.

        Intended to be connected directly to
        ``AppSignals.frame_ready``.

        Args:
            frame: A BGR frame as a NumPy array of shape
                ``(height, width, 3)`` and dtype ``uint8``, or ``None``
                if no frame is currently available.
        """
        if frame is None:
            logger.debug("Received None frame; clearing video panel")
            self._current_pixmap = None
            self.setPixmap(QPixmap())
            self.setText(self.placeholder_text)
            return

        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            logger.warning(
                "Received malformed frame (type=%s, shape=%s); ignoring",
                type(frame).__name__,
                getattr(frame, "shape", None),
            )
            return

        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        height, width, channel_count = rgb_frame.shape
        bytes_per_line = channel_count * width

        qimage = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        self._current_pixmap = QPixmap.fromImage(qimage)
        self._render_scaled_pixmap()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Rescales the currently displayed pixmap on widget resize.

        Args:
            event: The Qt resize event.
        """
        super().resizeEvent(event)
        self._render_scaled_pixmap()

    def _render_scaled_pixmap(self) -> None:
        """Renders the stored full-resolution pixmap scaled to fit the widget.

        Scaling always preserves the original aspect ratio using
        ``Qt.AspectRatioMode.KeepAspectRatio``. If no pixmap is
        currently stored, this method does nothing.
        """
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return

        scaled_pixmap = self._current_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled_pixmap)
