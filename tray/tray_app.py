"""System tray integration for the AirSign Developer B track.

This module implements :class:`TrayApplication`, a thin UI helper that
manages the application's system tray icon and context menu. It wraps
an already-constructed ``QApplication`` and main window, wiring the
tray menu's actions to window show/hide/restore behavior and the
application's quit action. It contains no gesture, threading, PDF
export, or configuration logic of any kind.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon, QWidget

if TYPE_CHECKING:
    from ui.main_window import MainWindow

logger = logging.getLogger(__name__)


class TrayApplication:
    """Manages the system tray icon, menu, and window show/hide behavior.

    This class does not construct a ``QApplication`` or a
    ``MainWindow``; both are supplied by the caller and merely
    referenced here. If the current platform has no system tray
    support, the tray icon is simply never shown and every public
    method degrades to a safe no-op (or, for :meth:`show_window`,
    :meth:`hide_window`, and :meth:`restore_window`, still operates
    directly on the window itself, since those do not require a tray
    icon to function).

    Attributes:
        main_window: The existing main window this tray controls.
    """

    def __init__(
        self,
        app: QApplication,
        main_window: "MainWindow | QWidget",
        icon: Optional[QIcon] = None,
        tooltip: str = "AirSign",
    ) -> None:
        """Initializes the tray icon and menu, if the platform supports it.

        Args:
            app: The already-constructed ``QApplication`` instance.
                Used only to connect the menu's quit action; no new
                ``QApplication`` is created.
            main_window: The already-constructed main window instance
                to show, hide, and restore.
            icon: Icon to display in the system tray. If ``None`` or
                null, a simple generated fallback icon is used instead
                so the tray still functions without a bundled icon
                file.
            tooltip: Tooltip text shown when hovering over the tray
                icon.
        """
        self._app = app
        self.main_window = main_window
        self._tray_icon: Optional[QSystemTrayIcon] = None
        self._menu: Optional[QMenu] = None
        self._show_action: Optional[QAction] = None
        self._hide_action: Optional[QAction] = None
        self._quit_action: Optional[QAction] = None

        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning(
                "System tray is not available on this platform; "
                "tray integration is disabled"
            )
            return

        resolved_icon = icon if icon is not None and not icon.isNull() else self._build_fallback_icon()

        try:
            self._tray_icon = QSystemTrayIcon(resolved_icon)
            self._tray_icon.setToolTip(tooltip)

            self._menu = self._build_menu()
            self._tray_icon.setContextMenu(self._menu)
            self._tray_icon.activated.connect(self._on_activated)

            self._tray_icon.show()
        except Exception:
            logger.exception("Failed to initialize system tray icon")
            self._tray_icon = None
            return

        logger.info("TrayApplication initialized")

    @staticmethod
    def _build_fallback_icon() -> QIcon:
        """Builds a simple generated icon for use when none is supplied.

        Returns:
            A small solid-colored circular ``QIcon`` that renders
            correctly in a tray even without a bundled icon file.
        """
        size = 32
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QColor("#00ccff"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(2, 2, size - 4, size - 4)
        finally:
            painter.end()

        return QIcon(pixmap)

    def _build_menu(self) -> QMenu:
        """Builds the tray context menu and its actions.

        Returns:
            The constructed ``QMenu``, with its "Show Window",
            "Hide Window", and "Quit" actions already connected.
        """
        menu = QMenu()

        self._show_action = QAction("Show Window", menu)
        self._hide_action = QAction("Hide Window", menu)
        self._quit_action = QAction("Quit", menu)

        self._show_action.triggered.connect(self.show_window)
        self._hide_action.triggered.connect(self.hide_window)
        self._quit_action.triggered.connect(self._app.quit)

        menu.addAction(self._show_action)
        menu.addAction(self._hide_action)
        menu.addSeparator()
        menu.addAction(self._quit_action)

        return menu

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Restores the window when the tray icon is activated.

        Args:
            reason: The activation reason reported by Qt. The window
                is restored for a normal click (``Trigger``) or a
                double-click (``DoubleClick``); other reasons (for
                example, a context-menu request) are ignored here
                since the context menu handles itself.
        """
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.restore_window()

    def is_available(self) -> bool:
        """Returns whether a system tray icon is active.

        Returns:
            True if the tray icon was successfully created and shown,
            False if system tray support is unavailable on this
            platform or initialization failed.
        """
        return self._tray_icon is not None

    def show_window(self) -> None:
        """Shows the main window, raising and activating it."""
        self.main_window.show()
        self.main_window.raise_()
        self.main_window.activateWindow()

    def hide_window(self) -> None:
        """Hides the main window without closing the application."""
        self.main_window.hide()

    def restore_window(self) -> None:
        """Restores the main window from a minimized or hidden state."""
        if self.main_window.isMinimized():
            self.main_window.showNormal()
        self.show_window()

    def notify(
        self,
        title: str,
        message: str,
        icon: QSystemTrayIcon.MessageIcon = QSystemTrayIcon.MessageIcon.Information,
        timeout_ms: int = 5000,
    ) -> None:
        """Displays a tray notification balloon, if the tray is available.

        The caller decides when a notification is warranted (for
        example, "AirSign started", "Export completed", or "Gesture
        profile loaded"); this method only renders whatever title and
        message it is given.

        Args:
            title: Notification title.
            message: Notification body text.
            icon: Icon style shown alongside the notification. Defaults
                to an informational icon.
            timeout_ms: Approximate duration, in milliseconds, the
                notification remains visible before auto-dismissing.
        """
        if self._tray_icon is None:
            logger.debug(
                "Tray notification suppressed (tray unavailable): %s - %s",
                title,
                message,
            )
            return

        try:
            self._tray_icon.showMessage(title, message, icon, timeout_ms)
        except Exception:
            logger.exception("Failed to display tray notification: %s", title)

    def shutdown(self) -> None:
        """Hides and releases the tray icon.

        Intended to be called once, before the application exits, so
        the tray icon does not linger as a stale entry after the
        process has ended. Safe to call even if the tray icon was
        never successfully created.
        """
        if self._tray_icon is None:
            return

        try:
            self._tray_icon.hide()
            self._tray_icon.setContextMenu(None)
            self._tray_icon.deleteLater()
        except Exception:
            logger.exception("Failed to hide tray icon during shutdown")
        finally:
            self._tray_icon = None

        logger.info("TrayApplication shut down")
