"""OS-level mouse input injection for the AirSign Developer B track.

This module implements :class:`MouseInjector`, which translates FSM
gesture events into actual operating-system mouse movement, clicks,
button holds, and scroll events. Cursor movement and simple clicks are
dispatched via ``pyautogui``; button hold state and scrolling are
dispatched via ``pynput`` for finer-grained control. All OS-level
failures are caught and logged rather than propagated, since a failed
injection should never crash the application's action-dispatch thread.
"""

from __future__ import annotations

import logging
import threading
import time

import pyautogui
from pynput.mouse import Button, Controller as PynputMouseController

from shared.constants import COOLDOWN_CLICK_MS, COOLDOWN_SCROLL_MS

logger = logging.getLogger(__name__)

pyautogui.FAILSAFE = False


class MouseInjector:
    """Thread-safe OS mouse input injector with cooldown gating.

    Attributes:
        click_cooldown_ms: Minimum milliseconds between consecutive
            click-family actions (left click, right click, double
            click).
        scroll_cooldown_ms: Minimum milliseconds between consecutive
            scroll injections.
    """

    def __init__(
        self,
        click_cooldown_ms: float = COOLDOWN_CLICK_MS,
        scroll_cooldown_ms: float = COOLDOWN_SCROLL_MS,
    ) -> None:
        """Initializes the mouse injector.

        Args:
            click_cooldown_ms: Minimum milliseconds required between
                consecutive click-family actions. Defaults to
                :data:`shared.constants.COOLDOWN_CLICK_MS`.
            scroll_cooldown_ms: Minimum milliseconds required between
                consecutive scroll injections. Defaults to
                :data:`shared.constants.COOLDOWN_SCROLL_MS`.
        """
        self._lock = threading.RLock()
        self.click_cooldown_ms = click_cooldown_ms
        self.scroll_cooldown_ms = scroll_cooldown_ms

        self._pynput_mouse = PynputMouseController()
        self._is_pressed = False
        self._last_click_time: float = float("-inf")
        self._last_scroll_time: float = float("-inf")

        logger.info(
            "MouseInjector initialized (click_cooldown_ms=%s, "
            "scroll_cooldown_ms=%s)",
            self.click_cooldown_ms,
            self.scroll_cooldown_ms,
        )

    @staticmethod
    def _cooldown_elapsed(last_time: float, cooldown_ms: float) -> bool:
        """Returns True if cooldown_ms has elapsed since last_time.

        Args:
            last_time: Monotonic timestamp of the previous action.
            cooldown_ms: Required cooldown duration in milliseconds.

        Returns:
            True if enough time has passed to permit another action.
        """
        return (time.monotonic() - last_time) * 1000.0 >= cooldown_ms

    def move(self, x: float, y: float) -> None:
        """Moves the OS cursor to an absolute screen position.

        Movement is never cooldown-gated, since continuous cursor
        tracking requires updating on every frame.

        Args:
            x: Target screen x-coordinate in pixels.
            y: Target screen y-coordinate in pixels.
        """
        try:
            pyautogui.moveTo(int(x), int(y))
        except Exception:
            logger.exception("Failed to move cursor to (%s, %s)", x, y)

    def left_click(self) -> None:
        """Performs a left mouse click, subject to the click cooldown."""
        with self._lock:
            if not self._cooldown_elapsed(
                self._last_click_time, self.click_cooldown_ms
            ):
                logger.debug("left_click() suppressed by cooldown")
                return
            self._last_click_time = time.monotonic()

        try:
            pyautogui.click(button="left")
            logger.info("Left click dispatched")
        except Exception:
            logger.exception("Failed to dispatch left click")

    def right_click(self) -> None:
        """Performs a right mouse click, subject to the click cooldown."""
        with self._lock:
            if not self._cooldown_elapsed(
                self._last_click_time, self.click_cooldown_ms
            ):
                logger.debug("right_click() suppressed by cooldown")
                return
            self._last_click_time = time.monotonic()

        try:
            pyautogui.click(button="right")
            logger.info("Right click dispatched")
        except Exception:
            logger.exception("Failed to dispatch right click")

    def double_click(self) -> None:
        """Performs a double left-click, subject to the click cooldown."""
        with self._lock:
            if not self._cooldown_elapsed(
                self._last_click_time, self.click_cooldown_ms
            ):
                logger.debug("double_click() suppressed by cooldown")
                return
            self._last_click_time = time.monotonic()

        try:
            pyautogui.doubleClick()
            logger.info("Double click dispatched")
        except Exception:
            logger.exception("Failed to dispatch double click")

    def press(self) -> None:
        """Presses and holds the left mouse button.

        If the button is already held down, this call is a no-op, to
        prevent duplicate press events from being sent to the OS.
        """
        with self._lock:
            if self._is_pressed:
                logger.debug("press() ignored: left mouse button already held")
                return
            try:
                self._pynput_mouse.press(Button.left)
                self._is_pressed = True
                logger.info("Left mouse button pressed (hold)")
            except Exception:
                logger.exception("Failed to press left mouse button")

    def release(self) -> None:
        """Releases the left mouse button if currently held.

        If the button is not currently held, this call is a no-op.
        """
        with self._lock:
            if not self._is_pressed:
                logger.debug("release() ignored: left mouse button not held")
                return
            try:
                self._pynput_mouse.release(Button.left)
                logger.info("Left mouse button released")
            except Exception:
                logger.exception("Failed to release left mouse button")
            finally:
                self._is_pressed = False

    def scroll(self, amount: int) -> None:
        """Injects a vertical scroll event, subject to the scroll cooldown.

        Args:
            amount: Number of scroll units. Positive values scroll up,
                negative values scroll down.
        """
        with self._lock:
            if not self._cooldown_elapsed(
                self._last_scroll_time, self.scroll_cooldown_ms
            ):
                logger.debug("scroll() suppressed by cooldown")
                return
            self._last_scroll_time = time.monotonic()

        try:
            self._pynput_mouse.scroll(0, amount)
            logger.debug("Scroll dispatched: amount=%s", amount)
        except Exception:
            logger.exception("Failed to dispatch scroll event")

    def is_pressed(self) -> bool:
        """Returns whether the left mouse button is currently held down.

        Returns:
            True if :meth:`press` has been called without a matching
            :meth:`release`, False otherwise.
        """
        with self._lock:
            return self._is_pressed
