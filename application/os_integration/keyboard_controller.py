"""OS-level keyboard input injection for the AirSign Developer B track.

This module implements :class:`KeyboardInjector`, which translates
bound action identifier strings (resolved by
:mod:`application.config.bindings`) into actual operating-system
keyboard events via ``pynput``. Single keys and multi-key combinations
(e.g. Alt+Tab, Ctrl+C) are both supported. Unsupported or unrecognized
action identifiers are logged and ignored rather than raised.
"""

from __future__ import annotations

import logging
import threading
from typing import Union

from pynput.keyboard import Controller as PynputKeyboardController
from pynput.keyboard import Key, KeyCode

logger = logging.getLogger(__name__)

KeyType = Union[Key, KeyCode, str]


class KeyboardInjector:
    """Thread-safe OS keyboard input injector with action-to-combo mapping.

    Attributes:
        ACTION_MAP: Mapping of action identifier strings (as configured
            in ``config.json``'s ``bindings`` section) to an ordered
            tuple of keys that must be pressed together to perform that
            action. Keys are pressed in tuple order and released in
            reverse order.
    """

    ACTION_MAP: dict[str, tuple[KeyType, ...]] = {
        "prev_slide": (Key.page_up,),
        "next_slide": (Key.page_down,),
        "volume_up": (Key.media_volume_up,),
        "volume_down": (Key.media_volume_down,),
        "media_play_pause": (Key.media_play_pause,),
        "media_next": (Key.media_next,),
        "media_previous": (Key.media_previous,),
        "alt_tab": (Key.alt, Key.tab),
        "copy": (Key.ctrl, KeyCode.from_char("c")),
        "paste": (Key.ctrl, KeyCode.from_char("v")),
    }

    def __init__(self) -> None:
        """Initializes the keyboard injector."""
        self._lock = threading.RLock()
        self._controller = PynputKeyboardController()

        logger.info(
            "KeyboardInjector initialized with %d mapped action(s)",
            len(self.ACTION_MAP),
        )

    def perform_action(self, action: str) -> None:
        """Dispatches a named action as its mapped key combination.

        Presses every key in the mapped combination in order, then
        releases them in reverse order. If any key in the combination
        fails to press, all keys pressed so far are still released
        before the failure is logged, to avoid leaving a key stuck
        down.

        Args:
            action: The action identifier to dispatch, e.g.
                ``"volume_up"`` or ``"alt_tab"``.
        """
        with self._lock:
            combo = self.ACTION_MAP.get(action)

            if combo is None:
                logger.warning("Unknown action '%s', ignoring", action)
                return

            pressed: list[KeyType] = []
            try:
                for key in combo:
                    self._controller.press(key)
                    pressed.append(key)
                logger.info("Action dispatched: %s (combo=%s)", action, combo)
            except Exception:
                logger.exception("Failed to dispatch action '%s'", action)
            finally:
                for key in reversed(pressed):
                    try:
                        self._controller.release(key)
                    except Exception:
                        logger.exception(
                            "Failed to release key '%s' during action '%s'",
                            key,
                            action,
                        )

    def press_key(self, key: KeyType) -> None:
        """Presses and holds a single key.

        Args:
            key: The key to press, as a ``pynput.keyboard.Key``,
                ``KeyCode``, or single-character string.
        """
        with self._lock:
            try:
                self._controller.press(key)
                logger.debug("Key pressed: %s", key)
            except Exception:
                logger.exception("Failed to press key '%s'", key)

    def release_key(self, key: KeyType) -> None:
        """Releases a single previously held key.

        Args:
            key: The key to release, as a ``pynput.keyboard.Key``,
                ``KeyCode``, or single-character string.
        """
        with self._lock:
            try:
                self._controller.release(key)
                logger.debug("Key released: %s", key)
            except Exception:
                logger.exception("Failed to release key '%s'", key)

    def tap_key(self, key: KeyType) -> None:
        """Presses and immediately releases a single key.

        Args:
            key: The key to tap, as a ``pynput.keyboard.Key``,
                ``KeyCode``, or single-character string.
        """
        with self._lock:
            try:
                self._controller.press(key)
                self._controller.release(key)
                logger.debug("Key tapped: %s", key)
            except Exception:
                logger.exception("Failed to tap key '%s'", key)
