"""Unit tests for application.os_integration.mouse_controller.MouseInjector."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from application.os_integration import mouse_controller as mouse_controller_module
from application.os_integration.mouse_controller import MouseInjector


@pytest.fixture()
def mocked_pyautogui(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replaces the pyautogui functions used by MouseInjector with mocks."""
    mock = MagicMock()
    monkeypatch.setattr(mouse_controller_module.pyautogui, "moveTo", mock.moveTo)
    monkeypatch.setattr(mouse_controller_module.pyautogui, "click", mock.click)
    monkeypatch.setattr(
        mouse_controller_module.pyautogui, "doubleClick", mock.doubleClick
    )
    return mock


@pytest.fixture()
def injector(mocked_pyautogui: MagicMock) -> MouseInjector:
    """Returns a MouseInjector with a mocked pynput mouse controller."""
    instance = MouseInjector(click_cooldown_ms=50, scroll_cooldown_ms=20)
    instance._pynput_mouse = MagicMock()
    return instance


def test_movement(injector: MouseInjector, mocked_pyautogui: MagicMock) -> None:
    """move() delegates to pyautogui.moveTo with integer coordinates."""
    injector.move(123.7, 456.2)
    mocked_pyautogui.moveTo.assert_called_once_with(123, 456)


def test_movement_never_cooldown_gated(
    injector: MouseInjector, mocked_pyautogui: MagicMock
) -> None:
    """move() is never suppressed, unlike click-family actions."""
    injector.move(1, 1)
    injector.move(2, 2)
    injector.move(3, 3)
    assert mocked_pyautogui.moveTo.call_count == 3


def test_click_cooldown_suppresses_rapid_clicks(
    injector: MouseInjector, mocked_pyautogui: MagicMock
) -> None:
    """left_click() is suppressed if called again before the cooldown elapses."""
    injector.left_click()
    injector.left_click()
    assert mocked_pyautogui.click.call_count == 1

    time.sleep(0.06)
    injector.left_click()
    assert mocked_pyautogui.click.call_count == 2


def test_right_click_shares_click_cooldown(
    injector: MouseInjector, mocked_pyautogui: MagicMock
) -> None:
    """right_click() is gated by the same cooldown as left_click()."""
    injector.left_click()
    injector.right_click()
    assert mocked_pyautogui.click.call_count == 1


def test_double_click(injector: MouseInjector, mocked_pyautogui: MagicMock) -> None:
    """double_click() delegates to pyautogui.doubleClick, subject to cooldown."""
    injector.double_click()
    mocked_pyautogui.doubleClick.assert_called_once()

    injector.double_click()
    assert mocked_pyautogui.doubleClick.call_count == 1


def test_press_and_release(injector: MouseInjector) -> None:
    """press() and release() toggle is_pressed() and call the pynput backend."""
    assert injector.is_pressed() is False

    injector.press()
    assert injector.is_pressed() is True
    injector._pynput_mouse.press.assert_called_once()

    injector.release()
    assert injector.is_pressed() is False
    injector._pynput_mouse.release.assert_called_once()


def test_duplicate_press_prevention(injector: MouseInjector) -> None:
    """Calling press() twice without a release() only presses once."""
    injector.press()
    injector.press()
    injector.press()

    assert injector._pynput_mouse.press.call_count == 1
    assert injector.is_pressed() is True


def test_release_without_press_is_noop(injector: MouseInjector) -> None:
    """Calling release() without a prior press() does not call the backend."""
    injector.release()
    injector._pynput_mouse.release.assert_not_called()
    assert injector.is_pressed() is False


def test_press_release_press_cycle(injector: MouseInjector) -> None:
    """A press/release/press cycle allows a second genuine press."""
    injector.press()
    injector.release()
    injector.press()

    assert injector._pynput_mouse.press.call_count == 2
    assert injector.is_pressed() is True


def test_scroll(injector: MouseInjector) -> None:
    """scroll() delegates to the pynput backend, subject to the scroll cooldown."""
    injector.scroll(5)
    injector._pynput_mouse.scroll.assert_called_once_with(0, 5)

    injector.scroll(-3)
    assert injector._pynput_mouse.scroll.call_count == 1

    time.sleep(0.03)
    injector.scroll(-3)
    assert injector._pynput_mouse.scroll.call_count == 2


def test_mocked_pyautogui_failure_does_not_raise(
    injector: MouseInjector, mocked_pyautogui: MagicMock
) -> None:
    """An OS-level failure from pyautogui is caught and never propagated."""
    mocked_pyautogui.moveTo.side_effect = OSError("simulated OS failure")
    injector.move(10, 10)

    mocked_pyautogui.click.side_effect = OSError("simulated OS failure")
    injector.left_click()


def test_pynput_failure_does_not_raise(injector: MouseInjector) -> None:
    """An OS-level failure from pynput is caught and never propagated."""
    injector._pynput_mouse.press.side_effect = OSError("simulated OS failure")
    injector.press()
    assert injector.is_pressed() is False


def test_thread_safety(
    injector: MouseInjector, mocked_pyautogui: MagicMock
) -> None:
    """Concurrent calls across all public methods do not raise."""
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker() -> None:
        try:
            for _ in range(50):
                injector.move(1, 1)
                injector.left_click()
                injector.press()
                injector.release()
                injector.scroll(1)
                assert isinstance(injector.is_pressed(), bool)
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
