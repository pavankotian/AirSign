"""Unit tests for application.os_integration.keyboard_controller.KeyboardInjector."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, call

import pytest

from application.os_integration.keyboard_controller import KeyboardInjector, Key


@pytest.fixture()
def injector() -> KeyboardInjector:
    """Returns a KeyboardInjector with a mocked pynput keyboard controller."""
    instance = KeyboardInjector()
    instance._controller = MagicMock()
    return instance


def test_action_map_contains_expected_actions() -> None:
    """ACTION_MAP defines entries for all documented example actions."""
    expected_actions = {
        "prev_slide",
        "next_slide",
        "volume_up",
        "volume_down",
        "media_play_pause",
        "alt_tab",
        "copy",
        "paste",
    }
    assert expected_actions.issubset(KeyboardInjector.ACTION_MAP.keys())


def test_single_key_action_presses_and_releases(injector: KeyboardInjector) -> None:
    """A single-key action presses then releases exactly that key."""
    injector.perform_action("volume_up")

    injector._controller.press.assert_called_once_with(Key.media_volume_up)
    injector._controller.release.assert_called_once_with(Key.media_volume_up)


def test_key_combination_alt_tab(injector: KeyboardInjector) -> None:
    """A multi-key combination presses in order and releases in reverse order."""
    injector.perform_action("alt_tab")

    assert injector._controller.press.call_args_list == [
        call(Key.alt),
        call(Key.tab),
    ]
    assert injector._controller.release.call_args_list == [
        call(Key.tab),
        call(Key.alt),
    ]


def test_key_combination_ctrl_c(injector: KeyboardInjector) -> None:
    """The Ctrl+C combination presses Ctrl then 'c', releasing in reverse."""
    injector.perform_action("copy")

    press_calls = injector._controller.press.call_args_list
    release_calls = injector._controller.release.call_args_list

    assert press_calls[0] == call(Key.ctrl)
    assert release_calls[-1] == call(Key.ctrl)
    assert len(press_calls) == 2
    assert len(release_calls) == 2


def test_unknown_action_logged_and_ignored(
    injector: KeyboardInjector, caplog: pytest.LogCaptureFixture
) -> None:
    """An unrecognized action identifier is logged and does not press any key."""
    with caplog.at_level("WARNING"):
        injector.perform_action("not_a_real_action")

    injector._controller.press.assert_not_called()
    injector._controller.release.assert_not_called()
    assert any(
        "not_a_real_action" in record.message for record in caplog.records
    )


def test_press_key_release_key(injector: KeyboardInjector) -> None:
    """press_key() and release_key() delegate directly to the controller."""
    injector.press_key(Key.shift)
    injector._controller.press.assert_called_once_with(Key.shift)

    injector.release_key(Key.shift)
    injector._controller.release.assert_called_once_with(Key.shift)


def test_tap_key(injector: KeyboardInjector) -> None:
    """tap_key() presses and releases the same key once each."""
    injector.tap_key(Key.esc)
    injector._controller.press.assert_called_once_with(Key.esc)
    injector._controller.release.assert_called_once_with(Key.esc)


def test_mocked_pynput_press_failure_still_releases_pressed_keys(
    injector: KeyboardInjector,
) -> None:
    """If a key press fails mid-combination, already-pressed keys are released."""
    injector._controller.press.side_effect = [None, OSError("simulated failure")]

    injector.perform_action("alt_tab")

    injector._controller.release.assert_called_once_with(Key.alt)


def test_press_key_failure_does_not_raise(injector: KeyboardInjector) -> None:
    """An OS-level failure from pynput is caught and never propagated."""
    injector._controller.press.side_effect = OSError("simulated failure")
    injector.press_key(Key.space)


def test_thread_safety(injector: KeyboardInjector) -> None:
    """Concurrent perform_action() calls across multiple threads do not raise."""
    errors: list[BaseException] = []
    errors_lock = threading.Lock()
    actions = ["volume_up", "volume_down", "alt_tab", "copy", "unknown_action"]

    def worker(action: str) -> None:
        try:
            for _ in range(50):
                injector.perform_action(action)
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            with errors_lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(action,)) for action in actions
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
