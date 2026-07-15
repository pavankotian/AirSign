"""Unit tests for application.config.bindings."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterator

import pytest

from application.config import bindings
from application.config.settings_manager import SettingsManager


@pytest.fixture(autouse=True)
def reset_bindings_cache() -> Iterator[None]:
    """Ensures the module-level bindings cache is empty before each test."""
    bindings._bindings_cache.clear()
    bindings._loaded = False
    yield
    bindings._bindings_cache.clear()
    bindings._loaded = False


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    """Writes a config.json containing both valid and invalid binding keys."""
    config: dict[str, Any] = {
        "bindings": {
            "PINCH_RELEASED": "left_click",
            "DOUBLE_PINCH": "right_click",
            "PINCH_HELD": "mouse_drag",
            "SWIPE_LEFT": "prev_slide",
            "THUMBS_UP": "media_play_pause",
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def test_valid_binding_lookup(config_path: Path) -> None:
    """get_binding() returns the bound action for a valid FSM event."""
    manager = SettingsManager(config_path=config_path)
    bindings.load_bindings(manager)

    assert bindings.get_binding("PINCH_RELEASED") == "left_click"
    assert bindings.get_binding("SWIPE_LEFT") == "prev_slide"


def test_invalid_fsm_key_ignored(
    config_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A non-FSM-event key such as THUMBS_UP is ignored with a warning."""
    manager = SettingsManager(config_path=config_path)

    with caplog.at_level("WARNING"):
        bindings.load_bindings(manager)

    assert bindings.get_binding("THUMBS_UP") is None
    assert any("THUMBS_UP" in record.message for record in caplog.records)


def test_unknown_lookup_returns_none(config_path: Path) -> None:
    """get_binding() returns None for an event with no configured binding."""
    manager = SettingsManager(config_path=config_path)
    bindings.load_bindings(manager)

    assert bindings.get_binding("FIST_ENTER") is None
    assert bindings.get_binding("NOT_A_REAL_EVENT") is None


def test_lookup_before_load_returns_none(config_path: Path) -> None:
    """get_binding() returns None if called before any load_bindings() call."""
    assert bindings.get_binding("PINCH_RELEASED") is None


def test_reload_updates_cache(config_path: Path) -> None:
    """reload_bindings() replaces the cache with freshly read configuration."""
    manager = SettingsManager(config_path=config_path)
    bindings.load_bindings(manager)
    assert bindings.get_binding("PINCH_RELEASED") == "left_click"

    manager.set("bindings.PINCH_RELEASED", "middle_click")
    bindings.reload_bindings(manager)

    assert bindings.get_binding("PINCH_RELEASED") == "middle_click"


def test_thread_safety(config_path: Path) -> None:
    """Concurrent load_bindings() and get_binding() calls do not raise."""
    manager = SettingsManager(config_path=config_path)
    bindings.load_bindings(manager)

    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def reader() -> None:
        try:
            for _ in range(100):
                bindings.get_binding("PINCH_RELEASED")
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            with errors_lock:
                errors.append(exc)

    def reloader() -> None:
        try:
            for _ in range(10):
                bindings.reload_bindings(manager)
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    threads += [threading.Thread(target=reloader) for _ in range(2)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
