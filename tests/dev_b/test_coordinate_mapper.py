"""Unit tests for application.os_integration.coordinate_mapper.CoordinateMapper."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from application.config.settings_manager import SettingsManager
from application.os_integration import coordinate_mapper as coordinate_mapper_module
from application.os_integration.coordinate_mapper import CoordinateMapper


@dataclass
class _FakeMonitor:
    """Minimal stand-in for screeninfo.Monitor used to control test resolution."""

    width: int
    height: int
    x: int = 0
    y: int = 0
    name: str = "fake-monitor"
    is_primary: bool = True


@pytest.fixture()
def patch_monitor(monkeypatch: pytest.MonkeyPatch):
    """Returns a helper to patch screeninfo.get_monitors() with fixed monitors."""

    def _patch(monitors: list[_FakeMonitor]) -> None:
        monkeypatch.setattr(
            coordinate_mapper_module.screeninfo,
            "get_monitors",
            lambda: monitors,
        )

    return _patch


@pytest.fixture()
def default_config() -> dict[str, Any]:
    """Returns a configuration dict with a non-trivial active zone."""
    return {
        "tracking": {
            "sensitivity": 0.85,
            "pinch_tolerance": 0.15,
            "active_zone": {
                "x1": 0.1,
                "y1": 0.2,
                "x2": 0.9,
                "y2": 0.8,
            },
        }
    }


@pytest.fixture()
def settings_manager(tmp_path: Path, default_config: dict[str, Any]) -> SettingsManager:
    """Returns a real SettingsManager backed by a temporary config file."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(default_config, indent=2), encoding="utf-8")
    return SettingsManager(config_path=path)


@pytest.fixture()
def mapper(
    settings_manager: SettingsManager,
    patch_monitor,
) -> CoordinateMapper:
    """Returns a CoordinateMapper with a fixed 1920x1080 monitor."""
    patch_monitor([_FakeMonitor(width=1920, height=1080, is_primary=True)])
    return CoordinateMapper(settings_manager)


def test_screen_size(mapper: CoordinateMapper) -> None:
    """get_screen_size() returns the resolution reported by screeninfo."""
    assert mapper.get_screen_size() == (1920, 1080)


def test_screen_size_fallback_on_no_monitors(
    settings_manager: SettingsManager, patch_monitor
) -> None:
    """Screen size falls back to a default when no monitors are detected."""
    patch_monitor([])
    fallback_mapper = CoordinateMapper(settings_manager)
    assert fallback_mapper.get_screen_size() == (1920, 1080)


def test_screen_size_fallback_on_exception(
    settings_manager: SettingsManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Screen size falls back to a default when screeninfo raises."""

    def _raise() -> list[_FakeMonitor]:
        raise RuntimeError("no display server")

    monkeypatch.setattr(coordinate_mapper_module.screeninfo, "get_monitors", _raise)
    fallback_mapper = CoordinateMapper(settings_manager)
    assert fallback_mapper.get_screen_size() == (1920, 1080)


def test_normalization_center(mapper: CoordinateMapper) -> None:
    """normalize() maps the zone center to (0.5, 0.5)."""
    norm_x, norm_y = mapper.normalize(0.5, 0.5)
    assert norm_x == pytest.approx(0.5)
    assert norm_y == pytest.approx(0.5)


def test_normalization_zone_corners(mapper: CoordinateMapper) -> None:
    """normalize() maps the active zone's own corners to (0.0-1.0) bounds."""
    top_left = mapper.normalize(0.1, 0.2)
    bottom_right = mapper.normalize(0.9, 0.8)

    assert top_left[0] == pytest.approx(0.0)
    assert top_left[1] == pytest.approx(0.0)
    assert bottom_right[0] == pytest.approx(1.0)
    assert bottom_right[1] == pytest.approx(1.0)


def test_active_zone_scaling(mapper: CoordinateMapper) -> None:
    """to_screen() scales coordinates according to the configured active zone."""
    screen_x, screen_y = mapper.to_screen(0.5, 0.5)
    assert screen_x == pytest.approx(960, abs=1)
    assert screen_y == pytest.approx(540, abs=1)


def test_update_active_zone_changes_mapping(mapper: CoordinateMapper) -> None:
    """update_active_zone() replaces the zone used by normalize() and to_screen()."""
    mapper.update_active_zone({"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0})

    norm_x, norm_y = mapper.normalize(0.25, 0.75)
    assert norm_x == pytest.approx(0.25)
    assert norm_y == pytest.approx(0.75)

    assert mapper.settings_manager.get("tracking.active_zone") == {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 1.0,
        "y2": 1.0,
    }


def test_clamping_within_bounds(mapper: CoordinateMapper) -> None:
    """clamp() leaves in-bounds coordinates unchanged."""
    assert mapper.clamp(500.0, 300.0) == (500, 300)


def test_clamping_out_of_bounds(mapper: CoordinateMapper) -> None:
    """clamp() clips coordinates outside the screen to the screen edges."""
    assert mapper.clamp(-100.0, -50.0) == (0, 0)
    assert mapper.clamp(5000.0, 5000.0) == (1919, 1079)


def test_malformed_zone_missing_key(settings_manager: SettingsManager) -> None:
    """A zone missing a required key raises ValueError at construction time."""
    settings_manager.set("tracking.active_zone", {"x1": 0.0, "y1": 0.0, "x2": 1.0})

    with pytest.raises(ValueError):
        CoordinateMapper(settings_manager)


def test_malformed_zone_inverted_bounds(mapper: CoordinateMapper) -> None:
    """update_active_zone() rejects a zone where x1 >= x2."""
    with pytest.raises(ValueError):
        mapper.update_active_zone({"x1": 0.9, "y1": 0.1, "x2": 0.1, "y2": 0.9})


def test_malformed_zone_out_of_unit_range(mapper: CoordinateMapper) -> None:
    """update_active_zone() rejects a zone with values outside [0.0, 1.0]."""
    with pytest.raises(ValueError):
        mapper.update_active_zone({"x1": -0.5, "y1": 0.0, "x2": 1.0, "y2": 1.0})


def test_malformed_zone_non_numeric(mapper: CoordinateMapper) -> None:
    """update_active_zone() rejects a zone with a non-numeric value."""
    with pytest.raises(ValueError):
        mapper.update_active_zone({"x1": "left", "y1": 0.0, "x2": 1.0, "y2": 1.0})


def test_malformed_zone_not_a_dict(settings_manager: SettingsManager) -> None:
    """A non-object active zone value raises ValueError at construction time."""
    settings_manager.set("tracking.active_zone", "not-a-dict")

    with pytest.raises(ValueError):
        CoordinateMapper(settings_manager)


def test_edge_coordinates_clamped(mapper: CoordinateMapper) -> None:
    """to_screen() clamps camera-space coordinates outside [0.0, 1.0]."""
    screen_x, screen_y = mapper.to_screen(-5.0, -5.0)
    assert screen_x == 0
    assert screen_y == 0

    screen_x, screen_y = mapper.to_screen(5.0, 5.0)
    assert screen_x == 1919
    assert screen_y == 1079


def test_thread_safety(mapper: CoordinateMapper) -> None:
    """Concurrent to_screen() and update_active_zone() calls do not raise."""
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def reader() -> None:
        try:
            for _ in range(200):
                x, y = mapper.to_screen(0.5, 0.5)
                assert 0 <= x < 1920
                assert 0 <= y < 1080
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            with errors_lock:
                errors.append(exc)

    def writer() -> None:
        try:
            for _ in range(50):
                mapper.update_active_zone(
                    {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}
                )
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(6)]
    threads += [threading.Thread(target=writer) for _ in range(2)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
