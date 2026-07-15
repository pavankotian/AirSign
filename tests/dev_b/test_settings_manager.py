"""Unit tests for application.config.settings_manager.SettingsManager."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from application.config.settings_manager import SettingsManager


@pytest.fixture()
def sample_config() -> dict[str, Any]:
    """Returns a representative nested configuration dictionary."""
    return {
        "filter": {
            "mincutoff": 1.0,
            "beta": 0.007,
            "dcutoff": 1.0,
        },
        "tracking": {
            "sensitivity": 0.85,
            "pinch_tolerance": 0.15,
            "active_zone": {
                "x1": 0.05,
                "y1": 0.05,
                "x2": 0.95,
                "y2": 0.95,
            },
        },
        "bindings": {
            "PINCH_RELEASED": "left_click",
        },
    }


@pytest.fixture()
def config_path(tmp_path: Path, sample_config: dict[str, Any]) -> Path:
    """Writes ``sample_config`` to a temporary config.json and returns its path."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(sample_config, indent=2), encoding="utf-8")
    return path


def test_load_config(config_path: Path) -> None:
    """SettingsManager loads the configuration file on initialization."""
    manager = SettingsManager(config_path=config_path)
    assert manager.as_dict()["filter"]["beta"] == pytest.approx(0.007)


def test_dot_lookup(config_path: Path) -> None:
    """get() resolves a single-level dot-notation path."""
    manager = SettingsManager(config_path=config_path)
    assert manager.get("bindings") == {"PINCH_RELEASED": "left_click"}


def test_nested_lookup(config_path: Path) -> None:
    """get() resolves a multi-level dot-notation path."""
    manager = SettingsManager(config_path=config_path)
    assert manager.get("filter.beta") == pytest.approx(0.007)
    assert manager.get("tracking.active_zone.x1") == pytest.approx(0.05)


def test_missing_key_default(config_path: Path) -> None:
    """get() returns the provided default when the path does not resolve."""
    manager = SettingsManager(config_path=config_path)
    assert manager.get("filter.nonexistent", default="fallback") == "fallback"
    assert manager.get("nonexistent.section", default=None) is None
    assert manager.get("filter.beta.too_deep", default=-1) == -1


def test_get_empty_path_raises(config_path: Path) -> None:
    """get() raises ValueError for an empty path string."""
    manager = SettingsManager(config_path=config_path)
    with pytest.raises(ValueError):
        manager.get("")


def test_set_updates_cache(config_path: Path) -> None:
    """set() updates the in-memory cache at a dot-notation path."""
    manager = SettingsManager(config_path=config_path)
    manager.set("filter.beta", 0.5)
    assert manager.get("filter.beta") == pytest.approx(0.5)


def test_set_creates_intermediate_objects(config_path: Path) -> None:
    """set() creates intermediate dictionaries that do not yet exist."""
    manager = SettingsManager(config_path=config_path)
    manager.set("ui.theme", "dark")
    assert manager.get("ui.theme") == "dark"


def test_set_on_non_dict_segment_raises(config_path: Path) -> None:
    """set() raises ValueError when a path segment is not traversable."""
    manager = SettingsManager(config_path=config_path)
    with pytest.raises(ValueError):
        manager.set("filter.beta.nested", 1)


def test_set_empty_path_raises(config_path: Path) -> None:
    """set() raises ValueError for an empty path string."""
    manager = SettingsManager(config_path=config_path)
    with pytest.raises(ValueError):
        manager.set("", 1)


def test_save_writes_to_disk(config_path: Path) -> None:
    """save() persists in-memory changes to the configuration file."""
    manager = SettingsManager(config_path=config_path)
    manager.set("filter.beta", 0.9)
    manager.save()

    on_disk = json.loads(config_path.read_text(encoding="utf-8"))
    assert on_disk["filter"]["beta"] == pytest.approx(0.9)


def test_atomic_save_no_temp_files_remain(config_path: Path) -> None:
    """save() leaves no stray temporary files in the config directory."""
    manager = SettingsManager(config_path=config_path)
    manager.set("filter.beta", 0.3)
    manager.save()

    directory_contents = list(config_path.parent.iterdir())
    assert config_path in directory_contents
    assert all(
        entry == config_path
        or not entry.name.startswith(f".{config_path.name}.")
        for entry in directory_contents
    )


def test_reload_discards_unsaved_changes(config_path: Path) -> None:
    """reload() replaces the in-memory cache with the on-disk contents."""
    manager = SettingsManager(config_path=config_path)
    manager.set("filter.beta", 0.999)
    assert manager.get("filter.beta") == pytest.approx(0.999)

    manager.reload()
    assert manager.get("filter.beta") == pytest.approx(0.007)


def test_reload_picks_up_external_changes(config_path: Path) -> None:
    """reload() reflects changes written to disk by another process."""
    manager = SettingsManager(config_path=config_path)

    external_config = json.loads(config_path.read_text(encoding="utf-8"))
    external_config["filter"]["beta"] = 0.42
    config_path.write_text(json.dumps(external_config), encoding="utf-8")

    manager.reload()
    assert manager.get("filter.beta") == pytest.approx(0.42)


def test_corrupt_json_raises_value_error(tmp_path: Path) -> None:
    """Initialization raises ValueError when the file is not valid JSON."""
    bad_path = tmp_path / "config.json"
    bad_path.write_text("{ this is not valid json", encoding="utf-8")

    with pytest.raises(ValueError):
        SettingsManager(config_path=bad_path)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """Initialization raises FileNotFoundError when the path does not exist."""
    missing_path = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):
        SettingsManager(config_path=missing_path)


def test_concurrent_reads(config_path: Path) -> None:
    """Multiple threads may call get() concurrently without error."""
    manager = SettingsManager(config_path=config_path)
    results: list[Any] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()

    def read_value() -> None:
        try:
            value = manager.get("filter.beta")
            with results_lock:
                results.append(value)
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            with results_lock:
                errors.append(exc)

    threads = [threading.Thread(target=read_value) for _ in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(results) == 32
    assert all(value == pytest.approx(0.007) for value in results)
