"""Thread-safe configuration management for the AirSign Developer B track.

This module implements :class:`SettingsManager`, which loads a JSON
configuration file exactly once at initialization, caches it in memory,
and exposes dot-notation accessors alongside an atomic, crash-safe
persistence mechanism.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SettingsManager:
    """Thread-safe manager for AirSign's JSON configuration file.

    Loads a configuration file once at construction time, caches it in
    memory, and provides dot-notation get/set access plus an atomic
    save-to-disk operation. Every public method is protected by a single
    reentrant lock, so the manager is safe to share across threads (for
    example, the UI thread and ``ActionThread``).

    Attributes:
        config_path: Filesystem path to the JSON configuration file.
    """

    def __init__(self, config_path: str | os.PathLike[str] = "config.json") -> None:
        """Initializes the manager and loads the configuration file.

        Args:
            config_path: Path to the JSON configuration file to load.

        Raises:
            FileNotFoundError: If ``config_path`` does not exist.
            ValueError: If the file exists but does not contain a valid
                JSON object.
        """
        self._lock = threading.RLock()
        self.config_path: Path = Path(config_path)
        self._config: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Loads and parses the configuration file into memory.

        Raises:
            FileNotFoundError: If ``self.config_path`` does not exist.
            ValueError: If the file contents are not a valid JSON object.
        """
        if not self.config_path.is_file():
            logger.error("Configuration file not found: %s", self.config_path)
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        try:
            raw_text = self.config_path.read_text(encoding="utf-8")
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error(
                "Configuration file is not valid JSON: %s (%s)",
                self.config_path,
                exc,
            )
            raise ValueError(
                f"Configuration file is not valid JSON: {self.config_path}"
            ) from exc

        if not isinstance(parsed, dict):
            logger.error(
                "Configuration root must be a JSON object, got %s: %s",
                type(parsed).__name__,
                self.config_path,
            )
            raise ValueError(
                f"Configuration root must be a JSON object: {self.config_path}"
            )

        with self._lock:
            self._config = parsed

        logger.info("Loaded configuration from %s", self.config_path)

    def reload(self) -> None:
        """Reloads the configuration file from disk, replacing the cache.

        Raises:
            FileNotFoundError: If the configuration file no longer exists.
            ValueError: If the file contents are not a valid JSON object.
        """
        with self._lock:
            self._load()
        logger.info("Configuration reloaded from %s", self.config_path)

    def save(self) -> None:
        """Persists the in-memory configuration to disk atomically.

        The write sequence is: serialize to a temporary file created in
        the same directory as ``config_path``, flush the userspace
        buffer, ``fsync`` to force the write to storage, then atomically
        replace the target file with ``os.replace``. ``config_path`` is
        never opened for writing directly, so a crash mid-write cannot
        corrupt it.

        Raises:
            OSError: If the temporary file cannot be written or the
                atomic replace fails.
        """
        with self._lock:
            config_snapshot = json.loads(json.dumps(self._config))

        directory = self.config_path.parent
        directory.mkdir(parents=True, exist_ok=True)

        temp_fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.config_path.name}.",
            suffix=".tmp",
            dir=str(directory),
        )
        temp_path = Path(temp_name)

        try:
            with os.fdopen(temp_fd, mode="w", encoding="utf-8") as temp_file:
                json.dump(config_snapshot, temp_file, indent=2)
                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())

            os.replace(temp_path, self.config_path)
            logger.info("Configuration saved atomically to %s", self.config_path)
        except OSError:
            logger.exception(
                "Failed to atomically save configuration to %s", self.config_path
            )
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise

    def get(self, path: str, default: Any = None) -> Any:
        """Retrieves a value using dot-notation path lookup.

        Args:
            path: Dot-separated key path, e.g. ``"filter.beta"`` resolves
                to ``config["filter"]["beta"]``.
            default: Value returned if any segment of ``path`` is missing
                or if an intermediate value is not a mapping.

        Returns:
            The resolved value, or ``default`` if the path cannot be
            fully resolved.

        Raises:
            ValueError: If ``path`` is an empty string.
        """
        if not path:
            logger.error("get() called with an empty path")
            raise ValueError("path must be a non-empty string")

        segments = path.split(".")
        with self._lock:
            current: Any = self._config
            for segment in segments:
                if isinstance(current, dict) and segment in current:
                    current = current[segment]
                else:
                    logger.debug(
                        "Path '%s' not found in configuration; returning default",
                        path,
                    )
                    return default
            return current

    def set(self, path: str, value: Any) -> None:
        """Sets a value using a dot-notation path, creating intermediate dicts.

        Args:
            path: Dot-separated key path, e.g. ``"filter.beta"`` sets
                ``config["filter"]["beta"] = value``.
            value: The value to store at the resolved path.

        Raises:
            ValueError: If ``path`` is an empty string, or if an
                intermediate segment of the path already holds a
                non-dict value and therefore cannot be traversed into.
        """
        if not path:
            logger.error("set() called with an empty path")
            raise ValueError("path must be a non-empty string")

        segments = path.split(".")
        with self._lock:
            current: dict[str, Any] = self._config
            for segment in segments[:-1]:
                existing = current.get(segment)
                if existing is None:
                    new_node: dict[str, Any] = {}
                    current[segment] = new_node
                    current = new_node
                elif isinstance(existing, dict):
                    current = existing
                else:
                    logger.error(
                        "Cannot set path '%s': segment '%s' is not an object",
                        path,
                        segment,
                    )
                    raise ValueError(
                        f"Cannot set path '{path}': segment '{segment}' "
                        "is not an object"
                    )
            current[segments[-1]] = value

        logger.debug("Set configuration path '%s'", path)

    def as_dict(self) -> dict[str, Any]:
        """Returns a deep copy of the entire in-memory configuration.

        Returns:
            A deep copy of the cached configuration dictionary. Mutating
            the returned dictionary has no effect on the manager's
            internal state.
        """
        with self._lock:
            return json.loads(json.dumps(self._config))
