"""Normalized-to-screen coordinate mapping for the AirSign Developer B track.

This module implements :class:`CoordinateMapper`, a calibration and
testing utility that converts normalized camera-space coordinates
(0.0-1.0) into screen pixel coordinates, honoring a configurable active
zone read from :class:`~application.config.settings_manager.SettingsManager`.

This class is not on the real-time cursor-injection hot path (the
production cursor position is computed by Dev A's ``LandmarkSmoother``);
it exists for calibration tooling, manual testing, and any future
on-screen calibration workflow.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import screeninfo

if TYPE_CHECKING:
    from application.config.settings_manager import SettingsManager

logger = logging.getLogger(__name__)

_REQUIRED_ZONE_KEYS = ("x1", "y1", "x2", "y2")
_FALLBACK_SCREEN_WIDTH = 1920
_FALLBACK_SCREEN_HEIGHT = 1080


class CoordinateMapper:
    """Maps normalized camera coordinates to clamped screen pixel coordinates.

    The mapper reads its active zone from a
    :class:`~application.config.settings_manager.SettingsManager`
    instance rather than accessing configuration files directly, and it
    resolves the primary monitor's resolution once at construction time
    via ``screeninfo`` rather than ever hardcoding screen dimensions.

    All public methods are protected by a single reentrant lock so the
    mapper may be shared safely across threads.

    Attributes:
        settings_manager: The settings manager this mapper reads its
            active zone configuration from.
    """

    def __init__(self, settings_manager: "SettingsManager") -> None:
        """Initializes the mapper from configuration and monitor detection.

        Args:
            settings_manager: The settings manager to read the
                ``tracking.active_zone`` section from.

        Raises:
            ValueError: If the configured active zone is missing,
                malformed, or otherwise fails validation.
        """
        self._lock = threading.RLock()
        self.settings_manager = settings_manager

        default_zone = {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95}
        raw_zone = settings_manager.get("tracking.active_zone", default_zone)
        self._active_zone: dict[str, float] = self._validate_zone(raw_zone)

        self._screen_width, self._screen_height = self._detect_screen_size()

        logger.info(
            "CoordinateMapper initialized: active_zone=%s, screen=%dx%d",
            self._active_zone,
            self._screen_width,
            self._screen_height,
        )

    @staticmethod
    def _detect_screen_size() -> tuple[int, int]:
        """Resolves the primary monitor's resolution via screeninfo.

        Returns:
            A ``(width, height)`` tuple in pixels. Falls back to a
            fixed default resolution if no monitor can be detected,
            logging a warning in that case.
        """
        try:
            monitors = screeninfo.get_monitors()
        except Exception:  # noqa: BLE001 - screeninfo backends vary by OS
            logger.warning(
                "Failed to query screeninfo for monitor list; falling back "
                "to default resolution %dx%d",
                _FALLBACK_SCREEN_WIDTH,
                _FALLBACK_SCREEN_HEIGHT,
            )
            return _FALLBACK_SCREEN_WIDTH, _FALLBACK_SCREEN_HEIGHT

        if not monitors:
            logger.warning(
                "No monitors detected by screeninfo; falling back to "
                "default resolution %dx%d",
                _FALLBACK_SCREEN_WIDTH,
                _FALLBACK_SCREEN_HEIGHT,
            )
            return _FALLBACK_SCREEN_WIDTH, _FALLBACK_SCREEN_HEIGHT

        primary = next((monitor for monitor in monitors if monitor.is_primary), None)
        selected = primary if primary is not None else monitors[0]

        return int(selected.width), int(selected.height)

    @staticmethod
    def _validate_zone(zone: Any) -> dict[str, float]:
        """Validates and normalizes an active zone mapping.

        Args:
            zone: A candidate active zone value, expected to be a
                mapping containing the keys ``x1``, ``y1``, ``x2``,
                and ``y2`` with numeric values in the range 0.0-1.0.

        Returns:
            A new dictionary with the four keys coerced to ``float``.

        Raises:
            ValueError: If ``zone`` is not a mapping, is missing a
                required key, contains a non-numeric value, or the
                bounds are not well-formed (``x1 < x2`` and
                ``y1 < y2``, all within ``[0.0, 1.0]``).
        """
        if not isinstance(zone, dict):
            logger.error("Active zone must be an object, got %s", type(zone).__name__)
            raise ValueError(
                f"Active zone must be an object, got {type(zone).__name__}"
            )

        missing_keys = [key for key in _REQUIRED_ZONE_KEYS if key not in zone]
        if missing_keys:
            logger.error("Active zone is missing key(s): %s", missing_keys)
            raise ValueError(f"Active zone is missing key(s): {missing_keys}")

        coerced: dict[str, float] = {}
        for key in _REQUIRED_ZONE_KEYS:
            value = zone[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                logger.error(
                    "Active zone key '%s' must be numeric, got %s",
                    key,
                    type(value).__name__,
                )
                raise ValueError(
                    f"Active zone key '{key}' must be numeric, got "
                    f"{type(value).__name__}"
                )
            coerced[key] = float(value)

        x1, y1, x2, y2 = coerced["x1"], coerced["y1"], coerced["x2"], coerced["y2"]

        for key, value in coerced.items():
            if not (0.0 <= value <= 1.0):
                logger.error(
                    "Active zone key '%s' must be within [0.0, 1.0], got %s",
                    key,
                    value,
                )
                raise ValueError(
                    f"Active zone key '{key}' must be within [0.0, 1.0], "
                    f"got {value}"
                )

        if x1 >= x2:
            logger.error("Active zone requires x1 < x2, got x1=%s, x2=%s", x1, x2)
            raise ValueError(f"Active zone requires x1 < x2, got x1={x1}, x2={x2}")

        if y1 >= y2:
            logger.error("Active zone requires y1 < y2, got y1=%s, y2=%s", y1, y2)
            raise ValueError(f"Active zone requires y1 < y2, got y1={y1}, y2={y2}")

        return coerced

    def _apply_calibration(self, x: float, y: float) -> tuple[float, float]:
        """Applies a calibration transform to raw normalized coordinates.

        This is an extension point reserved for future calibration
        support (for example, a per-user homography or lens-distortion
        correction). It is currently the identity transform and is
        called by :meth:`to_screen` before active-zone normalization,
        so future calibration logic can be added here without changing
        the public API.

        Args:
            x: Raw normalized camera x-coordinate.
            y: Raw normalized camera y-coordinate.

        Returns:
            The calibrated ``(x, y)`` coordinate pair.
        """
        return x, y

    def normalize(self, x: float, y: float) -> tuple[float, float]:
        """Maps a raw normalized camera coordinate into active-zone space.

        Args:
            x: Raw normalized camera x-coordinate (0.0-1.0, though
                values outside this range are accepted and clamped).
            y: Raw normalized camera y-coordinate (0.0-1.0, though
                values outside this range are accepted and clamped).

        Returns:
            A ``(norm_x, norm_y)`` tuple, each clamped to ``[0.0, 1.0]``,
            representing the position of ``(x, y)`` relative to the
            configured active zone.
        """
        with self._lock:
            zone = self._active_zone

        zone_width = zone["x2"] - zone["x1"]
        zone_height = zone["y2"] - zone["y1"]

        norm_x = (x - zone["x1"]) / zone_width
        norm_y = (y - zone["y1"]) / zone_height

        norm_x = min(max(norm_x, 0.0), 1.0)
        norm_y = min(max(norm_y, 0.0), 1.0)

        return norm_x, norm_y

    def clamp(self, x: float, y: float) -> tuple[int, int]:
        """Clamps a screen pixel coordinate to the detected screen bounds.

        Args:
            x: Screen x-coordinate in pixels, may be fractional or
                outside screen bounds.
            y: Screen y-coordinate in pixels, may be fractional or
                outside screen bounds.

        Returns:
            An integer ``(x, y)`` pixel coordinate, clamped to
            ``[0, width - 1]`` and ``[0, height - 1]`` respectively.
        """
        with self._lock:
            width, height = self._screen_width, self._screen_height

        clamped_x = min(max(round(x), 0), width - 1)
        clamped_y = min(max(round(y), 0), height - 1)

        return clamped_x, clamped_y

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Converts a normalized camera coordinate to a screen pixel coordinate.

        Args:
            x: Raw normalized camera x-coordinate (0.0-1.0).
            y: Raw normalized camera y-coordinate (0.0-1.0).

        Returns:
            An integer ``(screen_x, screen_y)`` pixel coordinate, mapped
            through the active zone and clamped to screen bounds.
        """
        calibrated_x, calibrated_y = self._apply_calibration(x, y)
        norm_x, norm_y = self.normalize(calibrated_x, calibrated_y)

        with self._lock:
            width, height = self._screen_width, self._screen_height

        screen_x = norm_x * width
        screen_y = norm_y * height

        return self.clamp(screen_x, screen_y)

    def update_active_zone(self, zone: dict[str, float]) -> None:
        """Validates and applies a new active zone, persisting it to settings.

        Args:
            zone: A mapping containing ``x1``, ``y1``, ``x2``, and
                ``y2`` numeric keys, each within ``[0.0, 1.0]``, with
                ``x1 < x2`` and ``y1 < y2``.

        Raises:
            ValueError: If ``zone`` fails validation.
        """
        validated = self._validate_zone(zone)

        with self._lock:
            self._active_zone = validated

        self.settings_manager.set("tracking.active_zone", validated)

        logger.info("Active zone updated to %s", validated)

    def get_screen_size(self) -> tuple[int, int]:
        """Returns the detected primary monitor resolution.

        Returns:
            A ``(width, height)`` tuple in pixels.
        """
        with self._lock:
            return self._screen_width, self._screen_height
