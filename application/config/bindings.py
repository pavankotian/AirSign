"""FSM event to OS action binding resolution for the AirSign Developer B track.

This module reads the ``bindings`` section of the application
configuration (exposed via
:class:`~application.config.settings_manager.SettingsManager`) and
exposes a validated, thread-safe lookup from FSM event strings to action
identifier strings. Keys that are not recognized FSM events are logged
as warnings and discarded rather than raising an exception.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from shared.fsm_states import FSM_EVENT_SET, FSM_EVENTS

if TYPE_CHECKING:
    from application.config.settings_manager import SettingsManager

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_bindings_cache: dict[str, str] = {}
_loaded: bool = False


def _validate_and_build(raw_bindings: dict[str, Any]) -> dict[str, str]:
    """Validates raw binding entries and builds the event-to-action mapping.

    Args:
        raw_bindings: The raw ``bindings`` mapping loaded from
            configuration, with arbitrary string keys and values.

    Returns:
        A mapping containing only entries whose key is a valid FSM event
        string (per :data:`shared.fsm_states.FSM_EVENT_SET`) and whose
        value is a non-null string.
    """
    validated: dict[str, str] = {}

    for key, value in raw_bindings.items():
        if key not in FSM_EVENT_SET:
            logger.warning(
                "Ignoring configuration binding for '%s': not a recognized "
                "FSM event",
                key,
            )
            continue

        if value is None:
            logger.debug("Binding for FSM event '%s' is explicitly unset", key)
            continue

        if not isinstance(value, str):
            logger.warning(
                "Ignoring binding for FSM event '%s': action value must be "
                "a string, got %s",
                key,
                type(value).__name__,
            )
            continue

        validated[key] = value

    return validated


def load_bindings(settings_manager: "SettingsManager") -> None:
    """Loads and validates bindings from configuration into the cache.

    Reads the ``bindings`` section of ``settings_manager``, discards any
    key that is not a member of ``FSM_EVENTS`` (logging a warning for
    each discarded key), and populates the module-level cache with the
    remainder.

    Args:
        settings_manager: The settings manager instance to read the
            ``bindings`` section from.
    """
    raw_bindings = settings_manager.get("bindings", {})

    if not isinstance(raw_bindings, dict):
        logger.error(
            "Configuration 'bindings' section is not an object (got %s); "
            "clearing binding cache",
            type(raw_bindings).__name__,
        )
        raw_bindings = {}

    validated = _validate_and_build(raw_bindings)

    global _loaded
    with _lock:
        _bindings_cache.clear()
        _bindings_cache.update(validated)
        _loaded = True

    logger.info(
        "Loaded %d valid FSM event binding(s) out of %d known FSM events",
        len(validated),
        len(FSM_EVENTS),
    )


def reload_bindings(settings_manager: "SettingsManager") -> None:
    """Reloads bindings from configuration, replacing the current cache.

    Args:
        settings_manager: The settings manager instance to reload the
            ``bindings`` section from. Callers typically invoke
            ``settings_manager.reload()`` beforehand to refresh the
            manager's own on-disk view before this call.
    """
    logger.info("Reloading FSM event bindings")
    load_bindings(settings_manager)


def get_binding(event: str) -> str | None:
    """Resolves an FSM event string to its bound action identifier.

    Args:
        event: The FSM event string to look up, e.g.
            ``"PINCH_RELEASED"``.

    Returns:
        The bound action identifier string if ``event`` has a valid,
        loaded binding; ``None`` if the event is unbound, unrecognized,
        or bindings have not yet been loaded.
    """
    with _lock:
        if not _loaded:
            logger.debug(
                "get_binding('%s') called before load_bindings(); returning None",
                event,
            )
            return None
        return _bindings_cache.get(event)
