# =============================================================================
# shared/gesture_labels.py
# AirSign — Gesture Label Vocabulary
# -----------------------------------------------------------------------------
# Single source of truth for all gesture classification output strings.
# Both Dev A (classifier output) and Dev B (action binding keys) import
# exclusively from this file. Raw gesture string literals must never appear
# anywhere else in the codebase.
#
# FROZEN: Do not add, remove, or rename entries without coordinating with
# both Dev A and Dev B, as changes break the inter-thread data contract.
# =============================================================================

from __future__ import annotations

# ---------------------------------------------------------------------------
# Primary vocabulary list — order is preserved and meaningful.
# The MLP label encoder uses this ordering as its class index mapping.
# ---------------------------------------------------------------------------

GESTURE_LABELS: list[str] = [
    "OPEN_PALM",   # All 5 fingers fully extended — neutral hover / cursor move state
    "FIST",        # All 4 fingers fully curled into palm — scroll mode trigger
    "PINCH",       # Thumb tip + index tip distance below pinch_tolerance — click trigger
    "POINT",       # Index finger extended, all others curled — swipe / navigate mode
    "TWO_FINGER",  # Index + middle extended, ring + pinky curled — right-click / mode switch
    "THUMBS_UP",   # Thumb extended upward, all fingers curled — confirm / media play
    "NONE",        # No hand detected in the current frame
    "UNKNOWN",     # Hand detected but classifier confidence is below MIN_ACTION_CONFIDENCE
]

# ---------------------------------------------------------------------------
# Frozenset for O(1) membership validation.
# Use this in schema validators and assert statements — never iterate over it.
# ---------------------------------------------------------------------------

GESTURE_LABEL_SET: frozenset[str] = frozenset(GESTURE_LABELS)

# ---------------------------------------------------------------------------
# Classifiable gestures — excludes NONE and UNKNOWN.
# These are the 6 classes the MLP is trained on and the data collector targets.
# Use this list when iterating over training classes or building the binding UI.
# ---------------------------------------------------------------------------

CLASSIFIABLE_GESTURES: list[str] = [
    label for label in GESTURE_LABELS
    if label not in ("NONE", "UNKNOWN")
]

# ---------------------------------------------------------------------------
# Public validation helper
# ---------------------------------------------------------------------------

def is_valid_gesture(label: str) -> bool:
    """
    Returns True if label is a member of GESTURE_LABELS.

    Used by:
    - mock_application.py schema validator (Dev A side)
    - action_thread.py input validation (Dev B side)

    Args:
        label: Gesture label string to validate.

    Returns:
        True if label is a recognized gesture string, False otherwise.

    Example:
        >>> is_valid_gesture("PINCH")
        True
        >>> is_valid_gesture("pinch")   # case-sensitive
        False
        >>> is_valid_gesture("UNKNOWN")
        True
    """
    return label in GESTURE_LABEL_SET


def is_actionable_gesture(label: str) -> bool:
    """
    Returns True if label is classifiable (not NONE or UNKNOWN).

    Use this to gate logic that should only run when a real gesture
    has been confidently detected.

    Args:
        label: Gesture label string to check.

    Returns:
        True if label is one of the 6 trained classifiable gestures.
    """
    return label in GESTURE_LABEL_SET and label not in ("NONE", "UNKNOWN")