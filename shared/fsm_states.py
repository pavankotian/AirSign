# =============================================================================
# shared/fsm_states.py
# AirSign — FSM State & Event Vocabulary
# -----------------------------------------------------------------------------
# Single source of truth for all Finite State Machine state and event strings.
#
# Consumed by:
#   Dev A — perception/gesture/fsm.py       (produces states and events)
#   Dev B — application/action_thread.py    (dispatches on events)
#           ui/gesture_panel.py             (displays current state)
#
# FROZEN: Do not add, remove, or rename entries without coordinating with
# both tracks. The ActionThread switch-on-event logic will silently ignore
# any event string not present in its dispatch table.
# =============================================================================

from __future__ import annotations

# ---------------------------------------------------------------------------
# FSM States — the set of stable configurations the gesture FSM can occupy.
#
# The FSM is always in exactly one state. States are displayed in the
# GesturePanel and color-coded to give the user feedback about mode.
#
# Transition rules are implemented in perception/gesture/fsm.py.
# Display colors for each state are defined in ui/gesture_panel.py.
# ---------------------------------------------------------------------------

FSM_STATES: list[str] = [
    "IDLE",           # No hand present, or NONE/UNKNOWN gesture held for
                      # HAND_LOST_FRAMES consecutive frames. Cursor injection
                      # paused. Entry state on app start.

    "HOVER",          # Hand tracked with OPEN_PALM. Cursor moves freely.
                      # No click or scroll actions dispatched in this state.
                      # The "default active" state — most time is spent here.

    "PINCH_START",    # PINCH gesture detected with sufficient confidence.
                      # Waiting to determine if this is a click (short) or
                      # drag (held beyond PINCH_HOLD_MS). Cursor still moves.

    "PINCH_HELD",     # PINCH sustained longer than PINCH_HOLD_MS threshold.
                      # Mouse button is held down — drag mode is active.
                      # Releases on next non-PINCH frame.

    "SCROLL_MODE",    # FIST gesture detected. Wrist Y-delta drives scroll
                      # velocity. Continuous scroll injection active.
                      # Cursor does not move while in this state.

    "SWIPE_PENDING",  # POINT gesture detected. FSM is tracking the index
                      # fingertip trajectory. Waiting for displacement to
                      # cross SWIPE_THRESHOLD in a cardinal direction.

    "SWIPE_COMMIT",   # Swipe threshold crossed. Fires for exactly ONE frame,
                      # then auto-transitions back to HOVER. The corresponding
                      # SWIPE_* event fires on this same frame.
]

# ---------------------------------------------------------------------------
# FSM States frozenset — O(1) membership validation.
# ---------------------------------------------------------------------------

FSM_STATE_SET: frozenset[str] = frozenset(FSM_STATES)

# ---------------------------------------------------------------------------
# FSM Events — discrete transition signals produced by the FSM.
#
# Events are what Dev B's ActionThread dispatches on. An event fires at most
# once per state transition (not every frame). Most frames produce event=None.
#
# The FSM emits exactly one event per transition, or None if no transition
# occurred on that frame. Dev B should treat event=None as a no-op.
# ---------------------------------------------------------------------------

FSM_EVENTS: list[str] = [
    # ── Entry / exit events ──────────────────────────────────────────────────
    "ENTER_HOVER",      # Any state → HOVER. Hand appeared or gesture returned
                        # to open palm. Cursor injection resumes.

    "HAND_LOST",        # Any state → IDLE. Hand left the frame or disappeared
                        # for HAND_LOST_FRAMES consecutive frames.

    # ── Pinch events ─────────────────────────────────────────────────────────
    "PINCH_DETECTED",   # HOVER → PINCH_START. PINCH gesture first confirmed
                        # with confidence >= MIN_ACTION_CONFIDENCE.

    "PINCH_HELD",       # PINCH_START → PINCH_HELD. PINCH sustained beyond
                        # PINCH_HOLD_MS. Dev B presses mouse button on this event.

    "PINCH_RELEASED",   # PINCH_START → HOVER (short tap → left click).
                        # PINCH_HELD → HOVER (drag end → mouse release).
                        # Dev B maps this to left_click for the short-tap case.

    "DOUBLE_PINCH",     # Replaces PINCH_RELEASED when two pinch-release cycles
                        # occur within DOUBLE_PINCH_MS of each other.
                        # Dev B maps this to right_click.

    # ── Scroll events ────────────────────────────────────────────────────────
    "FIST_ENTER",       # HOVER → SCROLL_MODE. FIST gesture confirmed.
                        # Dev B begins reading wrist Y-delta for scroll velocity.

    "FIST_EXIT",        # SCROLL_MODE → HOVER. FIST gesture released.
                        # Dev B stops scroll injection.

    # ── Navigate / swipe events ──────────────────────────────────────────────
    "POINT_DETECTED",   # HOVER → SWIPE_PENDING. POINT gesture confirmed.
                        # FSM begins tracking index tip trajectory.

    "SWIPE_LEFT",       # SWIPE_PENDING → SWIPE_COMMIT. Index tip traveled
                        # > SWIPE_THRESHOLD leftward. Fires once per swipe.

    "SWIPE_RIGHT",      # SWIPE_PENDING → SWIPE_COMMIT. Index tip traveled
                        # > SWIPE_THRESHOLD rightward. Fires once per swipe.

    "SWIPE_UP",         # SWIPE_PENDING → SWIPE_COMMIT. Index tip traveled
                        # > SWIPE_THRESHOLD upward. Fires once per swipe.

    "SWIPE_DOWN",       # SWIPE_PENDING → SWIPE_COMMIT. Index tip traveled
                        # > SWIPE_THRESHOLD downward. Fires once per swipe.
]

# ---------------------------------------------------------------------------
# FSM Events frozenset — O(1) membership validation.
# ---------------------------------------------------------------------------

FSM_EVENT_SET: frozenset[str] = frozenset(FSM_EVENTS)

# ---------------------------------------------------------------------------
# Convenience groupings — used by Dev B for dispatch table construction
# and by Dev A for FSM internal logic grouping.
# ---------------------------------------------------------------------------

SWIPE_EVENTS: frozenset[str] = frozenset({
    "SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN"
})

PINCH_EVENTS: frozenset[str] = frozenset({
    "PINCH_DETECTED", "PINCH_HELD", "PINCH_RELEASED", "DOUBLE_PINCH"
})

SCROLL_EVENTS: frozenset[str] = frozenset({
    "FIST_ENTER", "FIST_EXIT"
})

# ---------------------------------------------------------------------------
# Public validation helpers
# ---------------------------------------------------------------------------

def is_valid_state(state: str) -> bool:
    """
    Returns True if state is a member of FSM_STATES.

    Args:
        state: FSM state string to validate.

    Returns:
        True if state is a recognized FSM state string.

    Example:
        >>> is_valid_state("HOVER")
        True
        >>> is_valid_state("CLICKING")
        False
    """
    return state in FSM_STATE_SET


def is_valid_event(event: str) -> bool:
    """
    Returns True if event is a member of FSM_EVENTS.

    Args:
        event: FSM event string to validate.

    Returns:
        True if event is a recognized FSM event string.

    Example:
        >>> is_valid_event("SWIPE_LEFT")
        True
        >>> is_valid_event(None)
        False
    """
    return event in FSM_EVENT_SET


def is_swipe_event(event: str) -> bool:
    """Returns True if event is one of the four directional swipe events."""
    return event in SWIPE_EVENTS


def is_pinch_event(event: str) -> bool:
    """Returns True if event is part of the pinch event family."""
    return event in PINCH_EVENTS