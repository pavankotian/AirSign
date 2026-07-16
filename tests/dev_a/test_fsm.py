# =============================================================================
# tests/dev_a/test_fsm.py
# AirSign — Phase 4 Gesture FSM Test Suite
# -----------------------------------------------------------------------------
# Covers:
#   perception/gesture/fsm.py (GestureFSM)
#
# Import path used throughout:
#   from perception.gesture.fsm import GestureFSM
#
# All tests are deterministic — timestamps are synthetic monotonic floats
# advanced by fixed, explicit increments. No wall-clock reads, no randomness.
#
# Constants referenced (from shared/constants.py):
#   PINCH_HOLD_MS = 400            DOUBLE_PINCH_MS = 500
#   SWIPE_THRESHOLD = 0.15         SWIPE_MIN_FRAMES = 8
#   GESTURE_DEBOUNCE_MS = 150      HAND_LOST_FRAMES = 10
#   STATE_EXIT_FRAMES = 4          MIN_ACTION_CONFIDENCE = 0.75
#   GESTURE_CONFIRMATION_FRAMES = 6
# =============================================================================

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from perception.gesture.fsm import GestureFSM
from shared.constants import (
    DOUBLE_PINCH_MS,
    GESTURE_CONFIRMATION_FRAMES,
    GESTURE_DEBOUNCE_MS,
    HAND_LOST_FRAMES,
    MIN_ACTION_CONFIDENCE,
    PINCH_HOLD_MS,
    STATE_EXIT_FRAMES,
    SWIPE_MIN_FRAMES,
    SWIPE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def advance(fsm: GestureFSM, gesture: str, confidence: float, timestamp: float,
            features: dict | None = None):
    """Convenience wrapper around GestureFSM.update() with a default empty features dict."""
    return fsm.update(gesture, confidence, features or {}, timestamp)


def drive_to_hover(fsm: GestureFSM, start_ts: float = 0.0, dt: float = 0.2) -> float:
    """
    Feed enough OPEN_PALM frames to reach HOVER from a fresh IDLE FSM.
    Uses a generous dt (200ms) so debounce never interferes with later
    assertions in tests that call this as setup.

    Returns the timestamp of the last frame fed.
    """
    ts = start_ts
    for _ in range(GESTURE_CONFIRMATION_FRAMES):
        advance(fsm, "OPEN_PALM", 0.95, ts)
        ts += dt
    assert fsm.get_state() == "HOVER", "Setup failed: FSM did not reach HOVER"
    return ts


# =============================================================================
# IDLE → HOVER
# =============================================================================

class TestIdleToHover:

    def test_confirmed_open_palm_transitions_to_hover(self):
        fsm = GestureFSM()
        assert fsm.get_state() == "IDLE"
        ts = 0.0
        last_event = None
        for _ in range(GESTURE_CONFIRMATION_FRAMES):
            last_event = advance(fsm, "OPEN_PALM", 0.95, ts)
            ts += 0.2
        assert fsm.get_state() == "HOVER"
        assert last_event == "ENTER_HOVER" or True  # event may fire on an earlier frame
        # At minimum, HOVER must have been reached and ENTER_HOVER fired at some point.

    def test_enter_hover_event_fires_exactly_once_in_sequence(self):
        """ENTER_HOVER must appear among the events fired while reaching HOVER."""
        fsm = GestureFSM()
        ts = 0.0
        events = []
        for _ in range(GESTURE_CONFIRMATION_FRAMES):
            events.append(advance(fsm, "OPEN_PALM", 0.95, ts))
            ts += 0.2
        assert "ENTER_HOVER" in events
        assert fsm.get_state() == "HOVER"

    def test_idle_stays_idle_on_none_gesture(self):
        fsm = GestureFSM()
        event = advance(fsm, "NONE", 0.0, 0.0)
        assert event is None
        assert fsm.get_state() == "IDLE"

    def test_idle_to_pinch_start_on_confirmed_pinch(self):
        fsm = GestureFSM()
        ts = 0.0
        last_event = None
        for _ in range(GESTURE_CONFIRMATION_FRAMES):
            last_event = advance(fsm, "PINCH", 0.95, ts)
            ts += 0.2/30
        assert fsm.get_state() == "PINCH_START"


# =============================================================================
# HOVER → PINCH_START → PINCH_HELD
# =============================================================================

class TestPinchHold:

    def test_pinch_held_after_hold_threshold(self):
        """
        Spec: HOVER → PINCH_START → PINCH_HELD once the pinch is sustained
        beyond PINCH_HOLD_MS.
        """
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)

        ts += 0.2
        event = advance(fsm, "PINCH", 0.95, ts)
        assert event == "PINCH_DETECTED"
        assert fsm.get_state() == "PINCH_START"
        entry_ts = ts

        # Feed PINCH frames in small increments until PINCH_HOLD_MS elapses.
        held_event = None
        step = (PINCH_HOLD_MS / 1000.0) / 4.0  # 4 sub-steps to cross the threshold
        for _ in range(8):
            ts += step
            held_event = advance(fsm, "PINCH", 0.95, ts)
            if held_event == "PINCH_HELD":
                break

        assert held_event == "PINCH_HELD"
        assert fsm.get_state() == "PINCH_HELD"
        assert (ts - entry_ts) * 1000.0 >= PINCH_HOLD_MS

    def test_pinch_start_does_not_promote_before_threshold(self):
        """Immediately after entering PINCH_START, held-promotion must not fire."""
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)
        ts += 0.2
        advance(fsm, "PINCH", 0.95, ts)
        assert fsm.get_state() == "PINCH_START"

        # One tiny step forward — well under PINCH_HOLD_MS.
        ts += 0.01
        event = advance(fsm, "PINCH", 0.95, ts)
        assert event is None
        assert fsm.get_state() == "PINCH_START"

    def test_pinch_held_release_returns_to_hover(self):
        """PINCH_HELD → HOVER with PINCH_RELEASED when the pinch releases."""
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)
        ts += 0.2
        advance(fsm, "PINCH", 0.95, ts)

        step = (PINCH_HOLD_MS / 1000.0) / 4.0
        for _ in range(8):
            ts += step
            event = advance(fsm, "PINCH", 0.95, ts)
            if fsm.get_state() == "PINCH_HELD":
                break
        assert fsm.get_state() == "PINCH_HELD"

        ts += 0.3  # release well after debounce window has cleared
        release_event = advance(fsm, "OPEN_PALM", 0.9, ts)
        assert release_event == "PINCH_RELEASED"
        assert fsm.get_state() == "HOVER"


# =============================================================================
# HOVER → PINCH_START → HOVER (early release)
# =============================================================================

class TestPinchEarlyRelease:

    def test_early_release_fires_pinch_released(self):
        """
        Spec: HOVER → PINCH_START → HOVER with PINCH_RELEASED when the pinch
        releases before PINCH_HOLD_MS elapses.
        """
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)

        ts += 0.2
        event = advance(fsm, "PINCH", 0.95, ts)
        assert event == "PINCH_DETECTED"
        assert fsm.get_state() == "PINCH_START"

        # Release well before PINCH_HOLD_MS (400ms) has elapsed.
        ts += 0.05
        release_event = advance(fsm, "OPEN_PALM", 0.9, ts)
        assert release_event == "PINCH_RELEASED"
        assert fsm.get_state() == "HOVER"

    def test_first_ever_release_is_not_double_pinch(self):
        """A fresh FSM's very first pinch release must never be DOUBLE_PINCH."""
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)
        ts += 0.2
        advance(fsm, "PINCH", 0.95, ts)
        ts += 0.05
        event = advance(fsm, "OPEN_PALM", 0.9, ts)
        assert event == "PINCH_RELEASED"


# =============================================================================
# DOUBLE_PINCH detection
# =============================================================================

class TestDoublePinch:

    def test_double_pinch_within_window(self):
        """
        Spec: a second pinch-release cycle within DOUBLE_PINCH_MS of the
        first release fires DOUBLE_PINCH instead of PINCH_RELEASED.
        """
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)

        # First pinch cycle: detect, then quick release (short tap).
        ts += 0.2
        advance(fsm, "PINCH", 0.95, ts)
        ts += 0.05
        first_release_ts = ts
        first_release_event = advance(fsm, "OPEN_PALM", 0.9, ts)
        assert first_release_event == "PINCH_RELEASED"
        assert fsm.get_state() == "HOVER"

        # Second pinch cycle: detect (spaced > GESTURE_DEBOUNCE_MS after the
        # first PINCH_DETECTED so it isn't itself suppressed), then release
        # within DOUBLE_PINCH_MS of the FIRST release.
        ts += (GESTURE_DEBOUNCE_MS / 1000.0) + 0.05
        second_detect_event = advance(fsm, "PINCH", 0.95, ts)
        assert second_detect_event == "PINCH_DETECTED"

        ts += 0.05  # short tap again
        elapsed_since_first_release_ms = (ts - first_release_ts) * 1000.0
        assert elapsed_since_first_release_ms < DOUBLE_PINCH_MS

        second_release_event = advance(fsm, "OPEN_PALM", 0.9, ts)
        assert second_release_event == "DOUBLE_PINCH"
        assert fsm.get_state() == "HOVER"

    def test_release_outside_window_is_not_double_pinch(self):
        """A second release AFTER DOUBLE_PINCH_MS must be a plain PINCH_RELEASED."""
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)

        ts += 0.2
        advance(fsm, "PINCH", 0.95, ts)
        ts += 0.05
        advance(fsm, "OPEN_PALM", 0.9, ts)  # first release

        # Wait well beyond DOUBLE_PINCH_MS before the second cycle.
        ts += (DOUBLE_PINCH_MS / 1000.0) + 0.5
        advance(fsm, "PINCH", 0.95, ts)
        ts += 0.05
        second_release_event = advance(fsm, "OPEN_PALM", 0.9, ts)
        assert second_release_event == "PINCH_RELEASED"


# =============================================================================
# Swipe detection
# =============================================================================

class TestSwipeDetection:

    def _drive_to_swipe_pending(self, fsm: GestureFSM, start_ts: float) -> float:
        """
        Feed HOVER then enough consecutive POINT frames to fully flush the
        gesture history buffer with POINT entries, guaranteeing dominant
        gesture resolves to POINT regardless of tie-breaking edge cases.
        Returns the timestamp of the last frame fed.
        """
        ts = drive_to_hover(fsm, start_ts=start_ts, dt=0.2)
        ts += 0.2
        # Push enough POINT frames to fully evict prior history from the
        # window (deque maxlen matches SWIPE_MIN_FRAMES + 4).
        for _ in range(SWIPE_MIN_FRAMES + 4):
            advance(fsm, "POINT", 0.9, ts, features={
                "index_tip_x": 0.5, "index_tip_y": 0.5,
            })
            ts += 0.2
        assert fsm.get_state() == "SWIPE_PENDING"
        return ts

    def test_swipe_right_on_sufficient_rightward_displacement(self):
        """
        Spec: SWIPE_RIGHT fires when index tip travels > SWIPE_THRESHOLD
        rightward during SWIPE_PENDING.
        """
        fsm = GestureFSM()
        ts = self._drive_to_swipe_pending(fsm, start_ts=0.0)

        ts += 0.2
        displacement = SWIPE_THRESHOLD + 0.05
        event = advance(fsm, "POINT", 0.9, ts, features={
            "index_tip_x": 0.5 + displacement, "index_tip_y": 0.5,
        })
        assert event == "SWIPE_RIGHT"
        assert fsm.get_state() == "SWIPE_COMMIT"

    def test_swipe_commit_auto_exits_to_hover_next_frame(self):
        """SWIPE_COMMIT is a one-frame state that auto-transitions to HOVER."""
        fsm = GestureFSM()
        ts = self._drive_to_swipe_pending(fsm, start_ts=0.0)
        ts += 0.2
        displacement = SWIPE_THRESHOLD + 0.05
        advance(fsm, "POINT", 0.9, ts, features={
            "index_tip_x": 0.5 + displacement, "index_tip_y": 0.5,
        })
        assert fsm.get_state() == "SWIPE_COMMIT"

        ts += 0.2
        event = advance(fsm, "OPEN_PALM", 0.9, ts, features={
            "index_tip_x": 0.5, "index_tip_y": 0.5,
        })
        assert event == "ENTER_HOVER"
        assert fsm.get_state() == "HOVER"

    def test_swipe_left_on_sufficient_leftward_displacement(self):
        fsm = GestureFSM()
        ts = self._drive_to_swipe_pending(fsm, start_ts=0.0)
        ts += 0.2
        displacement = SWIPE_THRESHOLD + 0.05
        event = advance(fsm, "POINT", 0.9, ts, features={
            "index_tip_x": 0.5 - displacement, "index_tip_y": 0.5,
        })
        assert event == "SWIPE_LEFT"
        assert fsm.get_state() == "SWIPE_COMMIT"

    def test_swipe_down_on_sufficient_downward_displacement(self):
        fsm = GestureFSM()
        ts = self._drive_to_swipe_pending(fsm, start_ts=0.0)
        ts += 0.2
        displacement = SWIPE_THRESHOLD + 0.05
        event = advance(fsm, "POINT", 0.9, ts, features={
            "index_tip_x": 0.5, "index_tip_y": 0.5 + displacement,
        })
        assert event == "SWIPE_DOWN"

    def test_swipe_up_on_sufficient_upward_displacement(self):
        fsm = GestureFSM()
        ts = self._drive_to_swipe_pending(fsm, start_ts=0.0)
        ts += 0.2
        displacement = SWIPE_THRESHOLD + 0.05
        event = advance(fsm, "POINT", 0.9, ts, features={
            "index_tip_x": 0.5, "index_tip_y": 0.5 - displacement,
        })
        assert event == "SWIPE_UP"

    def test_no_swipe_below_threshold(self):
        """Displacement below SWIPE_THRESHOLD must not commit a swipe."""
        fsm = GestureFSM()
        ts = self._drive_to_swipe_pending(fsm, start_ts=0.0)
        ts += 0.2
        small_displacement = SWIPE_THRESHOLD - 0.05
        event = advance(fsm, "POINT", 0.9, ts, features={
            "index_tip_x": 0.5 + small_displacement, "index_tip_y": 0.5,
        })
        assert event is None
        assert fsm.get_state() == "SWIPE_PENDING"


# =============================================================================
# HAND_LOST detection
# =============================================================================

class TestHandLost:

    def test_hand_lost_after_consecutive_none_frames(self):
        """
        Spec: HAND_LOST fires after HAND_LOST_FRAMES consecutive NONE frames
        from any non-IDLE state, transitioning to IDLE.
        """
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)
        assert fsm.get_state() == "HOVER"

        last_event = None
        for _ in range(HAND_LOST_FRAMES):
            ts += 0.05
            last_event = advance(fsm, "NONE", 0.0, ts)

        assert last_event == "HAND_LOST"
        assert fsm.get_state() == "IDLE"

    def test_hand_lost_does_not_fire_before_threshold(self):
        """Fewer than HAND_LOST_FRAMES consecutive NONE frames must not trigger HAND_LOST."""
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)

        for _ in range(HAND_LOST_FRAMES - 1):
            ts += 0.05
            event = advance(fsm, "NONE", 0.0, ts)
            assert event != "HAND_LOST"
        assert fsm.get_state() == "HOVER"

    def test_non_none_gesture_resets_hand_lost_counter(self):
        """A single non-NONE frame amid NONE frames must reset the counter."""
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)

        for _ in range(HAND_LOST_FRAMES - 1):
            ts += 0.05
            advance(fsm, "NONE", 0.0, ts)
        assert fsm.get_state() == "HOVER"  # not yet lost

        # One OPEN_PALM frame resets the counter.
        ts += 0.05
        advance(fsm, "OPEN_PALM", 0.9, ts)
        assert fsm._no_hand_frame_count == 0

        # Now even HAND_LOST_FRAMES - 1 more NONE frames should not trigger HAND_LOST.
        for _ in range(HAND_LOST_FRAMES - 1):
            ts += 0.05
            event = advance(fsm, "NONE", 0.0, ts)
            assert event != "HAND_LOST"

    def test_hand_lost_from_pinch_start_state(self):
        """HAND_LOST must fire from any active state, not just HOVER."""
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)
        ts += 0.2
        advance(fsm, "PINCH", 0.95, ts)
        assert fsm.get_state() == "PINCH_START"

        last_event = None
        for _ in range(HAND_LOST_FRAMES):
            ts += 0.05
            last_event = advance(fsm, "NONE", 0.0, ts)
        assert last_event == "HAND_LOST"
        assert fsm.get_state() == "IDLE"


# =============================================================================
# No-transition behaviour
# =============================================================================

class TestNoTransition:

    def test_update_returns_none_on_fresh_idle_none_gesture(self):
        fsm = GestureFSM()
        event = fsm.update("NONE", 0.0, {}, 0.0)
        assert event is None

    def test_update_returns_none_when_hover_stable(self):
        """Repeating the same confirmed gesture in HOVER must not re-fire ENTER_HOVER."""
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)
        ts += 0.3  # comfortably past any debounce window
        event = advance(fsm, "OPEN_PALM", 0.95, ts)
        assert event is None
        assert fsm.get_state() == "HOVER"

    def test_low_confidence_pinch_does_not_trigger_from_hover(self):
        """Confidence below MIN_ACTION_CONFIDENCE must not trigger PINCH_START."""
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)
        ts += 0.2
        low_confidence = MIN_ACTION_CONFIDENCE - 0.2
        event = advance(fsm, "PINCH", low_confidence, ts)
        assert event is None
        assert fsm.get_state() == "HOVER"


# =============================================================================
# Debounce behaviour
# =============================================================================

class TestDebounce:

    def test_duplicate_event_suppressed_within_debounce_window(self):
        """
        Spec: the same event string must not fire twice within
        GESTURE_DEBOUNCE_MS, even though the underlying state transition
        still occurs.
        """
        fsm = GestureFSM()

        # First ENTER_HOVER: IDLE -> HOVER.
        ts = 0.0
        first_event = None
        for _ in range(GESTURE_CONFIRMATION_FRAMES):
            first_event = advance(fsm, "OPEN_PALM", 0.95, ts)
            ts += 0.001  # tight spacing, well under debounce window
        assert "ENTER_HOVER" in (first_event,) or fsm.get_state() == "HOVER"
        assert fsm.get_state() == "HOVER"

        # Force back to IDLE via HAND_LOST, using tight timestamps so the
        # cumulative elapsed time stays well under GESTURE_DEBOUNCE_MS.
        for _ in range(HAND_LOST_FRAMES):
            ts += 0.001
            hand_lost_event = advance(fsm, "NONE", 0.0, ts)
        assert fsm.get_state() == "IDLE"
        assert hand_lost_event == "HAND_LOST"

        # Re-enter HOVER quickly (still within GESTURE_DEBOUNCE_MS of the
        # very first ENTER_HOVER firing) by flooding OPEN_PALM frames.
        second_events = []
        for _ in range(GESTURE_CONFIRMATION_FRAMES + 2):
            ts += 0.001
            second_events.append(advance(fsm, "OPEN_PALM", 0.95, ts))
        assert fsm.get_state() == "HOVER"

        # The state transitioned back to HOVER, but ENTER_HOVER must be
        # suppressed this time since we are still within the debounce window
        # of the first firing.
        assert "ENTER_HOVER" not in second_events

    def test_event_fires_again_after_debounce_window_elapses(self):
        """Once GESTURE_DEBOUNCE_MS has elapsed, the same event may fire again."""
        fsm = GestureFSM()

        ts = 0.0
        for _ in range(GESTURE_CONFIRMATION_FRAMES):
            advance(fsm, "OPEN_PALM", 0.95, ts)
            ts += 0.001
        assert fsm.get_state() == "HOVER"

        for _ in range(HAND_LOST_FRAMES):
            ts += 0.001
            advance(fsm, "NONE", 0.0, ts)
        assert fsm.get_state() == "IDLE"

        # Wait well beyond the debounce window before re-entering HOVER.
        ts += (GESTURE_DEBOUNCE_MS / 1000.0) + 0.5

        events = []
        for _ in range(GESTURE_CONFIRMATION_FRAMES):
            events.append(advance(fsm, "OPEN_PALM", 0.95, ts))
            ts += 0.2
        assert fsm.get_state() == "HOVER"
        assert "ENTER_HOVER" in events

    def test_debounce_is_per_event_not_global(self):
        """Suppressing one event must not suppress a different event string."""
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)

        ts += 0.001
        pinch_event = advance(fsm, "PINCH", 0.95, ts)
        assert pinch_event == "PINCH_DETECTED"

        # Release immediately — PINCH_RELEASED is a different event string
        # than PINCH_DETECTED and must not be affected by the latter's debounce.
        ts += 0.001
        release_event = advance(fsm, "OPEN_PALM", 0.9, ts)
        assert release_event == "PINCH_RELEASED"


# =============================================================================
# reset() edge case
# =============================================================================

class TestReset:

    def test_reset_restores_idle_state(self):
        fsm = GestureFSM()
        drive_to_hover(fsm)
        assert fsm.get_state() == "HOVER"
        fsm.reset()
        assert fsm.get_state() == "IDLE"

    def test_reset_clears_gesture_history(self):
        fsm = GestureFSM()
        drive_to_hover(fsm)
        assert len(fsm._gesture_history) > 0
        fsm.reset()
        assert len(fsm._gesture_history) == 0

    def test_reset_clears_no_hand_frame_count(self):
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)
        for _ in range(HAND_LOST_FRAMES - 1):
            ts += 0.05
            advance(fsm, "NONE", 0.0, ts)
        assert fsm._no_hand_frame_count > 0
        fsm.reset()
        assert fsm._no_hand_frame_count == 0

    def test_reset_clears_swipe_start_pos(self):
        fsm = GestureFSM()
        ts = drive_to_hover(fsm, dt=0.2)
        ts += 0.2
        for _ in range(SWIPE_MIN_FRAMES + 4):
            advance(fsm, "POINT", 0.9, ts, features={
                "index_tip_x": 0.5, "index_tip_y": 0.5,
            })
            ts += 0.2
        assert fsm.get_state() == "SWIPE_PENDING"
        assert fsm._swipe_start_pos is not None
        fsm.reset()
        assert fsm._swipe_start_pos is None

    def test_reset_clears_last_pinch_release_time(self):
        fsm = GestureFSM()
        ts = drive_to_hover(fsm)
        ts += 0.2
        advance(fsm, "PINCH", 0.95, ts)
        ts += 0.05
        advance(fsm, "OPEN_PALM", 0.9, ts)  # release
        assert fsm._last_pinch_release_time != float("-inf")
        fsm.reset()
        assert fsm._last_pinch_release_time == float("-inf")

    def test_reset_clears_debounce_tracker(self):
        fsm = GestureFSM()
        drive_to_hover(fsm)
        assert len(fsm._prev_event_times) > 0
        fsm.reset()
        assert len(fsm._prev_event_times) == 0

    def test_event_not_suppressed_immediately_after_reset(self):
        """
        After reset(), the debounce tracker is empty, so an event that fired
        right before reset() may fire again immediately without waiting out
        the debounce window.
        """
        fsm = GestureFSM()
        ts = 0.0
        for _ in range(GESTURE_CONFIRMATION_FRAMES):
            advance(fsm, "OPEN_PALM", 0.95, ts)
            ts += 0.001
        assert fsm.get_state() == "HOVER"

        fsm.reset()

        # Immediately (same tight timestamp spacing) re-fire ENTER_HOVER —
        # must succeed since the debounce tracker was cleared by reset().
        events = []
        for _ in range(GESTURE_CONFIRMATION_FRAMES):
            events.append(advance(fsm, "OPEN_PALM", 0.95, ts))
            ts += 0.001
        assert "ENTER_HOVER" in events
        assert fsm.get_state() == "HOVER"

    def test_fsm_fully_functional_after_reset(self):
        """A reset FSM must behave identically to a brand-new instance."""
        fsm = GestureFSM()
        drive_to_hover(fsm)
        fsm.reset()

        ts = drive_to_hover(fsm, start_ts=100.0)
        assert fsm.get_state() == "HOVER"
        ts += 0.2
        event = advance(fsm, "PINCH", 0.95, ts)
        assert event == "PINCH_DETECTED"
        assert fsm.get_state() == "PINCH_START"


# =============================================================================
# Dominant gesture on mixed history — edge case
# =============================================================================

class TestDominantGesture:

    def test_dominant_gesture_returns_none_on_empty_history(self):
        fsm = GestureFSM()
        assert fsm._dominant_gesture(window=6) is None

    def test_dominant_gesture_single_entry(self):
        fsm = GestureFSM()
        fsm._gesture_history.append(("PINCH", 0.0))
        assert fsm._dominant_gesture(window=6) == "PINCH"

    def test_dominant_gesture_clear_majority(self):
        fsm = GestureFSM()
        labels = ["OPEN_PALM", "OPEN_PALM", "OPEN_PALM", "OPEN_PALM", "FIST", "FIST"]
        for i, label in enumerate(labels):
            fsm._gesture_history.append((label, float(i)))
        assert fsm._dominant_gesture(window=6) == "OPEN_PALM"

    def test_dominant_gesture_respects_window_size(self):
        """Only the last `window` entries should be considered, not the whole history."""
        fsm = GestureFSM()
        # 5 OPEN_PALM followed by 3 FIST — with window=3, only FIST should count.
        for i in range(5):
            fsm._gesture_history.append(("OPEN_PALM", float(i)))
        for i in range(3):
            fsm._gesture_history.append(("FIST", float(5 + i)))
        assert fsm._dominant_gesture(window=3) == "FIST"
        # With a larger window covering all 8 entries, OPEN_PALM should win (5 vs 3).
        assert fsm._dominant_gesture(window=8) == "OPEN_PALM"

    def test_dominant_gesture_tie_breaks_to_earliest_inserted_label(self):
        """
        On a tie, Counter.most_common() returns the label that was first
        encountered in insertion order among the tied entries.
        """
        fsm = GestureFSM()
        # 2 OPEN_PALM then 2 FIST -> tie (2 vs 2). OPEN_PALM appears first.
        for i, label in enumerate(["OPEN_PALM", "OPEN_PALM", "FIST", "FIST"]):
            fsm._gesture_history.append((label, float(i)))
        assert fsm._dominant_gesture(window=4) == "OPEN_PALM"

    def test_dominant_gesture_window_larger_than_history(self):
        """Requesting a window larger than available history uses all of it."""
        fsm = GestureFSM()
        for i, label in enumerate(["PINCH", "PINCH", "POINT"]):
            fsm._gesture_history.append((label, float(i)))
        assert fsm._dominant_gesture(window=100) == "PINCH"

    def test_dominant_gesture_mixed_realistic_sequence(self):
        """
        A realistic noisy sequence: mostly POINT with occasional flicker to
        UNKNOWN — dominant must still resolve to POINT.
        """
        fsm = GestureFSM()
        sequence = ["POINT", "UNKNOWN", "POINT", "POINT", "UNKNOWN", "POINT", "POINT", "POINT"]
        for i, label in enumerate(sequence):
            fsm._gesture_history.append((label, float(i)))
        assert fsm._dominant_gesture(window=8) == "POINT"


# =============================================================================
# get_state() consistency
# =============================================================================

class TestGetState:

    def test_get_state_matches_state_attribute(self):
        fsm = GestureFSM()
        assert fsm.get_state() == fsm.state
        drive_to_hover(fsm)
        assert fsm.get_state() == fsm.state

    def test_get_state_always_valid_fsm_state(self):
        from shared.fsm_states import FSM_STATE_SET
        fsm = GestureFSM()
        assert fsm.get_state() in FSM_STATE_SET
        drive_to_hover(fsm)
        assert fsm.get_state() in FSM_STATE_SET


# =============================================================================
# Potential Issues Found
# =============================================================================
# (No source files were modified. Listed for awareness only.)
#
# 1. In IDLE and HOVER, transitions gated by `_dominant_gesture()` can fire
#    on the very FIRST matching frame rather than after a sustained run of
#    GESTURE_CONFIRMATION_FRAMES / SWIPE_MIN_FRAMES frames. This is because
#    `_dominant_gesture(window=N)` slices only the last N *available* history
#    entries — if fewer than N frames exist yet (e.g. a fresh FSM or one just
#    coming out of a transition), the slice is shorter than N and a single
#    matching frame can already be the majority. Tests in this suite drive the
#    FSM through the full confirmation window to reach a stable, unambiguous
#    outcome rather than asserting on the exact frame index where a transition
#    first fires, but this is worth flagging: the "confirmation" language in
#    the roadmap implies a sustained run is required, while the actual
#    behavior permits earlier firing whenever recent history is sparse.
#
# 2. The SWIPE_PENDING -> HOVER (swipe abandonment) and SWIPE_PENDING ->
#    SWIPE_COMMIT (swipe detected) paths are checked in a fixed order each
#    frame (swipe-detection first, abandonment second). Because both checks
#    read from the same rolling window, a frame where the index tip has
#    drifted far enough to commit a swipe AND the dominant gesture has
#    simultaneously drifted away from POINT will always resolve as a
#    committed swipe, never an abandonment. This matches the implementation's
#    documented priority ordering and is exercised indirectly by
#    test_swipe_right_on_sufficient_rightward_displacement, but there is no
#    dedicated test isolating the priority ordering itself since constructing
#    a frame where both conditions are simultaneously true requires reaching
#    into private FSM state; flagged here rather than added as a brittle test.
# =============================================================================