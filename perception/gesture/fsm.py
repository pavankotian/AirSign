# =============================================================================
# perception/gesture/fsm.py
# AirSign — Gesture Finite State Machine
# -----------------------------------------------------------------------------
# Converts a stream of per-frame gesture classifications into discrete,
# debounced FSM state transitions and events. This is the temporal layer
# that sits between GestureClassifier (per-frame labels) and ActionThread
# (OS action dispatch) — it is what turns "I saw PINCH for one frame" into
# "the user deliberately clicked."
#
# Design contract:
#   - Deterministic: no wall-clock reads inside this module. Every method
#     that needs "now" receives it as an explicit `timestamp` parameter
#     (monotonic seconds), exactly like OneEuroFilter and LandmarkSmoother.
#   - update() is called once per frame and returns at most one event
#     string per call, or None if no transition occurred.
#   - All states returned by get_state() are members of FSM_STATES.
#   - All events returned by update() are members of FSM_EVENTS, or None.
#   - Uses collections.Counter for dominant-gesture computation rather than
#     manual tally loops.
#   - Zero MediaPipe imports. Operates purely on the feature dict produced
#     by extract_features() and the (label, confidence) tuple produced by
#     GestureClassifier.classify().
#
# Threading:
#   One GestureFSM instance is owned exclusively by InferenceThread and
#   called once per processed frame. Do not share an instance across threads.
# =============================================================================

from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Dict, Optional, Tuple

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
from shared.fsm_states import FSM_EVENTS, FSM_STATES

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FeatureDict = Dict[str, float]

# History buffer holds (gesture_label, timestamp) pairs. Sized to cover the
# largest window any handler queries — SWIPE_MIN_FRAMES (8) — with a small
# margin (+4) so dominant-gesture queries never run against a starved buffer
# immediately after a state transition.
_HISTORY_MAXLEN: int = SWIPE_MIN_FRAMES + 4


# =============================================================================
# GestureFSM
# =============================================================================

class GestureFSM:
    """
    Deterministic finite state machine for temporal gesture sequencing.

    Maintains a rolling history of recent per-frame gesture classifications
    and applies the AirSign state transition table (see module docstring and
    individual handler methods below) to produce debounced, high-confidence
    FSM events. This is the sole authority on ``fsm_state`` and ``fsm_event``
    in the ``RESULT_SCHEMA`` dict pushed to ``result_queue``.

    Attributes:
        state (str): Current FSM state. Always a member of ``FSM_STATES``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """
        Initialise the FSM in its entry state (``IDLE``) with empty history.
        """
        self.state: str = "IDLE"

        # Rolling buffer of (gesture_label, timestamp) tuples, most recent last.
        self._gesture_history: Deque[Tuple[str, float]] = deque(
            maxlen=_HISTORY_MAXLEN
        )

        # Monotonic timestamp of the most recent transition into self.state.
        self._state_entry_time: float = 0.0

        # Timestamp of the most recent PINCH_RELEASED-family release, used
        # for double-pinch (tap-tap → right-click) detection.
        self._last_pinch_release_time: float = float("-inf")

        # Index fingertip (x, y) captured at the moment SWIPE_PENDING is
        # entered. Trajectory displacement is measured relative to this point.
        self._swipe_start_pos: Optional[Tuple[float, float]] = None

        # Consecutive frames with gesture == "NONE". Drives the global
        # HAND_LOST transition once it reaches HAND_LOST_FRAMES.
        self._no_hand_frame_count: int = 0

        # Debounce tracker: event string → timestamp it last fired at.
        self._prev_event_times: Dict[str, float] = {}

        # Per-state handler dispatch. Built once so update() stays short.
        self._state_handlers = {
            "IDLE":          self._handle_idle,
            "HOVER":         self._handle_hover,
            "PINCH_START":   self._handle_pinch_start,
            "PINCH_HELD":    self._handle_pinch_held,
            "SCROLL_MODE":   self._handle_scroll_mode,
            "SWIPE_PENDING": self._handle_swipe_pending,
            # SWIPE_COMMIT is handled unconditionally at the top of update()
            # since it is a one-frame auto-exit state, not a decision state.
        }

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def update(
        self,
        gesture: str,
        confidence: float,
        features: FeatureDict,
        timestamp: float,
    ) -> Optional[str]:
        """
        Advance the FSM by one frame and return the event fired, if any.

        This is the sole method called by ``InferenceThread`` once per
        processed frame. Internally this:
            1. Appends the frame to the rolling gesture history.
            2. Updates the consecutive no-hand frame counter.
            3. Auto-exits ``SWIPE_COMMIT`` (always, unconditionally — it is
               a one-frame state).
            4. Checks the global ``HAND_LOST`` condition (overrides any
               active mode).
            5. Dispatches to the handler for the current state.
            6. Gates the resulting event through the debounce window.

        Args:
            gesture:    Current-frame gesture label, e.g. from
                        ``GestureClassifier.classify()``. Should be a member
                        of ``GESTURE_LABELS``, including ``"NONE"`` when no
                        hand is detected.
            confidence: Classifier confidence for ``gesture``, in [0.0, 1.0].
                        Used directly by the ``HOVER`` → ``PINCH_START``
                        transition, which requires immediate (non-windowed)
                        confirmation.
            features:   Output dict of ``extract_features()`` for the current
                        frame. Used for swipe trajectory tracking
                        (``index_tip_x`` / ``index_tip_y``). May be an empty
                        dict on frames with no hand detected; ``features`` is
                        only read when a hand is present.
            timestamp:  Monotonic timestamp in seconds for this frame
                        (``time.monotonic()``). Drives all timing-based
                        transitions (``PINCH_HOLD_MS``, ``DOUBLE_PINCH_MS``,
                        ``GESTURE_DEBOUNCE_MS``).

        Returns:
            The FSM event string that fired this frame (a member of
            ``FSM_EVENTS``), or ``None`` if no transition occurred or the
            resulting event was suppressed by debounce.
        """
        self._gesture_history.append((gesture, timestamp))
        self._update_hand_lost_counter(gesture)

        # ── 1. SWIPE_COMMIT is a one-frame auto-exit state ───────────────
        if self.state == "SWIPE_COMMIT":
            event = self._transition("HOVER", "ENTER_HOVER", timestamp)
            return self._gate(event, timestamp)

        # ── 2. Global HAND_LOST — overrides any active mode ──────────────
        if self.state != "IDLE" and self._no_hand_frame_count >= HAND_LOST_FRAMES:
            event = self._transition("IDLE", "HAND_LOST", timestamp)
            self._no_hand_frame_count = 0
            return self._gate(event, timestamp)

        # ── 3. Per-state handler dispatch ─────────────────────────────────
        handler = self._state_handlers.get(self.state)
        event = (
            handler(gesture, confidence, features, timestamp)
            if handler is not None
            else None
        )

        return self._gate(event, timestamp)

    # ------------------------------------------------------------------
    # Per-state handlers
    # ------------------------------------------------------------------

    def _handle_idle(
        self, gesture: str, confidence: float, features: FeatureDict, timestamp: float
    ) -> Optional[str]:
        """
        ``IDLE`` → ``HOVER``       on confirmed ``OPEN_PALM``.
        ``IDLE`` → ``PINCH_START`` on confirmed ``PINCH``.

        Both use the windowed ``_dominant_gesture()`` confirmation (not the
        raw per-frame gesture) to avoid entering an active mode on a single
        noisy classification.
        """
        dominant = self._dominant_gesture(window=GESTURE_CONFIRMATION_FRAMES)

        if dominant == "OPEN_PALM":
            return self._transition("HOVER", "ENTER_HOVER", timestamp)

        if dominant == "PINCH":
            return self._transition("PINCH_START", "PINCH_DETECTED", timestamp)

        return None

    def _handle_hover(
        self, gesture: str, confidence: float, features: FeatureDict, timestamp: float
    ) -> Optional[str]:
        """
        ``HOVER`` → ``PINCH_START``   immediately on ``PINCH`` with
                                       sufficient per-frame confidence.
        ``HOVER`` → ``SCROLL_MODE``   on confirmed ``FIST``.
        ``HOVER`` → ``SWIPE_PENDING`` on confirmed ``POINT`` (requires the
                                       longer ``SWIPE_MIN_FRAMES`` window,
                                       since accidental swipe entry is more
                                       disruptive than a missed one).
        """
        if gesture == "PINCH" and confidence >= MIN_ACTION_CONFIDENCE:
            return self._transition("PINCH_START", "PINCH_DETECTED", timestamp)

        if self._dominant_gesture(window=GESTURE_CONFIRMATION_FRAMES) == "FIST":
            return self._transition("SCROLL_MODE", "FIST_ENTER", timestamp)

        if self._dominant_gesture(window=SWIPE_MIN_FRAMES) == "POINT":
            self._swipe_start_pos = (
                features["index_tip_x"],
                features["index_tip_y"],
            )
            return self._transition("SWIPE_PENDING", "POINT_DETECTED", timestamp)

        return None

    def _handle_pinch_start(
        self, gesture: str, confidence: float, features: FeatureDict, timestamp: float
    ) -> Optional[str]:
        """
        ``PINCH_START`` → ``HOVER``       if the pinch releases before the
                                           hold threshold (short tap → click,
                                           or double-tap → right-click).
        ``PINCH_START`` → ``PINCH_HELD``  if the pinch is sustained past
                                           ``PINCH_HOLD_MS`` (drag mode).
        """
        if gesture != "PINCH":
            return self._handle_pinch_release(timestamp, check_double=True)

        if self._time_in_state(timestamp) >= PINCH_HOLD_MS:
            return self._transition("PINCH_HELD", "PINCH_HELD", timestamp)

        return None

    def _handle_pinch_held(
        self, gesture: str, confidence: float, features: FeatureDict, timestamp: float
    ) -> Optional[str]:
        """
        ``PINCH_HELD`` → ``HOVER`` when the pinch releases (drag end).

        Does not perform double-pinch detection: a completed drag is a
        distinct, deliberate action and should not be conflated with a
        rapid double-tap gesture.
        """
        if gesture != "PINCH":
            return self._handle_pinch_release(timestamp, check_double=False)

        return None

    def _handle_scroll_mode(
        self, gesture: str, confidence: float, features: FeatureDict, timestamp: float
    ) -> Optional[str]:
        """
        ``SCROLL_MODE`` → ``HOVER`` once ``FIST`` is no longer dominant over
        the short ``STATE_EXIT_FRAMES`` window (fast exit — scrolling should
        stop promptly once the user opens their hand).
        """
        if self._dominant_gesture(window=STATE_EXIT_FRAMES) != "FIST":
            return self._transition("HOVER", "FIST_EXIT", timestamp)

        return None

    def _handle_swipe_pending(
        self, gesture: str, confidence: float, features: FeatureDict, timestamp: float
    ) -> Optional[str]:
        """
        ``SWIPE_PENDING`` → ``SWIPE_COMMIT`` once the index fingertip has
                             travelled past ``SWIPE_THRESHOLD`` in a cardinal
                             direction relative to its position when
                             ``SWIPE_PENDING`` was entered.
        ``SWIPE_PENDING`` → ``HOVER``        (abandonment) if ``POINT`` is no
                             longer dominant over the short
                             ``STATE_EXIT_FRAMES`` window and no swipe was
                             committed.
        """
        swipe_event = self._detect_swipe(features)
        if swipe_event is not None:
            return self._transition("SWIPE_COMMIT", swipe_event, timestamp)

        if self._dominant_gesture(window=STATE_EXIT_FRAMES) != "POINT":
            return self._transition("HOVER", "ENTER_HOVER", timestamp)

        return None

    # ------------------------------------------------------------------
    # Shared release / double-pinch logic
    # ------------------------------------------------------------------

    def _handle_pinch_release(self, timestamp: float, check_double: bool) -> str:
        """
        Shared exit path for both ``PINCH_START`` and ``PINCH_HELD`` when
        the pinch gesture is released, transitioning to ``HOVER``.

        Args:
            timestamp:    Current frame's monotonic timestamp.
            check_double: If ``True`` (called from ``PINCH_START``), checks
                          whether this release occurred within
                          ``DOUBLE_PINCH_MS`` of the previous release and
                          fires ``DOUBLE_PINCH`` instead of
                          ``PINCH_RELEASED`` if so. If ``False`` (called
                          from ``PINCH_HELD``), always fires
                          ``PINCH_RELEASED`` — a completed drag is never
                          reinterpreted as a double-tap.

        Returns:
            The event string passed to ``_transition()`` — either
            ``"PINCH_RELEASED"`` or ``"DOUBLE_PINCH"``.
        """
        event = "PINCH_RELEASED"

        if check_double:
            elapsed_ms = (timestamp - self._last_pinch_release_time) * 1000.0
            if elapsed_ms < DOUBLE_PINCH_MS:
                event = "DOUBLE_PINCH"

        self._last_pinch_release_time = timestamp
        return self._transition("HOVER", event, timestamp)

    # ------------------------------------------------------------------
    # Swipe trajectory detection
    # ------------------------------------------------------------------

    def _detect_swipe(self, features: FeatureDict) -> Optional[str]:
        """
        Check whether the index fingertip has travelled far enough from its
        ``SWIPE_PENDING``-entry position to commit a directional swipe.

        Displacement is measured independently on each axis; whichever axis
        has the larger absolute displacement determines the swipe direction,
        provided that axis also exceeds ``SWIPE_THRESHOLD``. Diagonal
        movement is resolved to the dominant axis rather than producing a
        combined event, since ``FSM_EVENTS`` only defines the four cardinal
        directions.

        Coordinate convention: x increases rightward, y increases downward
        (top-left origin, matching MediaPipe's normalised frame space and
        ``feature_extractor``'s ``index_tip_x`` / ``index_tip_y`` outputs).

        Args:
            features: Output dict of ``extract_features()`` for the current
                      frame. Must contain ``index_tip_x`` and
                      ``index_tip_y``.

        Returns:
            One of ``"SWIPE_LEFT"``, ``"SWIPE_RIGHT"``, ``"SWIPE_UP"``,
            ``"SWIPE_DOWN"`` if the threshold was crossed, or ``None`` if
            not (including the case where ``_swipe_start_pos`` has not yet
            been set).
        """
        if self._swipe_start_pos is None:
            return None

        start_x, start_y = self._swipe_start_pos
        dx = features["index_tip_x"] - start_x
        dy = features["index_tip_y"] - start_y

        if abs(dx) > SWIPE_THRESHOLD and abs(dx) >= abs(dy):
            return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"

        if abs(dy) > SWIPE_THRESHOLD and abs(dy) > abs(dx):
            return "SWIPE_DOWN" if dy > 0 else "SWIPE_UP"

        return None

    # ------------------------------------------------------------------
    # State bookkeeping helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: str, event: str, timestamp: float) -> str:
        """
        Perform the state transition bookkeeping and return the event.

        Sets ``self.state`` and resets the state-entry timer. When the
        destination is ``HOVER``, also clears ``_swipe_start_pos`` — it is
        only meaningful during the ``SWIPE_PENDING`` / ``SWIPE_COMMIT``
        lifecycle and would otherwise carry stale coordinates into whatever
        the next swipe attempt turns out to be.

        Args:
            new_state: Destination state. Must be a member of ``FSM_STATES``.
            event:     Event string to return. Must be a member of
                       ``FSM_EVENTS``.
            timestamp: Current frame's monotonic timestamp.

        Returns:
            ``event``, unchanged — returned for direct use as the handler's
            return value.
        """
        self.state = new_state
        self._state_entry_time = timestamp

        if new_state == "HOVER":
            self._swipe_start_pos = None

        return event

    def _time_in_state(self, timestamp: float) -> float:
        """
        Milliseconds elapsed since the current state was entered.

        Args:
            timestamp: Current frame's monotonic timestamp.

        Returns:
            Elapsed time in milliseconds since ``_state_entry_time``.
        """
        return (timestamp - self._state_entry_time) * 1000.0

    def _dominant_gesture(self, window: int) -> Optional[str]:
        """
        Most frequent gesture label among the last ``window`` frames.

        Uses ``collections.Counter`` over a trailing slice of the gesture
        history rather than a manual tally loop. Ties are broken by
        ``Counter.most_common()``'s stable ordering (first-inserted label
        among tied counts wins), which is deterministic given a fixed input
        sequence.

        Args:
            window: Number of most recent frames to consider. If the history
                    contains fewer than ``window`` frames, all available
                    frames are used.

        Returns:
            The most common gesture label string in the window, or ``None``
            if the history is empty.
        """
        if not self._gesture_history:
            return None

        recent_labels = [
            label for label, _ in list(self._gesture_history)[-window:]
        ]
        return Counter(recent_labels).most_common(1)[0][0]

    def _update_hand_lost_counter(self, gesture: str) -> None:
        """
        Update the consecutive no-hand frame counter.

        Increments on ``gesture == "NONE"`` (no hand detected at all).
        Resets to zero on any other label, including ``"UNKNOWN"`` — an
        unclear gesture still means a hand was present in frame.

        Args:
            gesture: Current-frame gesture label.
        """
        if gesture == "NONE":
            self._no_hand_frame_count += 1
        else:
            self._no_hand_frame_count = 0

    def _check_debounce(self, event: str, timestamp: float) -> bool:
        """
        Check whether ``event`` is permitted to fire given the debounce
        window, updating the last-fired timestamp if permitted.

        Args:
            event:     Event string about to be returned to the caller.
            timestamp: Current frame's monotonic timestamp.

        Returns:
            ``True`` if ``event`` has not fired within ``GESTURE_DEBOUNCE_MS``
            milliseconds (and the last-fired time has been updated to
            ``timestamp``), ``False`` if the event is currently suppressed.
        """
        last_fired = self._prev_event_times.get(event)

        if last_fired is not None:
            elapsed_ms = (timestamp - last_fired) * 1000.0
            if elapsed_ms < GESTURE_DEBOUNCE_MS:
                return False

        self._prev_event_times[event] = timestamp
        return True

    def _gate(self, event: Optional[str], timestamp: float) -> Optional[str]:
        """
        Apply the debounce gate to a handler's returned event.

        The underlying state transition (``self.state``) has already
        occurred by the time this is called — only the *returned event* is
        suppressed on a debounce hit, never the state change itself.

        Args:
            event:     Event returned by a handler, or ``None``.
            timestamp: Current frame's monotonic timestamp.

        Returns:
            ``event`` unchanged if ``event`` is ``None`` or passes the
            debounce check; otherwise ``None``.
        """
        if event is None:
            return None
        return event if self._check_debounce(event, timestamp) else None

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_state(self) -> str:
        """
        Return the current FSM state.

        Returns:
            Current state string, always a member of ``FSM_STATES``.
        """
        return self.state

    def reset(self) -> None:
        """
        Reset the FSM to its entry state with empty history.

        Call this when the perception pipeline restarts or the camera
        reconnects after a prolonged interruption, to discard any stale
        gesture history, in-progress pinch/swipe tracking, and debounce
        timestamps from before the interruption.

        After ``reset()``:
        - ``get_state()`` returns ``"IDLE"``.
        - The gesture history buffer is empty.
        - Any in-progress swipe or pinch tracking is discarded.
        - All debounce timers are cleared, so the next occurrence of any
          event fires immediately regardless of prior timing.
        """
        self.state = "IDLE"
        self._gesture_history.clear()
        self._state_entry_time = 0.0
        self._last_pinch_release_time = float("-inf")
        self._swipe_start_pos = None
        self._no_hand_frame_count = 0
        self._prev_event_times.clear()

    def __repr__(self) -> str:
        return (
            f"GestureFSM("
            f"state={self.state!r}, "
            f"history_len={len(self._gesture_history)}, "
            f"no_hand_frames={self._no_hand_frame_count}"
            f")"
        )