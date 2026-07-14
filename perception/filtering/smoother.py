# =============================================================================
# perception/filtering/smoother.py
# AirSign — Landmark Smoother
# -----------------------------------------------------------------------------
# Applies a OneEuroFilter across all 21 MediaPipe hand landmark coordinates,
# managing a lazy pool of up to 63 independent filter instances (21 landmarks
# × 3 axes: x, y, z). Also computes the screen-space cursor position from the
# filtered index fingertip coordinate (landmark index 8).
#
# Design contract:
#   - Never imports MediaPipe types. Input is plain Python lists of tuples.
#   - Filters are instantiated on first use (lazy) — zero overhead for axes
#     that are never exercised (e.g. z-axis if only x/y are needed).
#   - reset() cascades to every active filter instance simultaneously.
#   - update_params() hot-swaps mincutoff and beta on all active filters,
#     taking effect on the very next smooth() call.
#   - get_cursor_position() is the sole coordinate mapping entry point for
#     InferenceThread — it normalises the active-zone subspace to full screen
#     bounds and applies sensitivity scaling.
#
# Threading:
#   One LandmarkSmoother instance is owned exclusively by InferenceThread.
#   Do not share an instance across threads.
#
# Usage::
#
#   smoother = LandmarkSmoother(freq=30.0, mincutoff=1.0, beta=0.007)
#
#   # Per-frame call inside InferenceThread:
#   landmarks_raw      = [(lm.x, lm.y, lm.z) for lm in results.hand_landmarks]
#   landmarks_filtered = smoother.smooth(landmarks_raw, timestamp=time.monotonic())
#   cursor_xy          = smoother.get_cursor_position(
#                            landmarks_filtered,
#                            sensitivity=config["tracking"]["sensitivity"],
#                            screen_w=1920, screen_h=1080,
#                            active_zone=config["tracking"]["active_zone"],
#                        )
#
#   # On hand disappearance:
#   smoother.reset()
# =============================================================================

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

from perception.filtering.one_euro import OneEuroFilter
from shared.constants import (
    CURSOR_LANDMARK_INDEX,
    DEFAULT_FILTER_BETA,
    DEFAULT_FILTER_DCUTOFF,
    DEFAULT_FILTER_MINCUTOFF,
    DEFAULT_PINCH_TOLERANCE,
    DEFAULT_SENSITIVITY,
    DISTANCE_EPSILON,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    TARGET_FPS,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A single landmark as a 3-tuple of normalised floats (x, y, z) ∈ [0, 1].
Landmark  = Tuple[float, float, float]

# The full 21-landmark array produced / consumed by smooth().
LandmarkList = List[Landmark]

# Pool key: (landmark_index ∈ [0,20], axis_index ∈ {0,1,2})
_PoolKey = Tuple[int, int]

# Axis index → human-readable name (for debug / repr only)
_AXIS_NAME: Dict[int, str] = {0: "x", 1: "y", 2: "z"}

# Number of landmarks guaranteed by MediaPipe Hands.
_NUM_LANDMARKS: int = 21

# Number of spatial axes per landmark.
_NUM_AXES: int = 3


# =============================================================================
# LandmarkSmoother
# =============================================================================

class LandmarkSmoother:
    """
    Per-landmark, per-axis One Euro Filter pool for MediaPipe hand landmarks.

    Manages a dictionary of up to 63 ``OneEuroFilter`` instances keyed by
    ``(landmark_index, axis_index)``. Filters are created on the first access
    of each (landmark, axis) pair and reused for every subsequent frame.

    Attributes:
        freq      (float): Nominal sampling frequency in Hz.
        mincutoff (float): Current minimum cutoff frequency for all filters.
        beta      (float): Current speed coefficient for all filters.
        dcutoff   (float): Derivative filter cutoff (fixed; rarely changed).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        freq: float       = TARGET_FPS,
        mincutoff: float  = DEFAULT_FILTER_MINCUTOFF,
        beta: float       = DEFAULT_FILTER_BETA,
        dcutoff: float    = DEFAULT_FILTER_DCUTOFF,
    ) -> None:
        """
        Initialise the smoother with filter parameters.

        No ``OneEuroFilter`` instances are created here — they are
        instantiated lazily on the first ``smooth()`` call for each axis.

        Args:
            freq:      Nominal sampling rate in Hz. Should match the camera
                       target FPS (``TARGET_FPS = 30``). Used to initialise
                       each filter and to fall back when no timestamp is given.
            mincutoff: Minimum cutoff frequency (Hz) at zero velocity.
                       Controls jitter at rest. Forwarded to all filters.
            beta:      Speed coefficient. Controls lag on fast motion.
                       Forwarded to all filters.
            dcutoff:   Derivative low-pass cutoff (Hz). Leave at 1.0.
                       Forwarded to all filters; rarely changed at runtime.

        Raises:
            ValueError: If any parameter violates ``OneEuroFilter`` constraints
                        (freq > 0, mincutoff > 0, dcutoff > 0, beta >= 0).
        """
        # Validate by constructing a throwaway filter — reuses OneEuroFilter's
        # own validation logic without duplicating the checks here.
        _probe = OneEuroFilter(
            freq=freq,
            mincutoff=mincutoff,
            beta=beta,
            dcutoff=dcutoff,
        )
        del _probe

        self.freq: float      = freq
        self.mincutoff: float = mincutoff
        self.beta: float      = beta
        self.dcutoff: float   = dcutoff

        # Lazy filter pool: keyed by (landmark_index, axis_index).
        # Populated on first access in _get_filter().
        self._filters: Dict[_PoolKey, OneEuroFilter] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_filter(self, landmark_idx: int, axis_idx: int) -> OneEuroFilter:
        """
        Return the filter for ``(landmark_idx, axis_idx)``, creating it on
        first access with the current parameter set.

        Args:
            landmark_idx: MediaPipe landmark index, 0–20.
            axis_idx:     Axis index: 0 = x, 1 = y, 2 = z.

        Returns:
            The ``OneEuroFilter`` instance for this (landmark, axis) pair.
        """
        key: _PoolKey = (landmark_idx, axis_idx)
        if key not in self._filters:
            self._filters[key] = OneEuroFilter(
                freq      = self.freq,
                mincutoff = self.mincutoff,
                beta      = self.beta,
                dcutoff   = self.dcutoff,
            )
        return self._filters[key]

    # ------------------------------------------------------------------
    # Primary smoothing entry point
    # ------------------------------------------------------------------

    def smooth(
        self,
        landmarks: LandmarkList,
        timestamp: Optional[float] = None,
    ) -> LandmarkList:
        """
        Apply per-axis One Euro filtering to all 21 hand landmarks.

        Iterates over every landmark and every axis, routing each scalar
        through its dedicated ``OneEuroFilter`` instance. The output list
        has the same structure as the input — a list of 21 ``(x, y, z)``
        tuples — but with each coordinate independently smoothed.

        Args:
            landmarks: 21-item list of ``(x, y, z)`` tuples. All values
                       should be normalised to [0.0, 1.0] by MediaPipe.
                       Passing a list with length != 21 raises ``ValueError``.
            timestamp: Monotonic timestamp in seconds (``time.monotonic()``).
                       Providing accurate timestamps enables dynamic frequency
                       adjustment for dropped frames. Pass ``None`` to use
                       the nominal ``self.freq`` (less accurate on frame drops
                       but functionally correct).

        Returns:
            21-item list of ``(x, y, z)`` tuples with each axis filtered.
            The returned list is always a fresh allocation — the input is
            never mutated.

        Raises:
            ValueError: If ``landmarks`` does not contain exactly 21 entries,
                        or if any coordinate is non-finite.
        """
        if len(landmarks) != _NUM_LANDMARKS:
            raise ValueError(
                f"Expected exactly {_NUM_LANDMARKS} landmarks, "
                f"got {len(landmarks)}. Ensure MediaPipe max_num_hands=1 "
                f"and that landmarks are extracted before calling smooth()."
            )

        filtered: LandmarkList = []

        for lm_idx, landmark in enumerate(landmarks):
            if len(landmark) != _NUM_AXES:
                raise ValueError(
                    f"Landmark {lm_idx} has {len(landmark)} axes; "
                    f"expected {_NUM_AXES} (x, y, z)."
                )

            smoothed_axes: list[float] = []

            for ax_idx, raw_value in enumerate(landmark):
                if not math.isfinite(raw_value):
                    raise ValueError(
                        f"Non-finite value at landmark {lm_idx}, "
                        f"axis {_AXIS_NAME.get(ax_idx, ax_idx)}: {raw_value!r}. "
                        f"Check MediaPipe output for degenerate frames."
                    )

                f = self._get_filter(lm_idx, ax_idx)
                smoothed_axes.append(f.filter(raw_value, timestamp=timestamp))

            filtered.append(
                (smoothed_axes[0], smoothed_axes[1], smoothed_axes[2])
            )

        return filtered

    # ------------------------------------------------------------------
    # Cursor position mapping
    # ------------------------------------------------------------------

    def get_cursor_position(
        self,
        filtered_landmarks: LandmarkList,
        sensitivity: float,
        screen_w: int,
        screen_h: int,
        active_zone: dict,
    ) -> Tuple[int, int]:
        """
        Map the filtered index fingertip coordinate to absolute screen pixels.

        Uses landmark ``CURSOR_LANDMARK_INDEX`` (index 8 = INDEX_FINGER_TIP)
        as the cursor control point. The active zone defines a sub-rectangle
        of the camera frame within which hand positions are valid; coordinates
        are normalised relative to this sub-rectangle before screen mapping,
        so that users do not need to move their hand to the extreme edges of
        the frame to reach screen corners.

        Coordinate pipeline:
            1. Read ``(lm_x, lm_y)`` from filtered landmark 8 (normalised
               [0, 1] relative to full frame).
            2. Remap into the active zone subspace:
                   ``norm_x = (lm_x - zone_x1) / (zone_x2 - zone_x1)``
                   ``norm_y = (lm_y - zone_y1) / (zone_y2 - zone_y1)``
            3. Clamp to [0.0, 1.0] so hands outside the zone reach the
               screen edge rather than wrapping or going negative.
            4. Apply sensitivity scaling:
                   ``screen_x = int(norm_x * screen_w * sensitivity)``
                   ``screen_y = int(norm_y * screen_h * sensitivity)``
            5. Clamp final pixel values to [0, screen_w-1] / [0, screen_h-1].

        Args:
            filtered_landmarks: 21-item output from ``smooth()``. Must have
                                 exactly 21 entries.
            sensitivity:        Scalar multiplier in (0, 1] that scales the
                                 effective cursor range. Values < 1.0 shrink
                                 the reachable area (requires more hand travel
                                 per pixel). Read from ``config.tracking.
                                 sensitivity`` (default ``DEFAULT_SENSITIVITY``).
            screen_w:           Primary monitor width in pixels.
            screen_h:           Primary monitor height in pixels.
            active_zone:        Dict with keys ``x1``, ``y1``, ``x2``, ``y2``
                                 as normalised floats [0, 1] defining the
                                 camera sub-rectangle used for mapping.
                                 Example: ``{"x1": 0.05, "y1": 0.05,
                                              "x2": 0.95, "y2": 0.95}``

        Returns:
            ``(screen_x, screen_y)`` as a tuple of non-negative integers,
            clamped to the screen bounds.

        Raises:
            ValueError: If ``filtered_landmarks`` does not have 21 entries,
                        or if ``active_zone`` is missing required keys,
                        or if the zone width/height is zero (degenerate zone).
            ValueError: If ``sensitivity`` is not in (0, 1] or screen
                        dimensions are not positive integers.
        """
        # --- Parameter validation ------------------------------------------
        if len(filtered_landmarks) != _NUM_LANDMARKS:
            raise ValueError(
                f"filtered_landmarks must have {_NUM_LANDMARKS} entries, "
                f"got {len(filtered_landmarks)}."
            )

        required_zone_keys = ("x1", "y1", "x2", "y2")
        missing = [k for k in required_zone_keys if k not in active_zone]
        if missing:
            raise ValueError(
                f"active_zone is missing required keys: {missing}. "
                f"Expected all of: {required_zone_keys}."
            )

        if sensitivity <= 0.0 or sensitivity > 2.0:
            raise ValueError(
                f"sensitivity must be in (0, 2.0], got {sensitivity!r}. "
                f"Typical range is 0.5 – 1.2."
            )

        if screen_w <= 0 or screen_h <= 0:
            raise ValueError(
                f"screen_w and screen_h must be positive integers, "
                f"got screen_w={screen_w}, screen_h={screen_h}."
            )

        # --- Extract active zone bounds ------------------------------------
        zone_x1: float = float(active_zone["x1"])
        zone_y1: float = float(active_zone["y1"])
        zone_x2: float = float(active_zone["x2"])
        zone_y2: float = float(active_zone["y2"])

        zone_w = zone_x2 - zone_x1
        zone_h = zone_y2 - zone_y1

        if abs(zone_w) < DISTANCE_EPSILON:
            raise ValueError(
                f"active_zone x-width is effectively zero "
                f"(x1={zone_x1}, x2={zone_x2}). "
                f"Check config.json tracking.active_zone."
            )
        if abs(zone_h) < DISTANCE_EPSILON:
            raise ValueError(
                f"active_zone y-height is effectively zero "
                f"(y1={zone_y1}, y2={zone_y2}). "
                f"Check config.json tracking.active_zone."
            )

        # --- Read cursor landmark (INDEX_FINGER_TIP = landmark 8) ----------
        tip = filtered_landmarks[CURSOR_LANDMARK_INDEX]
        lm_x: float = tip[0]   # normalised [0, 1] in frame space
        lm_y: float = tip[1]   # normalised [0, 1] in frame space

        # --- Step 2: Remap into active zone subspace -----------------------
        # Position relative to the active zone's top-left corner, scaled to
        # span [0, 1] across the zone's width and height respectively.
        norm_x = (lm_x - zone_x1) / zone_w
        norm_y = (lm_y - zone_y1) / zone_h

        # --- Step 3: Clamp to [0.0, 1.0] -----------------------------------
        # Hands outside the zone are clamped to the nearest screen edge
        # rather than wrapping or producing out-of-range pixel coordinates.
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        # --- Step 4: Scale to screen space with sensitivity ----------------
        raw_screen_x = norm_x * screen_w * sensitivity
        raw_screen_y = norm_y * screen_h * sensitivity

        # --- Step 5: Clamp to screen bounds and convert to int -------------
        screen_x = int(max(0.0, min(float(screen_w - 1), raw_screen_x)))
        screen_y = int(max(0.0, min(float(screen_h - 1), raw_screen_y)))

        return (screen_x, screen_y)

    # ------------------------------------------------------------------
    # Parameter hot-swap
    # ------------------------------------------------------------------

    def update_params(
        self,
        mincutoff: Optional[float] = None,
        beta: Optional[float]      = None,
        dcutoff: Optional[float]   = None,
    ) -> None:
        """
        Hot-swap filter parameters across the entire active filter pool.

        Updates ``self.mincutoff``, ``self.beta``, and/or ``self.dcutoff``
        and forwards the change to every ``OneEuroFilter`` currently in the
        pool. New filters created after this call inherit the updated values.
        Filter states (smoothed values) are preserved — there is no
        discontinuity in the output signal.

        Called by ``InferenceThread.update_filter_params()`` when the user
        adjusts the mincutoff or beta sliders in Dev B's settings panel.

        Args:
            mincutoff: New minimum cutoff frequency in Hz, or ``None`` to
                       leave unchanged. Must be > 0 if provided.
            beta:      New speed coefficient, or ``None`` to leave unchanged.
                       Must be >= 0 if provided.
            dcutoff:   New derivative filter cutoff in Hz, or ``None`` to
                       leave unchanged. Must be > 0 if provided. Rarely
                       changed at runtime.

        Raises:
            ValueError: Propagated from ``OneEuroFilter.update_params()`` if
                        any provided value fails its positivity constraint.
        """
        # Validate ALL provided values up front before mutating any state.
        # This mirrors OneEuroFilter.__init__ constraints and ensures that a
        # partially-valid call (e.g. valid mincutoff + invalid beta) does not
        # leave self in an inconsistent state. Validation is done here rather
        # than relying solely on OneEuroFilter.update_params() because the pool
        # may be empty (no smooth() called yet), in which case there are no
        # filter instances to delegate validation to.
        if mincutoff is not None and mincutoff <= 0.0:
            raise ValueError(
                f"mincutoff must be > 0, got {mincutoff!r}."
            )
        if beta is not None and beta < 0.0:
            raise ValueError(
                f"beta must be >= 0, got {beta!r}."
            )
        if dcutoff is not None and dcutoff <= 0.0:
            raise ValueError(
                f"dcutoff must be > 0, got {dcutoff!r}."
            )

        # Update stored defaults (so newly-created filters inherit them).
        if mincutoff is not None:
            self.mincutoff = mincutoff
        if beta is not None:
            self.beta = beta
        if dcutoff is not None:
            self.dcutoff = dcutoff

        # Forward to all existing filter instances in the pool.
        for f in self._filters.values():
            f.update_params(
                mincutoff = mincutoff,
                beta      = beta,
                dcutoff   = dcutoff,
            )

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset all active filters in the pool to their cold-start state.

        Calls ``OneEuroFilter.reset()`` on every filter currently in
        ``self._filters``. The pool itself is retained (filter instances are
        reused), but their internal state (smoothed value, last timestamp)
        is cleared so the next ``smooth()`` call is treated as a first sample.

        Filter parameters (mincutoff, beta, dcutoff) are preserved on all
        instances.

        Call this whenever the tracked signal undergoes a discontinuity:
        - ``HAND_LOST`` FSM event (hand disappeared from frame)
        - Camera source change or reconnect
        - Application resume from pause / minimise

        After ``reset()``:
        - ``is_initialised()`` returns ``False`` on all pool members.
        - The next ``smooth()`` call outputs the raw input unchanged for
          that frame (identical to cold-start behaviour).
        """
        for f in self._filters.values():
            f.reset()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_filter_count(self) -> int:
        """
        Number of ``OneEuroFilter`` instances currently in the pool.

        Starts at 0 and grows to at most 63 (21 × 3) as new (landmark, axis)
        pairs are accessed. Useful for diagnostics and test assertions.

        Returns:
            Integer count of instantiated filters, 0 – 63.
        """
        return len(self._filters)

    def is_initialised(self) -> bool:
        """
        Return ``True`` if at least one filter in the pool has processed
        at least one sample.

        Returns:
            ``True`` if any filter is initialised, ``False`` if the pool
            is empty or all filters have been reset.
        """
        return any(f.is_initialised() for f in self._filters.values())

    def __repr__(self) -> str:
        return (
            f"LandmarkSmoother("
            f"freq={self.freq}, "
            f"mincutoff={self.mincutoff}, "
            f"beta={self.beta}, "
            f"dcutoff={self.dcutoff}, "
            f"active_filters={self.active_filter_count}/63, "
            f"initialised={self.is_initialised()}"
            f")"
        )