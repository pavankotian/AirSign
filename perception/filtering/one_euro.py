# =============================================================================
# perception/filtering/one_euro.py
# AirSign — One Euro Filter Implementation
# -----------------------------------------------------------------------------
# Adaptive low-pass filter designed for interactive pointer/cursor smoothing.
#
# Reference:
#   Géry Casiez, Nicolas Roussel, Daniel Vogel. "1€ Filter: A Simple
#   Speed-based Low-pass Filter for Noisy Input in Interactive Systems."
#   CHI 2012. https://doi.org/10.1145/2207676.2208639
#
# Design contract:
#   - Self-contained: zero external dependencies beyond math and typing.
#   - Single-axis: one OneEuroFilter instance per coordinate axis.
#   - Thread-safe reads: filter() is NOT thread-safe for concurrent writes;
#     each thread must own its own instance.
#   - LowPassFilter is internal scaffolding — do not import it directly.
#     Use OneEuroFilter as the sole public API.
#
# Usage (in LandmarkSmoother):
#   f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
#   smoothed_x = f.filter(raw_x, timestamp=time.monotonic())
#
# Parameter tuning guide:
#   mincutoff : float
#       Cutoff frequency (Hz) applied at rest (zero velocity).
#       Lower  → more smoothing at rest, but more lag on slow motion.
#       Higher → less smoothing at rest, more jitter visible.
#       Recommended range for 30fps cursor: 0.5 – 2.0
#
#   beta : float
#       Speed coefficient. Scales how aggressively the cutoff rises with
#       estimated velocity. Higher β reduces lag on fast movements.
#       Too high → jitter reappears on slow/medium speed motion.
#       Recommended range for 30fps cursor: 0.001 – 0.02
#
#   dcutoff : float
#       Cutoff for the internal derivative (velocity) low-pass filter.
#       Rarely needs tuning. Leave at 1.0 unless experiencing oscillation.
# =============================================================================

from __future__ import annotations

import math
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Internal building block
# ---------------------------------------------------------------------------

class _LowPassFilter:
    """
    Exponential moving average low-pass filter for a single scalar signal.

    This is an internal implementation detail of OneEuroFilter. It is not
    part of the public API and should not be imported or instantiated directly
    outside this module.

    The filter equation is:
        s_n = α * x_n + (1 − α) * s_{n-1}

    where α ∈ (0, 1] is the smoothing coefficient:
        α close to 1  → fast response, little smoothing
        α close to 0  → heavy smoothing, significant lag

    On the first call (no prior sample), the filter is initialised with the
    input value so there is no cold-start transient.
    """

    __slots__ = ("_s", "_initialised")

    def __init__(self) -> None:
        # _s holds the current filter state (last smoothed value).
        # None signals "not yet initialised" — distinguished from 0.0.
        self._s: Optional[float] = None
        self._initialised: bool = False

    # ------------------------------------------------------------------
    # Core filter step
    # ------------------------------------------------------------------

    def filter(self, x: float, alpha: float) -> float:
        """
        Apply one filter step and return the smoothed value.

        On the very first call the output equals the input exactly — there
        is no warm-up artifact or initial-value assumption.

        Args:
            x:     Raw input sample (any finite float).
            alpha: Smoothing coefficient for this step, in (0.0, 1.0].
                   Computed externally by OneEuroFilter from the current
                   cutoff frequency and sampling interval.

        Returns:
            Smoothed scalar value.

        Raises:
            ValueError: If alpha is outside (0, 1] or x is not finite.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(
                f"alpha must be in (0, 1], got {alpha!r}. "
                "This indicates an invalid cutoff frequency or sampling rate."
            )
        if not math.isfinite(x):
            raise ValueError(
                f"Input x must be a finite float, got {x!r}."
            )

        if not self._initialised:
            self._s = x
            self._initialised = True
            return x

        # Standard EMA step — never raises because alpha and _s are both finite.
        self._s = alpha * x + (1.0 - alpha) * self._s  # type: ignore[operator]
        return self._s  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def last_value(self) -> Optional[float]:
        """
        Return the most recently computed smoothed value, or None if
        the filter has not yet processed any samples.

        Returns:
            Last smoothed value as float, or None before first filter() call.
        """
        return self._s

    def is_initialised(self) -> bool:
        """
        Return True if the filter has processed at least one sample.

        Returns:
            True after the first filter() call, False otherwise.
        """
        return self._initialised

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset the filter to its uninitialised state.

        After calling reset(), the next filter() call will behave identically
        to the very first call on a fresh instance (output = input, no lag).
        """
        self._s = None
        self._initialised = False


# ---------------------------------------------------------------------------
# Primary public class
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """
    One Euro (1€) adaptive low-pass filter for a single scalar coordinate.

    Adapts its cutoff frequency based on the estimated signal velocity:
    - Slow motion  → low cutoff  → strong smoothing  → minimal jitter
    - Fast motion  → high cutoff → weak smoothing    → minimal lag

    One instance must be created per coordinate axis. For 2D cursor tracking,
    instantiate two filters (one for X, one for Y) and call filter() on each
    with the same timestamp.

    For landmark smoothing across all 21 MediaPipe hand landmarks (63 axes
    total), use LandmarkSmoother in perception/filtering/smoother.py which
    manages the filter pool automatically.

    Attributes:
        freq      (float): Nominal sampling frequency in Hz.
        mincutoff (float): Minimum cutoff frequency at zero velocity.
        beta      (float): Speed coefficient for adaptive cutoff scaling.
        dcutoff   (float): Cutoff for the internal derivative filter.

    Example::

        import time
        f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)

        for raw_x in noisy_signal:
            smoothed = f.filter(raw_x, timestamp=time.monotonic())
    """

    __slots__ = (
        "freq",
        "mincutoff",
        "beta",
        "dcutoff",
        "_x_filter",
        "_dx_filter",
        "_last_timestamp",
    )

    def __init__(
        self,
        freq: float,
        mincutoff: float = 1.0,
        beta: float = 0.0,
        dcutoff: float = 1.0,
    ) -> None:
        """
        Initialise the One Euro Filter.

        Args:
            freq:      Nominal sampling frequency in Hz. Used to compute α
                       when no timestamp is provided or on the first sample.
                       Must be > 0. Typical value: 30.0 for webcam input.

            mincutoff: Minimum cutoff frequency (Hz) applied at rest.
                       Controls jitter at slow/zero velocity.
                       Default 1.0. Recommended range: 0.5 – 2.0.

            beta:      Speed coefficient. Scales the cutoff increase with
                       estimated signal velocity. Controls lag on fast motion.
                       Default 0.0 (no speed adaptation). For cursor tracking
                       set to 0.007.

            dcutoff:   Cutoff frequency (Hz) for the internal derivative
                       low-pass filter. Controls stability of the velocity
                       estimate. Default 1.0; rarely needs tuning.

        Raises:
            ValueError: If freq, mincutoff, or dcutoff are not strictly
                        positive, or if beta is negative.
        """
        if freq <= 0.0:
            raise ValueError(f"freq must be > 0, got {freq!r}.")
        if mincutoff <= 0.0:
            raise ValueError(f"mincutoff must be > 0, got {mincutoff!r}.")
        if dcutoff <= 0.0:
            raise ValueError(f"dcutoff must be > 0, got {dcutoff!r}.")
        if beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {beta!r}.")

        self.freq: float      = freq
        self.mincutoff: float = mincutoff
        self.beta: float      = beta
        self.dcutoff: float   = dcutoff

        # Two independent low-pass filters:
        #   _x_filter  — smooths the position signal
        #   _dx_filter — smooths the derivative (velocity) estimate
        self._x_filter: _LowPassFilter  = _LowPassFilter()
        self._dx_filter: _LowPassFilter = _LowPassFilter()

        # Monotonic timestamp of the previous filter() call.
        # None before the first sample; used to compute the dynamic sampling
        # interval te = t_now - t_prev when timestamps are supplied.
        self._last_timestamp: Optional[float] = None

    # ------------------------------------------------------------------
    # Core arithmetic helpers
    # ------------------------------------------------------------------

    def _compute_alpha(self, cutoff: float, te: float) -> float:
        """
        Compute the EMA smoothing coefficient α for a given cutoff frequency
        and sampling interval.

        Derivation from the bilinear transform of an RC low-pass filter:
            τ   = 1 / (2π · f_c)        time constant
            α   = 1 / (1 + τ / t_e)     discrete α, t_e = 1/freq

        Args:
            cutoff: Cutoff frequency in Hz (must be > 0).
            te:     Sampling interval in seconds (must be > 0).

        Returns:
            Smoothing coefficient α ∈ (0, 1].
        """
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def _effective_freq(self, timestamp: Optional[float]) -> float:
        """
        Compute the effective sampling frequency for this step.

        If a timestamp is provided AND a previous timestamp exists, use the
        actual inter-sample interval for accurate frequency estimation. This
        handles variable-rate input (e.g., dropped frames) correctly.

        If no timestamp is available (first call, or timestamp=None), fall
        back to the nominal self.freq. This is safe for the first sample
        because both filters initialise to the input on their first call.

        Args:
            timestamp: Current monotonic timestamp in seconds, or None.

        Returns:
            Effective sampling frequency in Hz for this step.
        """
        if timestamp is not None and self._last_timestamp is not None:
            te = timestamp - self._last_timestamp
            if te > 0.0:
                return 1.0 / te
        return self.freq

    # ------------------------------------------------------------------
    # Primary filter step
    # ------------------------------------------------------------------

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        """
        Filter one input sample and return the smoothed value.

        This is the method called once per frame by InferenceThread /
        LandmarkSmoother. Call it with consecutive samples in chronological
        order; calling it with out-of-order timestamps produces undefined
        (but non-crashing) output.

        The first call always returns x unchanged and seeds the filter state.
        Subsequent calls apply adaptive smoothing.

        Adaptive cutoff computation:
            1. Estimate velocity: dx = (x − x_prev) × freq_eff
            2. Smooth velocity:   edx = LPF(dx, α(dcutoff))
            3. Adaptive cutoff:   f_c = mincutoff + β × |edx|
            4. Smooth position:   x̂  = LPF(x, α(f_c))

        Args:
            x:         Raw input sample. Must be a finite float.
                       Typically a normalized landmark coordinate in [0, 1].

            timestamp: Monotonic timestamp in seconds (time.monotonic()).
                       Providing accurate timestamps enables dynamic frequency
                       adjustment for variable-rate input (dropped frames,
                       jitter in capture loop timing).
                       If None, the nominal self.freq is used — acceptable for
                       steady-rate streams but less accurate on frame drops.

        Returns:
            Smoothed output value. On the first call this equals x exactly.

        Raises:
            ValueError: If x is not finite (propagated from _LowPassFilter).
        """
        # --- 1. Determine effective sampling frequency for this step --------
        freq_eff = self._effective_freq(timestamp)
        te = 1.0 / freq_eff  # sampling interval in seconds

        # --- 2. Estimate and smooth the derivative (velocity) ---------------
        x_prev = self._x_filter.last_value()

        if x_prev is None:
            # First sample — derivative is undefined; seed at zero.
            dx = 0.0
        else:
            # Central difference approximation: velocity = displacement / time
            dx = (x - x_prev) * freq_eff

        alpha_d = self._compute_alpha(self.dcutoff, te)
        edx = self._dx_filter.filter(dx, alpha_d)

        # --- 3. Compute adaptive cutoff from smoothed velocity magnitude ----
        # The cutoff rises linearly with |velocity|, controlled by beta.
        # At rest (edx ≈ 0): cutoff → mincutoff (maximum smoothing)
        # At speed (edx >> 0): cutoff rises (reduces lag)
        adaptive_cutoff = self.mincutoff + self.beta * abs(edx)

        # Guard: cutoff must stay strictly positive for _compute_alpha.
        # In practice beta >= 0 and mincutoff > 0 guarantees this, but
        # defend against floating-point underflow near zero velocity.
        adaptive_cutoff = max(adaptive_cutoff, 1e-10)

        # --- 4. Filter the position signal with the adaptive cutoff ---------
        alpha_x = self._compute_alpha(adaptive_cutoff, te)
        x_hat = self._x_filter.filter(x, alpha_x)

        # --- 5. Advance the timestamp for the next call ---------------------
        if timestamp is not None:
            self._last_timestamp = timestamp

        return x_hat

    # ------------------------------------------------------------------
    # Parameter hot-swap
    # ------------------------------------------------------------------

    def update_params(
        self,
        mincutoff: Optional[float] = None,
        beta: Optional[float] = None,
        dcutoff: Optional[float] = None,
    ) -> None:
        """
        Update filter parameters without resetting internal filter state.

        Called by InferenceThread.update_filter_params() when the user moves
        the mincutoff or beta sliders in Dev B's settings panel. The change
        takes effect on the very next filter() call with no discontinuity in
        the output signal (state is preserved).

        Only provide the parameters you want to change; omit or pass None
        to leave a parameter unchanged.

        Args:
            mincutoff: New minimum cutoff frequency in Hz. Must be > 0 if provided.
            beta:      New speed coefficient. Must be >= 0 if provided.
            dcutoff:   New derivative filter cutoff in Hz. Must be > 0 if provided.
                       Rarely changed at runtime — prefer leaving as None.

        Raises:
            ValueError: If any provided value fails its positivity constraint.

        Example::

            # User moved the beta slider to 0.012
            one_euro_filter.update_params(beta=0.012)

            # User changed both mincutoff and beta simultaneously
            one_euro_filter.update_params(mincutoff=0.8, beta=0.015)
        """
        if mincutoff is not None:
            if mincutoff <= 0.0:
                raise ValueError(
                    f"mincutoff must be > 0, got {mincutoff!r}."
                )
            self.mincutoff = mincutoff

        if beta is not None:
            if beta < 0.0:
                raise ValueError(
                    f"beta must be >= 0, got {beta!r}."
                )
            self.beta = beta

        if dcutoff is not None:
            if dcutoff <= 0.0:
                raise ValueError(
                    f"dcutoff must be > 0, got {dcutoff!r}."
                )
            self.dcutoff = dcutoff

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset the filter to its cold-start state.

        After calling reset(), the next filter() call behaves identically to
        the very first call on a fresh instance: output equals input, no lag,
        no warm-up artifact. The configured parameters (freq, mincutoff, beta,
        dcutoff) are preserved.

        Call this whenever the tracked signal undergoes a discontinuity that
        would make prior filter state misleading:
        - Hand disappears from frame and then reappears (HAND_LOST → ENTER_HOVER)
        - Camera source changes
        - Application resumes from pause/minimise

        In LandmarkSmoother.reset(), this method is forwarded to all 63
        active filter instances simultaneously.
        """
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_timestamp = None

    # ------------------------------------------------------------------
    # Introspection helpers (used by tests and debug overlays)
    # ------------------------------------------------------------------

    def is_initialised(self) -> bool:
        """
        Return True if the filter has processed at least one sample.

        Returns:
            True after the first filter() call, False on a fresh or
            reset instance.
        """
        return self._x_filter.is_initialised()

    def last_value(self) -> Optional[float]:
        """
        Return the most recently smoothed output value.

        Returns:
            Last smoothed float, or None before the first filter() call
            or after reset().
        """
        return self._x_filter.last_value()

    def __repr__(self) -> str:
        return (
            f"OneEuroFilter("
            f"freq={self.freq}, "
            f"mincutoff={self.mincutoff}, "
            f"beta={self.beta}, "
            f"dcutoff={self.dcutoff}, "
            f"initialised={self.is_initialised()}"
            f")"
        )