# =============================================================================
# tests/dev_a/test_one_euro.py
# AirSign — Phase 1 Filtering Test Suite
# -----------------------------------------------------------------------------
# Covers:
#   perception/filtering/one_euro.py  (_LowPassFilter, OneEuroFilter)
#   perception/filtering/smoother.py  (LandmarkSmoother)
#
# Import path used throughout:
#   from perception.filtering.one_euro import OneEuroFilter, _LowPassFilter
#   from perception.filtering.smoother import LandmarkSmoother
#
# All tests are deterministic — no random seeds, no wall-clock timing.
# Monotonic timestamps are synthesised arithmetically at 1/30 s intervals.
# =============================================================================

from __future__ import annotations

import math
import sys
import os

import pytest

# Ensure project root is on sys.path when running from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from perception.filtering.one_euro import OneEuroFilter, _LowPassFilter
from perception.filtering.smoother import LandmarkSmoother
from shared.constants import (
    CURSOR_LANDMARK_INDEX,
    DEFAULT_ACTIVE_ZONE,
    DEFAULT_FILTER_BETA,
    DEFAULT_FILTER_DCUTOFF,
    DEFAULT_FILTER_MINCUTOFF,
    DISTANCE_EPSILON,
    TARGET_FPS,
)


# ---------------------------------------------------------------------------
# Shared test fixtures and helpers
# ---------------------------------------------------------------------------

def make_timestamps(n: int, fps: float = 30.0) -> list[float]:
    """Return a list of n evenly-spaced synthetic monotonic timestamps."""
    return [i / fps for i in range(n)]


def noisy_sine(n: int, amplitude: float = 0.3, noise_amp: float = 0.04) -> list[float]:
    """
    Return a deterministic noisy sine wave of length n.
    Noise is a fixed-amplitude triangle wave (no random) so tests are
    fully reproducible without seeding.
    """
    signal = []
    for i in range(n):
        clean = 0.5 + amplitude * math.sin(i * 2 * math.pi / 30)
        # Triangle-wave noise: deterministic, zero-mean, period=7 frames
        noise = noise_amp * (((i % 7) / 3.5) - 1.0)
        signal.append(max(0.0, min(1.0, clean + noise)))
    return signal


def flat_landmarks(x: float = 0.5, y: float = 0.5, z: float = 0.0) -> list:
    """21-item landmark list with every point at (x, y, z)."""
    return [(x, y, z)] * 21


def mean_abs_diff(seq: list[float]) -> float:
    """Mean absolute first difference of a sequence."""
    if len(seq) < 2:
        return 0.0
    return sum(abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))) / (len(seq) - 1)


# =============================================================================
# _LowPassFilter tests
# =============================================================================

class TestLowPassFilter:

    def test_uninitialised_state_before_first_call(self):
        """Fresh filter reports uninitialised and last_value is None."""
        lpf = _LowPassFilter()
        assert not lpf.is_initialised()
        assert lpf.last_value() is None

    def test_first_call_returns_input_exactly(self):
        """Cold-start: first filter() output must equal the raw input."""
        lpf = _LowPassFilter()
        for x in (0.0, 0.5, 1.0, -3.14, 999.0):
            lpf2 = _LowPassFilter()
            out = lpf2.filter(x, alpha=0.5)
            assert out == x, f"Expected {x}, got {out}"

    def test_is_initialised_after_first_call(self):
        """is_initialised() returns True after one filter() call."""
        lpf = _LowPassFilter()
        lpf.filter(0.42, alpha=0.5)
        assert lpf.is_initialised()

    def test_last_value_tracks_output(self):
        """last_value() matches the most recent filter() return value."""
        lpf = _LowPassFilter()
        out1 = lpf.filter(0.2, alpha=0.3)
        assert lpf.last_value() == out1
        out2 = lpf.filter(0.8, alpha=0.3)
        assert lpf.last_value() == out2

    def test_ema_formula_manual(self):
        """Verify the EMA step: s = α·x + (1-α)·s_prev."""
        lpf = _LowPassFilter()
        lpf.filter(0.0, alpha=0.5)          # initialise: s = 0.0
        out = lpf.filter(1.0, alpha=0.5)    # s = 0.5·1.0 + 0.5·0.0 = 0.5
        assert abs(out - 0.5) < 1e-12

    def test_alpha_one_passes_through(self):
        """alpha=1.0 means no smoothing — output equals input."""
        lpf = _LowPassFilter()
        lpf.filter(0.3, alpha=1.0)
        out = lpf.filter(0.9, alpha=1.0)
        assert out == 0.9

    def test_alpha_near_zero_heavily_smoothed(self):
        """alpha→0 means heavy smoothing — output barely moves."""
        lpf = _LowPassFilter()
        lpf.filter(0.5, alpha=0.001)
        out = lpf.filter(1.0, alpha=0.001)
        # With alpha=0.001 the output should still be very close to 0.5
        assert abs(out - 0.5) < 0.01

    def test_reset_restores_uninitialised_state(self):
        """After reset(), filter behaves as fresh instance."""
        lpf = _LowPassFilter()
        lpf.filter(0.7, alpha=0.5)
        lpf.filter(0.8, alpha=0.5)
        lpf.reset()
        assert not lpf.is_initialised()
        assert lpf.last_value() is None
        # Next call must return input exactly (cold-start)
        out = lpf.filter(0.42, alpha=0.5)
        assert out == 0.42

    def test_invalid_alpha_below_zero_raises(self):
        """alpha <= 0 must raise ValueError."""
        lpf = _LowPassFilter()
        with pytest.raises(ValueError):
            lpf.filter(0.5, alpha=0.0)
        with pytest.raises(ValueError):
            lpf.filter(0.5, alpha=-0.1)

    def test_invalid_alpha_above_one_raises(self):
        """alpha > 1.0 must raise ValueError."""
        lpf = _LowPassFilter()
        with pytest.raises(ValueError):
            lpf.filter(0.5, alpha=1.001)

    def test_non_finite_input_raises(self):
        """Non-finite x values must raise ValueError."""
        lpf = _LowPassFilter()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError):
                lpf.filter(bad, alpha=0.5)


# =============================================================================
# OneEuroFilter — construction and parameter validation
# =============================================================================

class TestOneEuroFilterConstruction:

    def test_default_params_stored(self):
        """Constructor stores the provided parameter values."""
        f = OneEuroFilter(freq=30.0, mincutoff=1.5, beta=0.01, dcutoff=2.0)
        assert f.freq      == 30.0
        assert f.mincutoff == 1.5
        assert f.beta      == 0.01
        assert f.dcutoff   == 2.0

    def test_default_beta_is_zero(self):
        """beta defaults to 0.0 (no speed adaptation) per spec."""
        f = OneEuroFilter(freq=30.0)
        assert f.beta == 0.0

    def test_invalid_freq_raises(self):
        with pytest.raises(ValueError):
            OneEuroFilter(freq=0.0)
        with pytest.raises(ValueError):
            OneEuroFilter(freq=-1.0)

    def test_invalid_mincutoff_raises(self):
        with pytest.raises(ValueError):
            OneEuroFilter(freq=30.0, mincutoff=0.0)
        with pytest.raises(ValueError):
            OneEuroFilter(freq=30.0, mincutoff=-1.0)

    def test_invalid_dcutoff_raises(self):
        with pytest.raises(ValueError):
            OneEuroFilter(freq=30.0, dcutoff=0.0)
        with pytest.raises(ValueError):
            OneEuroFilter(freq=30.0, dcutoff=-0.5)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            OneEuroFilter(freq=30.0, beta=-0.001)

    def test_beta_zero_is_valid(self):
        """beta=0.0 is explicitly valid (disables speed adaptation)."""
        f = OneEuroFilter(freq=30.0, beta=0.0)
        assert f.beta == 0.0

    def test_not_initialised_before_first_call(self):
        f = OneEuroFilter(freq=30.0)
        assert not f.is_initialised()
        assert f.last_value() is None


# =============================================================================
# OneEuroFilter — filter() behaviour
# =============================================================================

class TestOneEuroFilterBehaviour:

    def test_first_call_returns_input_exactly(self):
        """Spec: first filter() call returns x unchanged (cold-start)."""
        for x in (0.0, 0.5, 1.0, 0.123):
            f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
            out = f.filter(x)
            assert out == x, f"First call: expected {x}, got {out}"

    def test_first_call_without_timestamp(self):
        """filter() with timestamp=None works on first call."""
        f = OneEuroFilter(freq=30.0)
        out = f.filter(0.7, timestamp=None)
        assert out == 0.7

    def test_is_initialised_after_first_call(self):
        f = OneEuroFilter(freq=30.0)
        f.filter(0.5)
        assert f.is_initialised()

    def test_last_value_matches_last_output(self):
        f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
        ts = make_timestamps(10)
        out = None
        for i, t in enumerate(ts):
            out = f.filter(float(i) / 9.0, timestamp=t)
        assert f.last_value() == out

    def test_output_smoother_than_input_noisy_sine(self):
        """
        Spec: filter output MAD must be strictly less than input MAD on a
        noisy signal. Uses deterministic triangle-wave noise.
        """
        raw = noisy_sine(120, amplitude=0.3, noise_amp=0.04)
        ts  = make_timestamps(120)
        f   = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
        filtered = [f.filter(x, timestamp=t) for x, t in zip(raw, ts)]
        assert mean_abs_diff(filtered) < mean_abs_diff(raw), (
            f"Filter not smoother: raw_MAD={mean_abs_diff(raw):.5f}, "
            f"filtered_MAD={mean_abs_diff(filtered):.5f}"
        )

    def test_all_outputs_finite(self):
        """All outputs on a clean signal must be finite floats."""
        raw = noisy_sine(300)
        ts  = make_timestamps(300)
        f   = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
        for x, t in zip(raw, ts):
            out = f.filter(x, timestamp=t)
            assert math.isfinite(out), f"Non-finite output for x={x}, t={t}"

    def test_dynamic_frequency_via_timestamps(self):
        """
        Providing accurate timestamps enables dynamic te computation.
        Output must remain finite and smoother than input across 60 frames.
        """
        raw = noisy_sine(60)
        ts  = make_timestamps(60, fps=30.0)
        f   = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
        filtered = [f.filter(x, timestamp=t) for x, t in zip(raw, ts)]
        assert all(math.isfinite(v) for v in filtered)
        assert mean_abs_diff(filtered) < mean_abs_diff(raw)

    def test_timestamp_none_falls_back_to_nominal_freq(self):
        """
        With timestamp=None throughout, filter still produces valid output
        using the nominal freq for all te computations.
        """
        f   = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
        raw = noisy_sine(60)
        filtered = [f.filter(x, timestamp=None) for x in raw]
        assert all(math.isfinite(v) for v in filtered)
        assert mean_abs_diff(filtered) < mean_abs_diff(raw)

    def test_step_input_converges(self):
        """
        On a sudden step from 0 to 1, the filtered output must eventually
        approach 1.0 (within 5% after 90 frames at 30fps).
        """
        f  = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.1)
        ts = make_timestamps(90)
        # Warm up at 0.0
        for t in ts[:30]:
            f.filter(0.0, timestamp=t)
        # Step to 1.0
        out = None
        for t in ts[30:]:
            out = f.filter(1.0, timestamp=t)
        assert out is not None
        assert out > 0.95, f"Did not converge: final output={out:.4f}"

    def test_constant_input_passes_through_after_warmup(self):
        """
        After sufficient warmup on a constant signal, output must equal
        the input to within floating-point precision.
        """
        f  = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
        ts = make_timestamps(200)
        out = None
        for t in ts:
            out = f.filter(0.5, timestamp=t)
        assert abs(out - 0.5) < 1e-6, f"Constant input diverged: out={out}"

    def test_high_beta_reduces_lag_on_fast_motion(self):
        """
        Higher beta → lower lag on fast-moving inputs.
        Filter with beta=0.5 should track a ramp faster than beta=0.0.
        """
        ramp = [i / 59.0 for i in range(60)]
        ts   = make_timestamps(60)

        f_slow = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.0)
        f_fast = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.5)

        out_slow = [f_slow.filter(x, t) for x, t in zip(ramp, ts)]
        out_fast = [f_fast.filter(x, t) for x, t in zip(ramp, ts)]

        # Measure lag as sum of (target - output) over the ramp
        lag_slow = sum(ramp[i] - out_slow[i] for i in range(1, 60))
        lag_fast = sum(ramp[i] - out_fast[i] for i in range(1, 60))
        assert lag_fast < lag_slow, (
            f"High beta did not reduce lag: lag_slow={lag_slow:.4f}, "
            f"lag_fast={lag_fast:.4f}"
        )


# =============================================================================
# OneEuroFilter — reset()
# =============================================================================

class TestOneEuroFilterReset:

    def test_reset_clears_initialised_flag(self):
        f = OneEuroFilter(freq=30.0)
        f.filter(0.5)
        assert f.is_initialised()
        f.reset()
        assert not f.is_initialised()

    def test_reset_clears_last_value(self):
        f = OneEuroFilter(freq=30.0)
        f.filter(0.5)
        f.reset()
        assert f.last_value() is None

    def test_reset_clears_last_timestamp(self):
        f = OneEuroFilter(freq=30.0)
        ts = make_timestamps(10)
        for i, t in enumerate(ts):
            f.filter(float(i) / 9.0, timestamp=t)
        f.reset()
        assert f._last_timestamp is None

    def test_first_call_after_reset_returns_input_exactly(self):
        """Spec: post-reset cold-start — output == input on next call."""
        f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
        ts = make_timestamps(20)
        for i, t in enumerate(ts):
            f.filter(float(i) / 19.0, timestamp=t)
        f.reset()
        out = f.filter(0.99)
        assert out == 0.99, f"Post-reset first call: expected 0.99, got {out}"

    def test_reset_preserves_parameters(self):
        """reset() must not alter freq, mincutoff, beta, or dcutoff."""
        f = OneEuroFilter(freq=25.0, mincutoff=2.0, beta=0.05, dcutoff=1.5)
        f.filter(0.5)
        f.reset()
        assert f.freq      == 25.0
        assert f.mincutoff == 2.0
        assert f.beta      == 0.05
        assert f.dcutoff   == 1.5

    def test_multiple_resets_are_idempotent(self):
        """Calling reset() multiple times must not raise or corrupt state."""
        f = OneEuroFilter(freq=30.0)
        f.reset()
        f.reset()
        assert not f.is_initialised()
        out = f.filter(0.42)
        assert out == 0.42

    def test_filter_works_correctly_after_reset(self):
        """
        After reset, running the filter on a fresh noisy signal should
        produce smoother output than the raw signal (normal operation).
        """
        raw = noisy_sine(90)
        ts  = make_timestamps(90)
        f   = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
        # First pass
        for x, t in zip(raw[:45], ts[:45]):
            f.filter(x, timestamp=t)
        f.reset()
        # Second pass after reset
        filtered = [f.filter(x, timestamp=t) for x, t in zip(raw[45:], ts[45:])]
        assert mean_abs_diff(filtered) < mean_abs_diff(raw[45:])


# =============================================================================
# OneEuroFilter — update_params()
# =============================================================================

class TestOneEuroFilterUpdateParams:

    def test_update_mincutoff_stored(self):
        f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
        f.update_params(mincutoff=2.0)
        assert f.mincutoff == 2.0

    def test_update_beta_stored(self):
        f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
        f.update_params(beta=0.02)
        assert f.beta == 0.02

    def test_update_dcutoff_stored(self):
        f = OneEuroFilter(freq=30.0, dcutoff=1.0)
        f.update_params(dcutoff=2.0)
        assert f.dcutoff == 2.0

    def test_update_partial_leaves_others_unchanged(self):
        """Only the specified parameter changes; others stay the same."""
        f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007, dcutoff=1.0)
        f.update_params(beta=0.05)
        assert f.mincutoff == 1.0
        assert f.beta      == 0.05
        assert f.dcutoff   == 1.0

    def test_update_takes_effect_on_next_call(self):
        """
        Spec: update_params() takes effect on the very next filter() call.
        Filter state (smoothed value) must be preserved across the update.
        """
        f  = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
        ts = make_timestamps(20)
        for i, t in enumerate(ts):
            f.filter(0.5, timestamp=t)
        assert f.is_initialised()
        prev_val = f.last_value()
        # Hot-swap
        f.update_params(mincutoff=2.0, beta=0.02)
        # State preserved (is_initialised still True)
        assert f.is_initialised()
        # Next call succeeds and output is finite
        out = f.filter(0.8, timestamp=ts[-1] + 1.0 / 30.0)
        assert math.isfinite(out)

    def test_update_with_none_is_no_op(self):
        """Passing None for a parameter leaves it unchanged."""
        f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007, dcutoff=1.0)
        f.update_params(mincutoff=None, beta=None, dcutoff=None)
        assert f.mincutoff == 1.0
        assert f.beta      == 0.007
        assert f.dcutoff   == 1.0

    def test_invalid_mincutoff_raises(self):
        f = OneEuroFilter(freq=30.0)
        with pytest.raises(ValueError):
            f.update_params(mincutoff=0.0)
        with pytest.raises(ValueError):
            f.update_params(mincutoff=-1.0)

    def test_invalid_beta_raises(self):
        f = OneEuroFilter(freq=30.0)
        with pytest.raises(ValueError):
            f.update_params(beta=-0.001)

    def test_invalid_dcutoff_raises(self):
        f = OneEuroFilter(freq=30.0)
        with pytest.raises(ValueError):
            f.update_params(dcutoff=0.0)
        with pytest.raises(ValueError):
            f.update_params(dcutoff=-1.0)

    def test_update_on_uninitialised_filter_does_not_raise(self):
        """update_params() must work even before the first filter() call."""
        f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
        f.update_params(mincutoff=1.5, beta=0.01)
        assert f.mincutoff == 1.5
        assert f.beta      == 0.01
        out = f.filter(0.5)
        assert out == 0.5   # still cold-start after update on uninitialised


# =============================================================================
# OneEuroFilter — repr and introspection
# =============================================================================

class TestOneEuroFilterRepr:

    def test_repr_contains_class_name(self):
        f = OneEuroFilter(freq=30.0)
        assert "OneEuroFilter" in repr(f)

    def test_repr_contains_key_params(self):
        f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007, dcutoff=1.0)
        r = repr(f)
        assert "freq=30.0"     in r
        assert "mincutoff=1.0" in r
        assert "beta=0.007"    in r

    def test_repr_reflects_initialised_state(self):
        f = OneEuroFilter(freq=30.0)
        assert "False" in repr(f)
        f.filter(0.5)
        assert "True" in repr(f)


# =============================================================================
# LandmarkSmoother — construction
# =============================================================================

class TestLandmarkSmootherConstruction:

    def test_pool_empty_on_init(self):
        s = LandmarkSmoother()
        assert s.active_filter_count == 0

    def test_not_initialised_on_init(self):
        s = LandmarkSmoother()
        assert not s.is_initialised()

    def test_default_params_stored(self):
        s = LandmarkSmoother(
            freq=25.0, mincutoff=0.8, beta=0.01, dcutoff=1.5
        )
        assert s.freq      == 25.0
        assert s.mincutoff == 0.8
        assert s.beta      == 0.01
        assert s.dcutoff   == 1.5

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            LandmarkSmoother(freq=0.0)
        with pytest.raises(ValueError):
            LandmarkSmoother(mincutoff=-1.0)
        with pytest.raises(ValueError):
            LandmarkSmoother(beta=-0.1)


# =============================================================================
# LandmarkSmoother — smooth()
# =============================================================================

class TestLandmarkSmootherSmooth:

    def test_output_length_21(self):
        s   = LandmarkSmoother()
        out = s.smooth(flat_landmarks())
        assert len(out) == 21

    def test_output_tuples_have_3_components(self):
        s   = LandmarkSmoother()
        out = s.smooth(flat_landmarks())
        for i, lm in enumerate(out):
            assert len(lm) == 3, f"Landmark {i} has {len(lm)} components"

    def test_all_output_values_finite(self):
        s   = LandmarkSmoother()
        out = s.smooth(flat_landmarks())
        for i, lm in enumerate(out):
            for ax, val in enumerate(lm):
                assert math.isfinite(val), \
                    f"Non-finite at lm={i} ax={ax}: {val}"

    def test_first_call_equals_input_all_63_axes(self):
        """Spec: cold-start — all 63 axes must equal raw input."""
        s   = LandmarkSmoother()
        lms = [(0.1 * i % 1.0, 0.05 * i % 1.0, 0.02 * i % 1.0)
               for i in range(21)]
        out = s.smooth(lms)
        for i in range(21):
            for ax in range(3):
                assert out[i][ax] == lms[i][ax], \
                    f"Cold-start mismatch lm={i} ax={ax}: {out[i][ax]} != {lms[i][ax]}"

    def test_output_smoother_than_input_index_tip_x(self):
        """
        Spec: filtered MAD on landmark 8 x-axis must be < raw MAD.
        Uses deterministic triangle noise, 60 frames.
        """
        raw_signal = noisy_sine(60)
        ts         = make_timestamps(60)
        s          = LandmarkSmoother(freq=30.0, mincutoff=1.0, beta=0.007)

        raw_stream:      list[float] = []
        filtered_stream: list[float] = []

        for x_val, t in zip(raw_signal, ts):
            lms = flat_landmarks(x=x_val)
            out = s.smooth(lms, timestamp=t)
            raw_stream.append(x_val)
            filtered_stream.append(out[8][0])

        assert mean_abs_diff(filtered_stream) < mean_abs_diff(raw_stream), (
            f"Smoother check failed: "
            f"raw_MAD={mean_abs_diff(raw_stream):.5f}, "
            f"filtered_MAD={mean_abs_diff(filtered_stream):.5f}"
        )

    def test_lazy_pool_grows_to_63_on_first_call(self):
        """Spec: after one smooth() call all 63 filters are instantiated."""
        s = LandmarkSmoother()
        assert s.active_filter_count == 0
        s.smooth(flat_landmarks())
        assert s.active_filter_count == 63

    def test_pool_does_not_exceed_63(self):
        """Pool must never grow beyond 63 no matter how many frames pass."""
        s  = LandmarkSmoother()
        ts = make_timestamps(120)
        for t in ts:
            s.smooth(flat_landmarks(x=0.5), timestamp=t)
        assert s.active_filter_count == 63

    def test_wrong_landmark_count_raises(self):
        s = LandmarkSmoother()
        for n in (0, 20, 22, 100):
            with pytest.raises(ValueError):
                s.smooth([(0.5, 0.5, 0.0)] * n)

    def test_wrong_tuple_width_raises(self):
        s = LandmarkSmoother()
        with pytest.raises(ValueError):
            s.smooth([(0.5, 0.5)] * 21)          # 2-tuples

    def test_non_finite_value_raises(self):
        s = LandmarkSmoother()
        for bad in (float("nan"), float("inf"), float("-inf")):
            lms = flat_landmarks()
            lms_list = list(lms)
            lms_list[8] = (bad, 0.5, 0.0)
            with pytest.raises(ValueError):
                s.smooth(lms_list)

    def test_output_is_new_list_not_mutation(self):
        """smooth() must return a fresh list — input must not be mutated."""
        s   = LandmarkSmoother()
        lms = flat_landmarks(x=0.3)
        original_first = lms[0]
        s.smooth(lms)
        assert lms[0] == original_first, "smooth() mutated input list"


# =============================================================================
# LandmarkSmoother — reset()
# =============================================================================

class TestLandmarkSmootherReset:

    def test_reset_marks_all_filters_uninitialised(self):
        """Spec: after reset(), is_initialised() returns False."""
        s  = LandmarkSmoother()
        ts = make_timestamps(10)
        for t in ts:
            s.smooth(flat_landmarks(), timestamp=t)
        assert s.is_initialised()
        s.reset()
        assert not s.is_initialised()

    def test_reset_preserves_pool_instances(self):
        """Pool dict must retain its 63 entries after reset (reuse, not rebuild)."""
        s  = LandmarkSmoother()
        ts = make_timestamps(5)
        for t in ts:
            s.smooth(flat_landmarks(), timestamp=t)
        assert s.active_filter_count == 63
        s.reset()
        assert s.active_filter_count == 63

    def test_first_call_after_reset_equals_input(self):
        """Spec: post-reset cold-start — output must equal input for all 63 axes."""
        s  = LandmarkSmoother()
        ts = make_timestamps(20)
        for t in ts:
            s.smooth(flat_landmarks(x=0.8, y=0.2), timestamp=t)
        s.reset()
        lms = [(0.11 * i % 1.0, 0.07 * i % 1.0, 0.0) for i in range(21)]
        out = s.smooth(lms)
        for i in range(21):
            for ax in range(3):
                assert out[i][ax] == lms[i][ax], \
                    f"Post-reset cold-start mismatch lm={i} ax={ax}"

    def test_reset_preserves_smoother_params(self):
        """reset() must not alter mincutoff, beta, dcutoff, or freq."""
        s = LandmarkSmoother(freq=25.0, mincutoff=2.0, beta=0.01, dcutoff=1.5)
        s.smooth(flat_landmarks())
        s.reset()
        assert s.freq      == 25.0
        assert s.mincutoff == 2.0
        assert s.beta      == 0.01
        assert s.dcutoff   == 1.5

    def test_reset_cascades_to_all_pool_members(self):
        """Each individual filter in the pool must be uninitialised after reset."""
        s  = LandmarkSmoother()
        ts = make_timestamps(5)
        for t in ts:
            s.smooth(flat_landmarks(), timestamp=t)
        s.reset()
        for key, f in s._filters.items():
            assert not f.is_initialised(), \
                f"Filter at key {key} still initialised after reset()"


# =============================================================================
# LandmarkSmoother — update_params()
# =============================================================================

class TestLandmarkSmootherUpdateParams:

    def test_update_stored_on_smoother(self):
        s = LandmarkSmoother(mincutoff=1.0, beta=0.007)
        s.update_params(mincutoff=2.0, beta=0.015)
        assert s.mincutoff == 2.0
        assert s.beta      == 0.015

    def test_update_propagates_to_all_pool_members(self):
        """Spec: every filter in the pool must reflect the new parameters."""
        s  = LandmarkSmoother(mincutoff=1.0, beta=0.007)
        ts = make_timestamps(5)
        for t in ts:
            s.smooth(flat_landmarks(), timestamp=t)
        s.update_params(mincutoff=2.0, beta=0.02)
        for key, f in s._filters.items():
            assert f.mincutoff == 2.0, \
                f"Filter {key} mincutoff not updated: {f.mincutoff}"
            assert f.beta == 0.02, \
                f"Filter {key} beta not updated: {f.beta}"

    def test_new_filters_after_update_inherit_params(self):
        """
        Filters created after update_params() must use the new values.
        Reset clears state but we can check via stored smoother attrs.
        """
        s = LandmarkSmoother(mincutoff=1.0, beta=0.007)
        s.update_params(mincutoff=3.0, beta=0.05)
        # Trigger lazy creation of new filters (pool was empty before this)
        s.smooth(flat_landmarks())
        for f in s._filters.values():
            assert f.mincutoff == 3.0
            assert f.beta      == 0.05

    def test_state_preserved_after_update(self):
        """Spec: filter state (is_initialised) preserved across update."""
        s  = LandmarkSmoother(mincutoff=1.0, beta=0.007)
        ts = make_timestamps(10)
        for t in ts:
            s.smooth(flat_landmarks(), timestamp=t)
        assert s.is_initialised()
        s.update_params(mincutoff=1.5)
        assert s.is_initialised()

    def test_invalid_mincutoff_raises_on_empty_pool(self):
        """Spec: validation must work even before first smooth() call."""
        s = LandmarkSmoother()
        with pytest.raises(ValueError):
            s.update_params(mincutoff=-1.0)
        with pytest.raises(ValueError):
            s.update_params(mincutoff=0.0)

    def test_invalid_beta_raises_on_empty_pool(self):
        s = LandmarkSmoother()
        with pytest.raises(ValueError):
            s.update_params(beta=-0.1)

    def test_invalid_dcutoff_raises_on_empty_pool(self):
        s = LandmarkSmoother()
        with pytest.raises(ValueError):
            s.update_params(dcutoff=-1.0)
        with pytest.raises(ValueError):
            s.update_params(dcutoff=0.0)

    def test_invalid_mincutoff_raises_on_non_empty_pool(self):
        s = LandmarkSmoother()
        s.smooth(flat_landmarks())
        with pytest.raises(ValueError):
            s.update_params(mincutoff=-1.0)

    def test_none_params_are_no_op(self):
        s = LandmarkSmoother(mincutoff=1.0, beta=0.007, dcutoff=1.0)
        s.smooth(flat_landmarks())
        s.update_params(mincutoff=None, beta=None, dcutoff=None)
        assert s.mincutoff == 1.0
        assert s.beta      == 0.007
        assert s.dcutoff   == 1.0


# =============================================================================
# LandmarkSmoother — get_cursor_position()
# =============================================================================

class TestGetCursorPosition:

    def _lms_with_tip(self, x: float, y: float) -> list:
        """21-landmark list with INDEX_TIP (landmark 8) at (x, y, 0)."""
        lms = list(flat_landmarks())
        lms[CURSOR_LANDMARK_INDEX] = (x, y, 0.0)
        return lms

    def test_centre_maps_to_expected_pixel(self):
        """
        INDEX_TIP at frame centre (0.5, 0.5) must map to the correct
        pixel given the default active zone and sensitivity=1.0.
        """
        s  = LandmarkSmoother()
        z  = DEFAULT_ACTIVE_ZONE
        lms = self._lms_with_tip(0.5, 0.5)
        cx, cy = s.get_cursor_position(
            lms, sensitivity=1.0, screen_w=1920, screen_h=1080,
            active_zone=z,
        )
        expected_x = int(((0.5 - z["x1"]) / (z["x2"] - z["x1"])) * 1920)
        expected_y = int(((0.5 - z["y1"]) / (z["y2"] - z["y1"])) * 1080)
        assert cx == expected_x, f"cx={cx}, expected={expected_x}"
        assert cy == expected_y, f"cy={cy}, expected={expected_y}"

    def test_output_within_screen_bounds(self):
        """Return values must always satisfy 0 ≤ x < screen_w and 0 ≤ y < screen_h."""
        s = LandmarkSmoother()
        for tip_x, tip_y in [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0),
                              (0.5, 0.5), (0.25, 0.75)]:
            lms = self._lms_with_tip(tip_x, tip_y)
            cx, cy = s.get_cursor_position(
                lms, sensitivity=0.85,
                screen_w=1920, screen_h=1080,
                active_zone=DEFAULT_ACTIVE_ZONE,
            )
            assert 0 <= cx <= 1919, f"cx={cx} out of bounds for tip=({tip_x},{tip_y})"
            assert 0 <= cy <= 1079, f"cy={cy} out of bounds for tip=({tip_x},{tip_y})"

    def test_out_of_zone_clamped_to_screen_edge(self):
        """Landmarks outside the active zone must be clamped, not wrapped."""
        s   = LandmarkSmoother()
        lms = self._lms_with_tip(2.0, -1.0)   # far outside [0,1]
        cx, cy = s.get_cursor_position(
            lms, sensitivity=0.85,
            screen_w=1920, screen_h=1080,
            active_zone=DEFAULT_ACTIVE_ZONE,
        )
        assert 0 <= cx <= 1919
        assert 0 <= cy <= 1079

    def test_sensitivity_scales_output(self):
        """Higher sensitivity → larger screen coordinates for same tip position."""
        s   = LandmarkSmoother()
        lms = self._lms_with_tip(0.7, 0.7)
        cx_low,  cy_low  = s.get_cursor_position(
            lms, sensitivity=0.5, screen_w=1920, screen_h=1080,
            active_zone=DEFAULT_ACTIVE_ZONE,
        )
        cx_high, cy_high = s.get_cursor_position(
            lms, sensitivity=1.0, screen_w=1920, screen_h=1080,
            active_zone=DEFAULT_ACTIVE_ZONE,
        )
        assert cx_high >= cx_low
        assert cy_high >= cy_low

    def test_uses_cursor_landmark_index(self):
        """Only landmark CURSOR_LANDMARK_INDEX should drive cursor position."""
        s    = LandmarkSmoother()
        zone = DEFAULT_ACTIVE_ZONE
        # All landmarks at (0.5, 0.5) except the cursor landmark
        lms_a = list(flat_landmarks(x=0.5, y=0.5))
        lms_b = list(flat_landmarks(x=0.5, y=0.5))
        lms_b[CURSOR_LANDMARK_INDEX] = (0.8, 0.2, 0.0)

        pos_a = s.get_cursor_position(lms_a, 0.85, 1920, 1080, zone)
        pos_b = s.get_cursor_position(lms_b, 0.85, 1920, 1080, zone)
        assert pos_a != pos_b, "Cursor position did not change when tip moved"

    def test_invalid_landmark_count_raises(self):
        s = LandmarkSmoother()
        with pytest.raises(ValueError):
            s.get_cursor_position([(0.5, 0.5, 0.0)] * 10, 0.85, 1920, 1080,
                                  DEFAULT_ACTIVE_ZONE)

    def test_invalid_sensitivity_raises(self):
        s   = LandmarkSmoother()
        lms = flat_landmarks()
        with pytest.raises(ValueError):
            s.get_cursor_position(lms, sensitivity=-0.1,
                                  screen_w=1920, screen_h=1080,
                                  active_zone=DEFAULT_ACTIVE_ZONE)
        with pytest.raises(ValueError):
            s.get_cursor_position(lms, sensitivity=0.0,
                                  screen_w=1920, screen_h=1080,
                                  active_zone=DEFAULT_ACTIVE_ZONE)

    def test_invalid_screen_dimensions_raise(self):
        s   = LandmarkSmoother()
        lms = flat_landmarks()
        with pytest.raises(ValueError):
            s.get_cursor_position(lms, 0.85, screen_w=0, screen_h=1080,
                                  active_zone=DEFAULT_ACTIVE_ZONE)
        with pytest.raises(ValueError):
            s.get_cursor_position(lms, 0.85, screen_w=1920, screen_h=-1,
                                  active_zone=DEFAULT_ACTIVE_ZONE)

    def test_missing_active_zone_key_raises(self):
        s   = LandmarkSmoother()
        lms = flat_landmarks()
        bad_zone = {"x1": 0.05, "y1": 0.05, "x2": 0.95}  # missing y2
        with pytest.raises(ValueError):
            s.get_cursor_position(lms, 0.85, 1920, 1080, bad_zone)

    def test_degenerate_active_zone_raises(self):
        """Zone where x1==x2 or y1==y2 must raise (division by zero guard)."""
        s   = LandmarkSmoother()
        lms = flat_landmarks()
        with pytest.raises(ValueError):
            s.get_cursor_position(lms, 0.85, 1920, 1080,
                                  {"x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 1.0})
        with pytest.raises(ValueError):
            s.get_cursor_position(lms, 0.85, 1920, 1080,
                                  {"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5})

    def test_return_types_are_integers(self):
        """Return values must be plain Python ints, not numpy or float."""
        s     = LandmarkSmoother()
        cx, cy = s.get_cursor_position(
            flat_landmarks(), 0.85, 1920, 1080, DEFAULT_ACTIVE_ZONE
        )
        assert isinstance(cx, int), f"cx is {type(cx)}, expected int"
        assert isinstance(cy, int), f"cy is {type(cy)}, expected int"


# =============================================================================
# LandmarkSmoother — repr
# =============================================================================

class TestLandmarkSmootherRepr:

    def test_repr_contains_class_name(self):
        assert "LandmarkSmoother" in repr(LandmarkSmoother())

    def test_repr_shows_zero_filters_before_first_call(self):
        assert "0/63" in repr(LandmarkSmoother())

    def test_repr_shows_63_filters_after_first_call(self):
        s = LandmarkSmoother()
        s.smooth(flat_landmarks())
        assert "63/63" in repr(s)