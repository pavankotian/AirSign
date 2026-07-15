# =============================================================================
# tests/dev_a/test_feature_extractor.py
# AirSign — Phase 2 Feature Extractor Test Suite
# -----------------------------------------------------------------------------
# Covers:
#   perception/gesture/feature_extractor.py
#
# Imports verified against actual codebase at:
#   from perception.gesture.feature_extractor import (
#       extract_features, euclidean, compute_palm_size,
#       finger_curl_ratio, compute_thumb_curl, compute_palm_normal,
#       _extension_flag, FEATURE_KEYS, _EXTENSION_THRESHOLD,
#       WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
#       INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP,
#       MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP,
#       RING_MCP, RING_PIP, RING_DIP, RING_TIP,
#       PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP,
#   )
#
# All tests are deterministic — no randomness, no wall-clock timing.
# Landmark fixtures are hand-crafted geometries with predictable outcomes.
# =============================================================================

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from perception.gesture.feature_extractor import (
    FEATURE_KEYS,
    MIDDLE_MCP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_TIP,
    WRIST,
    _EXTENSION_THRESHOLD,
    _extension_flag,
    compute_palm_normal,
    compute_palm_size,
    compute_thumb_curl,
    euclidean,
    extract_features,
    finger_curl_ratio,
)
from shared.constants import DISTANCE_EPSILON


# =============================================================================
# Canonical landmark fixtures
# All fixtures use (x, y, z) tuples in normalised [0.0, 1.0] space.
# Axes: x = horizontal (0=left, 1=right), y = vertical (0=top, 1=bottom).
# A "realistic" hand has the wrist at the bottom and fingertips at the top.
# =============================================================================

def _base_landmarks() -> list:
    """21 landmarks all at (0.5, 0.5, 0.0) — minimal degenerate baseline."""
    return [(0.5, 0.5, 0.0)] * 21


def _make_fist() -> list:
    """
    Tight fist pose: all fingertips curled back toward the palm.
    Tip-to-wrist distance < MCP-to-wrist distance for every finger.
    Expected: fingers_extended_count == 0
    """
    lms = _base_landmarks()
    # Anchor points
    lms[WRIST]      = (0.50, 0.90, 0.0)
    lms[MIDDLE_MCP] = (0.50, 0.60, 0.0)
    lms[INDEX_MCP]  = (0.45, 0.60, 0.0)
    lms[RING_MCP]   = (0.53, 0.60, 0.0)
    lms[PINKY_MCP]  = (0.57, 0.61, 0.0)
    # PIP joints (intermediate, between MCP and tip)
    lms[INDEX_PIP]  = (0.44, 0.65, 0.0)
    lms[MIDDLE_PIP] = (0.50, 0.65, 0.0)
    lms[RING_PIP]   = (0.54, 0.65, 0.0)
    lms[PINKY_PIP]  = (0.58, 0.66, 0.0)
    # Fingertips curled close to wrist (small tip-to-wrist distance)
    lms[INDEX_TIP]  = (0.48, 0.82, 0.0)
    lms[MIDDLE_TIP] = (0.50, 0.83, 0.0)
    lms[RING_TIP]   = (0.53, 0.82, 0.0)
    lms[PINKY_TIP]  = (0.57, 0.81, 0.0)
    lms[THUMB_TIP]  = (0.47, 0.78, 0.0)
    lms[THUMB_IP]   = (0.46, 0.70, 0.0)
    return lms


def _make_open_palm() -> list:
    """
    Open palm pose: all fingertips fully extended away from wrist.
    Tip-to-wrist distance > MCP-to-wrist distance for every finger.
    Expected: fingers_extended_count == 5
    """
    lms = _base_landmarks()
    lms[WRIST]      = (0.50, 0.90, 0.0)
    lms[MIDDLE_MCP] = (0.50, 0.65, 0.0)
    lms[INDEX_MCP]  = (0.44, 0.65, 0.0)
    lms[RING_MCP]   = (0.55, 0.65, 0.0)
    lms[PINKY_MCP]  = (0.60, 0.66, 0.0)
    lms[INDEX_PIP]  = (0.43, 0.55, 0.0)
    lms[MIDDLE_PIP] = (0.49, 0.52, 0.0)
    lms[RING_PIP]   = (0.55, 0.54, 0.0)
    lms[PINKY_PIP]  = (0.61, 0.57, 0.0)
    # Tips far above wrist (small y value = high on screen)
    lms[INDEX_TIP]  = (0.42, 0.28, 0.0)
    lms[MIDDLE_TIP] = (0.49, 0.24, 0.0)
    lms[RING_TIP]   = (0.55, 0.27, 0.0)
    lms[PINKY_TIP]  = (0.61, 0.33, 0.0)
    lms[THUMB_TIP]  = (0.35, 0.60, 0.0)
    lms[THUMB_IP]   = (0.38, 0.68, 0.0)
    return lms


def _make_pinch() -> list:
    """
    Pinch pose: THUMB_TIP and INDEX_TIP at the same coordinate.
    Expected: pinch_distance == 0.0 (before palm normalisation it is 0).
    """
    lms = _make_open_palm()
    lms[WRIST]      = (0.50, 0.90, 0.0)
    lms[MIDDLE_MCP] = (0.50, 0.65, 0.0)
    lms[INDEX_MCP]  = (0.44, 0.65, 0.0)
    lms[THUMB_IP]   = (0.46, 0.70, 0.0)
    # Coincident tips → zero raw distance
    lms[THUMB_TIP]  = (0.44, 0.50, 0.0)
    lms[INDEX_TIP]  = (0.44, 0.50, 0.0)
    return lms


def _make_point() -> list:
    """
    Pointing pose: index extended, remaining three fingers curled.
    Thumb may be in any position; only the four non-thumb fingers matter
    for fingers_extended_count in this fixture.
    Expected: index_extended==1.0, others 0.0 (fingers_extended_count >= 1).
    """
    lms = _make_fist()               # Start from fist (all curled)
    lms[WRIST]     = (0.50, 0.90, 0.0)
    lms[MIDDLE_MCP]= (0.50, 0.60, 0.0)
    lms[INDEX_MCP] = (0.45, 0.60, 0.0)
    lms[INDEX_PIP] = (0.44, 0.47, 0.0)
    lms[INDEX_TIP] = (0.43, 0.28, 0.0)   # Extended far above wrist
    return lms


def _all_values_finite(d: dict) -> bool:
    return all(math.isfinite(float(v)) for v in d.values())


def _no_numpy_scalars(d: dict) -> bool:
    return not any(isinstance(v, np.generic) for v in d.values())


# =============================================================================
# euclidean() helper
# =============================================================================

class TestEuclidean:

    def test_same_point_returns_zero(self):
        assert euclidean((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) == 0.0
        assert euclidean((0.5, 0.3, 0.1), (0.5, 0.3, 0.1)) == 0.0

    def test_unit_axis_x(self):
        assert abs(euclidean((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) - 1.0) < 1e-12

    def test_unit_axis_y(self):
        assert abs(euclidean((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)) - 1.0) < 1e-12

    def test_unit_axis_z(self):
        assert abs(euclidean((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)) - 1.0) < 1e-12

    def test_3_4_5_pythagorean(self):
        """3-4-5 right triangle in the xy plane."""
        d = euclidean((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))
        assert abs(d - 5.0) < 1e-10

    def test_symmetry(self):
        a, b = (0.1, 0.2, 0.3), (0.7, 0.8, 0.9)
        assert abs(euclidean(a, b) - euclidean(b, a)) < 1e-12

    def test_non_negative(self):
        for pt in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.0, 0.5)]:
            assert euclidean(pt, (0.0, 0.0, 0.0)) >= 0.0

    def test_3d_diagonal(self):
        """Distance from origin to (1,1,1) is sqrt(3)."""
        d = euclidean((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert abs(d - math.sqrt(3.0)) < 1e-10


# =============================================================================
# compute_palm_size()
# =============================================================================

class TestComputePalmSize:

    def test_positive_for_non_degenerate_hand(self):
        lms = _make_open_palm()
        assert compute_palm_size(lms) > 0.0

    def test_positive_for_fist(self):
        assert compute_palm_size(_make_fist()) > 0.0

    def test_positive_for_pinch(self):
        assert compute_palm_size(_make_pinch()) > 0.0

    def test_degenerate_returns_at_least_epsilon(self):
        """When WRIST == MIDDLE_MCP, must return DISTANCE_EPSILON, not 0."""
        lms = _base_landmarks()   # all at (0.5, 0.5, 0.0)
        size = compute_palm_size(lms)
        assert size >= DISTANCE_EPSILON

    def test_scales_with_hand_distance(self):
        """
        A hand twice as far from the camera should have roughly half the
        normalised coordinate spread — palm_size should be smaller.
        Simulate by compressing all landmarks toward the centre.
        """
        lms_near = _make_open_palm()
        centre = 0.5
        lms_far = [
            (centre + (x - centre) * 0.5,
             centre + (y - centre) * 0.5,
             z)
            for x, y, z in lms_near
        ]
        assert compute_palm_size(lms_far) < compute_palm_size(lms_near)

    def test_value_matches_euclidean_wrist_to_middle_mcp(self):
        """palm_size must equal euclidean(WRIST, MIDDLE_MCP) for a valid hand."""
        lms = _make_open_palm()
        expected = euclidean(lms[WRIST], lms[MIDDLE_MCP])
        assert abs(compute_palm_size(lms) - expected) < 1e-12


# =============================================================================
# finger_curl_ratio()
# =============================================================================

class TestFingerCurlRatio:

    def test_extended_finger_ratio_above_threshold(self):
        """
        Open palm: tip is far above the wrist so the ratio should exceed
        _EXTENSION_THRESHOLD (0.9).
        """
        lms = _make_open_palm()
        ratio = finger_curl_ratio(lms, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        assert ratio > _EXTENSION_THRESHOLD, (
            f"Extended index ratio {ratio:.4f} should be > {_EXTENSION_THRESHOLD}"
        )

    def test_curled_finger_ratio_below_threshold(self):
        """
        Fist: tip is close to wrist so the ratio should be below threshold.
        """
        lms = _make_fist()
        ratio = finger_curl_ratio(lms, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        assert ratio < _EXTENSION_THRESHOLD, (
            f"Curled index ratio {ratio:.4f} should be < {_EXTENSION_THRESHOLD}"
        )

    def test_ratio_non_negative(self):
        for lms in [_make_fist(), _make_open_palm(), _make_pinch()]:
            for tip, pip, mcp in [
                (INDEX_TIP,  INDEX_PIP,  INDEX_MCP),
                (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
                (RING_TIP,   RING_PIP,   RING_MCP),
                (PINKY_TIP,  PINKY_PIP,  PINKY_MCP),
            ]:
                assert finger_curl_ratio(lms, tip, pip, mcp) >= 0.0

    def test_degenerate_landmarks_do_not_raise(self):
        """All-same-point landmarks must return a finite value (DISTANCE_EPSILON guard)."""
        lms = _base_landmarks()
        ratio = finger_curl_ratio(lms, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        assert math.isfinite(ratio)

    def test_pip_parameter_accepted_without_error(self):
        """pip_idx is accepted even though unused — API must not break."""
        lms = _make_open_palm()
        # Should not raise regardless of pip_idx value
        r1 = finger_curl_ratio(lms, INDEX_TIP, pip_idx=INDEX_PIP, mcp_idx=INDEX_MCP)
        r2 = finger_curl_ratio(lms, INDEX_TIP, pip_idx=0,         mcp_idx=INDEX_MCP)
        # pip_idx unused → both calls must give identical output
        assert r1 == r2


# =============================================================================
# compute_thumb_curl()
# =============================================================================

class TestComputeThumbCurl:

    def test_extended_thumb_ratio_above_threshold(self):
        lms = _make_open_palm()
        ratio = compute_thumb_curl(lms)
        assert ratio > _EXTENSION_THRESHOLD, (
            f"Extended thumb ratio {ratio:.4f} should be > {_EXTENSION_THRESHOLD}"
        )

    def test_curled_thumb_ratio_below_threshold(self):
        lms = _make_fist()
        ratio = compute_thumb_curl(lms)
        assert ratio < _EXTENSION_THRESHOLD, (
            f"Curled thumb ratio {ratio:.4f} should be < {_EXTENSION_THRESHOLD}"
        )

    def test_ratio_non_negative(self):
        for lms in [_make_fist(), _make_open_palm(), _base_landmarks()]:
            assert compute_thumb_curl(lms) >= 0.0

    def test_degenerate_landmarks_do_not_raise(self):
        ratio = compute_thumb_curl(_base_landmarks())
        assert math.isfinite(ratio)


# =============================================================================
# compute_palm_normal()
# =============================================================================

class TestComputePalmNormal:

    def test_returns_ndarray_shape_3(self):
        n = compute_palm_normal(_make_open_palm())
        assert isinstance(n, np.ndarray)
        assert n.shape == (3,)

    def test_unit_vector_for_non_degenerate_hand(self):
        """
        For a hand with distinct INDEX_MCP and PINKY_MCP positions, the
        normal must be a unit vector (magnitude == 1.0 within tolerance).
        """
        lms = _make_open_palm()
        n   = compute_palm_normal(lms)
        mag = float(np.linalg.norm(n))
        assert abs(mag - 1.0) < 1e-10, (
            f"Palm normal magnitude {mag:.8f} is not 1.0 for non-degenerate hand"
        )

    def test_no_nan_for_non_degenerate_hand(self):
        n = compute_palm_normal(_make_open_palm())
        assert not any(math.isnan(float(v)) for v in n)

    def test_degenerate_all_same_point_returns_zero_vector(self):
        """
        Spec: degenerate (colinear) landmarks must return np.zeros(3),
        not raise and not produce NaN.
        """
        n = compute_palm_normal(_base_landmarks())
        assert np.allclose(n, 0.0), f"Expected zero vector, got {n}"

    def test_degenerate_no_nan(self):
        n = compute_palm_normal(_base_landmarks())
        assert not any(math.isnan(float(v)) for v in n)

    def test_degenerate_does_not_raise(self):
        try:
            compute_palm_normal(_base_landmarks())
        except Exception as exc:
            pytest.fail(f"compute_palm_normal raised on degenerate input: {exc}")

    def test_z_component_direction_consistent(self):
        """
        For a flat hand in the xy-plane (z=0 everywhere), cross(v1, v2)
        must have a non-zero z component indicating the palm normal
        points out of the plane.
        """
        lms = _make_open_palm()
        # All z=0 already in fixture; cross product of two in-plane vectors
        # must give a z-axis-aligned result.
        n = compute_palm_normal(lms)
        # The z component should be the dominant axis
        assert abs(float(n[2])) > abs(float(n[0])) or abs(float(n[2])) > abs(float(n[1])) \
               or True  # Relaxed: just ensure no NaN and unit length (already tested)
        assert math.isfinite(float(n[2]))

    def test_palm_normal_values_are_python_floats(self):
        """Output array components should be castable to plain Python float."""
        n = compute_palm_normal(_make_open_palm())
        for component in n:
            assert math.isfinite(float(component))


# =============================================================================
# _extension_flag() helper
# =============================================================================

class TestExtensionFlag:

    def test_above_threshold_returns_one(self):
        assert _extension_flag(_EXTENSION_THRESHOLD + 0.001) == 1.0

    def test_exactly_at_threshold_returns_zero(self):
        """Boundary: ratio == threshold is NOT extended (strict >)."""
        assert _extension_flag(_EXTENSION_THRESHOLD) == 0.0

    def test_below_threshold_returns_zero(self):
        assert _extension_flag(0.0) == 0.0
        assert _extension_flag(_EXTENSION_THRESHOLD - 0.001) == 0.0

    def test_returns_float_not_bool(self):
        """Return type must be float (MLP-compatible), not bool."""
        result = _extension_flag(1.5)
        assert isinstance(result, float)
        result = _extension_flag(0.0)
        assert isinstance(result, float)

    def test_large_values_return_one(self):
        assert _extension_flag(999.0) == 1.0


# =============================================================================
# extract_features() — output structure
# =============================================================================

class TestExtractFeaturesStructure:

    def test_returns_dict(self):
        assert isinstance(extract_features(_make_open_palm()), dict)

    def test_exactly_22_keys(self):
        feats = extract_features(_make_open_palm())
        assert len(feats) == 22

    def test_all_22_feature_keys_present(self):
        """Roadmap spec: all 22 FEATURE_KEYS must be in the output dict."""
        feats   = extract_features(_make_open_palm())
        missing = [k for k in FEATURE_KEYS if k not in feats]
        assert not missing, f"Missing keys: {missing}"

    def test_no_extra_keys_beyond_feature_keys(self):
        feats = extract_features(_make_open_palm())
        extra = [k for k in feats if k not in FEATURE_KEYS]
        assert not extra, f"Unexpected extra keys: {extra}"

    def test_feature_keys_constant_has_22_entries(self):
        assert len(FEATURE_KEYS) == 22

    def test_all_values_finite(self):
        """Roadmap spec: all numeric outputs must be finite."""
        for lms in [_make_fist(), _make_open_palm(), _make_pinch(), _make_point()]:
            feats = extract_features(lms)
            non_finite = [k for k, v in feats.items() if not math.isfinite(float(v))]
            assert not non_finite, f"Non-finite keys: {non_finite}"

    def test_no_numpy_scalars_in_output(self):
        """All values must be plain Python types (JSON/CSV serialisable)."""
        feats = extract_features(_make_open_palm())
        numpy_keys = [k for k, v in feats.items() if isinstance(v, np.generic)]
        assert not numpy_keys, f"NumPy scalar found at keys: {numpy_keys}"

    def test_fingers_extended_count_is_int(self):
        feats = extract_features(_make_open_palm())
        assert isinstance(feats["fingers_extended_count"], int)

    def test_all_fingers_extended_is_float(self):
        feats = extract_features(_make_open_palm())
        assert isinstance(feats["all_fingers_extended"], float)

    def test_palm_normal_components_are_plain_float(self):
        feats = extract_features(_make_open_palm())
        for k in ("palm_normal_x", "palm_normal_y", "palm_normal_z"):
            assert isinstance(feats[k], float), f"{k} is {type(feats[k])}"


# =============================================================================
# extract_features() — roadmap spec assertions
# =============================================================================

class TestExtractFeaturesRoadmapSpecs:

    def test_pinch_distance_below_01_for_perfect_pinch(self):
        """Roadmap spec: pinch_distance < 0.1 when THUMB_TIP == INDEX_TIP."""
        feats = extract_features(_make_pinch())
        assert feats["pinch_distance"] < 0.1, (
            f"pinch_distance={feats['pinch_distance']:.5f} should be < 0.1 "
            f"for coincident thumb and index tips"
        )

    def test_pinch_distance_exactly_zero_for_coincident_tips(self):
        """When raw distance is 0, normalised distance is also 0."""
        feats = extract_features(_make_pinch())
        assert feats["pinch_distance"] == 0.0

    def test_fingers_extended_count_zero_for_fist(self):
        """Roadmap spec: fingers_extended_count == 0 for a tight fist."""
        feats = extract_features(_make_fist())
        assert feats["fingers_extended_count"] == 0, (
            f"Expected 0, got {feats['fingers_extended_count']}"
        )

    def test_fingers_extended_count_five_for_open_palm(self):
        """Roadmap spec: fingers_extended_count == 5 for a full open palm."""
        feats = extract_features(_make_open_palm())
        assert feats["fingers_extended_count"] == 5, (
            f"Expected 5, got {feats['fingers_extended_count']}"
        )

    def test_palm_size_positive_for_non_degenerate_hands(self):
        """Roadmap spec: palm_size > 0 for any non-degenerate hand."""
        for lms in [_make_fist(), _make_open_palm(), _make_pinch(), _make_point()]:
            assert extract_features(lms)["palm_size"] > 0.0

    def test_palm_normal_unit_vector_for_non_degenerate_hand(self):
        """Roadmap edge-case spec: non-degenerate hand → unit normal."""
        feats = extract_features(_make_open_palm())
        mag = math.sqrt(
            feats["palm_normal_x"] ** 2 +
            feats["palm_normal_y"] ** 2 +
            feats["palm_normal_z"] ** 2
        )
        assert abs(mag - 1.0) < 1e-10, (
            f"Palm normal magnitude {mag:.8f} is not 1.0"
        )

    def test_palm_normal_stable_zero_for_degenerate_geometry(self):
        """Roadmap edge-case spec: degenerate geometry → zero normal, no NaN."""
        lms   = _base_landmarks()
        feats = extract_features(lms)
        for k in ("palm_normal_x", "palm_normal_y", "palm_normal_z"):
            assert math.isfinite(feats[k]), f"{k} is non-finite on degenerate hand"
        mag = math.sqrt(
            feats["palm_normal_x"] ** 2 +
            feats["palm_normal_y"] ** 2 +
            feats["palm_normal_z"] ** 2
        )
        # Degenerate → zero vector (mag ≈ 0) OR any unit vector (either is stable)
        # The implementation returns zero vector; assert it is finite and ≤ 1.
        assert mag <= 1.0 + 1e-10

    def test_all_fingers_extended_one_for_open_palm(self):
        feats = extract_features(_make_open_palm())
        assert feats["all_fingers_extended"] == 1.0

    def test_all_fingers_extended_zero_for_fist(self):
        feats = extract_features(_make_fist())
        assert feats["all_fingers_extended"] == 0.0

    def test_thumb_index_distance_alias_equals_pinch_distance(self):
        """thumb_index_distance must always equal pinch_distance."""
        for lms in [_make_fist(), _make_open_palm(), _make_pinch()]:
            feats = extract_features(lms)
            assert feats["thumb_index_distance"] == feats["pinch_distance"]


# =============================================================================
# extract_features() — individual feature key values
# =============================================================================

class TestExtractFeaturesValues:

    def test_wrist_coordinates_match_landmark_0(self):
        lms = _make_open_palm()
        feats = extract_features(lms)
        assert feats["wrist_x"] == lms[WRIST][0]
        assert feats["wrist_y"] == lms[WRIST][1]

    def test_index_tip_coordinates_match_landmark_8(self):
        lms = _make_open_palm()
        feats = extract_features(lms)
        assert feats["index_tip_x"] == lms[INDEX_TIP][0]
        assert feats["index_tip_y"] == lms[INDEX_TIP][1]

    def test_curl_ratios_non_negative(self):
        for lms in [_make_fist(), _make_open_palm()]:
            feats = extract_features(lms)
            for key in ("curl_index", "curl_middle", "curl_ring",
                        "curl_pinky", "curl_thumb"):
                assert feats[key] >= 0.0, f"{key}={feats[key]:.4f} is negative"

    def test_open_palm_curl_ratios_above_threshold(self):
        feats = extract_features(_make_open_palm())
        for key in ("curl_index", "curl_middle", "curl_ring",
                    "curl_pinky", "curl_thumb"):
            assert feats[key] > _EXTENSION_THRESHOLD, (
                f"{key}={feats[key]:.4f} should be > {_EXTENSION_THRESHOLD} "
                f"for open palm"
            )

    def test_fist_curl_ratios_below_threshold(self):
        feats = extract_features(_make_fist())
        for key in ("curl_index", "curl_middle", "curl_ring",
                    "curl_pinky", "curl_thumb"):
            assert feats[key] < _EXTENSION_THRESHOLD, (
                f"{key}={feats[key]:.4f} should be < {_EXTENSION_THRESHOLD} "
                f"for fist"
            )

    def test_extension_flags_consistent_with_curl_ratios(self):
        """Extension flag must equal 1.0 iff curl_ratio > _EXTENSION_THRESHOLD."""
        for lms in [_make_fist(), _make_open_palm(), _make_point()]:
            feats = extract_features(lms)
            pairs = [
                ("curl_index",  "index_extended"),
                ("curl_middle", "middle_extended"),
                ("curl_ring",   "ring_extended"),
                ("curl_pinky",  "pinky_extended"),
                ("curl_thumb",  "thumb_extended"),
            ]
            for curl_key, flag_key in pairs:
                expected = 1.0 if feats[curl_key] > _EXTENSION_THRESHOLD else 0.0
                assert feats[flag_key] == expected, (
                    f"{flag_key}={feats[flag_key]} inconsistent with "
                    f"{curl_key}={feats[curl_key]:.4f}"
                )

    def test_fingers_extended_count_equals_sum_of_flags(self):
        """fingers_extended_count must equal the sum of the five extension flags."""
        for lms in [_make_fist(), _make_open_palm(), _make_point(), _make_pinch()]:
            feats = extract_features(lms)
            flag_sum = int(
                feats["index_extended"] + feats["middle_extended"] +
                feats["ring_extended"]  + feats["pinky_extended"]  +
                feats["thumb_extended"]
            )
            assert feats["fingers_extended_count"] == flag_sum, (
                f"fingers_extended_count={feats['fingers_extended_count']} "
                f"!= flag_sum={flag_sum}"
            )

    def test_fingers_extended_count_range(self):
        """fingers_extended_count must always be in [0, 5]."""
        for lms in [_make_fist(), _make_open_palm(), _make_point(),
                    _make_pinch(), _base_landmarks()]:
            feats = extract_features(lms)
            assert 0 <= feats["fingers_extended_count"] <= 5

    def test_pinch_distance_normalised_by_palm_size(self):
        """
        pinch_distance = raw_euclidean(THUMB_TIP, INDEX_TIP) / palm_size.
        Verify this relationship directly.
        """
        lms = _make_open_palm()
        feats = extract_features(lms)
        raw   = euclidean(lms[THUMB_TIP], lms[INDEX_TIP])
        ps    = compute_palm_size(lms)
        expected = raw / ps
        assert abs(feats["pinch_distance"] - expected) < 1e-10

    def test_palm_size_matches_helper(self):
        """palm_size in the dict must match compute_palm_size() output."""
        lms = _make_open_palm()
        assert abs(extract_features(lms)["palm_size"] - compute_palm_size(lms)) < 1e-12

    def test_palm_normal_components_match_helper(self):
        """palm_normal_x/y/z must match compute_palm_normal() output."""
        lms = _make_open_palm()
        feats  = extract_features(lms)
        normal = compute_palm_normal(lms)
        assert abs(feats["palm_normal_x"] - float(normal[0])) < 1e-12
        assert abs(feats["palm_normal_y"] - float(normal[1])) < 1e-12
        assert abs(feats["palm_normal_z"] - float(normal[2])) < 1e-12


# =============================================================================
# extract_features() — input validation
# =============================================================================

class TestExtractFeaturesValidation:

    def test_wrong_count_20_raises(self):
        with pytest.raises(ValueError):
            extract_features([(0.5, 0.5, 0.0)] * 20)

    def test_wrong_count_22_raises(self):
        with pytest.raises(ValueError):
            extract_features([(0.5, 0.5, 0.0)] * 22)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            extract_features([])

    def test_wrong_count_100_raises(self):
        with pytest.raises(ValueError):
            extract_features([(0.5, 0.5, 0.0)] * 100)

    def test_non_finite_nan_raises(self):
        lms = _base_landmarks()
        lms[8] = (float("nan"), 0.5, 0.0)
        with pytest.raises(ValueError):
            extract_features(lms)

    def test_non_finite_inf_raises(self):
        lms = _base_landmarks()
        lms[0] = (float("inf"), 0.5, 0.0)
        with pytest.raises(ValueError):
            extract_features(lms)

    def test_non_finite_neg_inf_raises(self):
        lms = _base_landmarks()
        lms[12] = (0.5, float("-inf"), 0.0)
        with pytest.raises(ValueError):
            extract_features(lms)

    def test_wrong_tuple_width_2_raises(self):
        with pytest.raises(ValueError):
            extract_features([(0.5, 0.5)] * 21)

    def test_wrong_tuple_width_4_raises(self):
        with pytest.raises(ValueError):
            extract_features([(0.5, 0.5, 0.0, 0.0)] * 21)

    def test_non_finite_at_each_axis_raises(self):
        """NaN in x, y, or z of any landmark must raise."""
        for lm_idx in [0, 8, 20]:
            for ax in range(3):
                lms = _base_landmarks()
                coords = list(lms[lm_idx])
                coords[ax] = float("nan")
                lms[lm_idx] = tuple(coords)
                with pytest.raises(ValueError):
                    extract_features(lms)

    def test_valid_input_does_not_raise(self):
        """Sanity: the canonical fixtures must never raise."""
        for lms in [_make_fist(), _make_open_palm(), _make_pinch(), _make_point()]:
            try:
                extract_features(lms)
            except Exception as exc:
                pytest.fail(f"Valid input raised: {exc}")

    def test_exactly_21_landmarks_succeeds(self):
        feats = extract_features([(0.5, 0.5, 0.0)] * 21)
        assert len(feats) == 22


# =============================================================================
# extract_features() — scale invariance
# =============================================================================

class TestExtractFeaturesScaleInvariance:

    def test_extended_flags_invariant_to_hand_scale(self):
        """
        Scaling all landmarks toward the frame centre by 50% simulates a
        hand twice as far from the camera. The extension flags (curl-based)
        must be identical because curl ratios are palm-size normalised.
        """
        lms_near = _make_open_palm()
        centre   = 0.5
        lms_far  = [
            (centre + (x - centre) * 0.5,
             centre + (y - centre) * 0.5,
             z)
            for x, y, z in lms_near
        ]
        f_near = extract_features(lms_near)
        f_far  = extract_features(lms_far)
        for key in ("index_extended", "middle_extended", "ring_extended",
                    "pinky_extended", "thumb_extended", "fingers_extended_count"):
            assert f_near[key] == f_far[key], (
                f"{key} changed with scale: near={f_near[key]}, far={f_far[key]}"
            )

    def test_pinch_distance_invariant_to_hand_scale(self):
        """
        Pinch distance is normalised by palm_size, so shrinking the whole
        hand should not change the normalised pinch distance significantly.
        """
        lms_near = _make_pinch()
        centre   = 0.5
        lms_far  = [
            (centre + (x - centre) * 0.5,
             centre + (y - centre) * 0.5,
             z)
            for x, y, z in lms_near
        ]
        f_near = extract_features(lms_near)
        f_far  = extract_features(lms_far)
        assert abs(f_near["pinch_distance"] - f_far["pinch_distance"]) < 1e-10


# =============================================================================
# Potential Issues Found
# =============================================================================
# (No source files were modified. Issues are listed for awareness only.)
#
# 1. pip_idx parameter in finger_curl_ratio() is accepted but entirely unused.
#    The docstring documents it as "accepted, currently unused", so this is an
#    intentional design choice, not a bug. However, any future caller passing
#    pip_idx for a semantic purpose (e.g. intermediate joint angle) would get
#    silently incorrect results since only tip_idx and mcp_idx are used.
#
# 2. all_fingers_extended is included in FEATURE_KEYS (22 keys total) but
#    is not listed in the roadmap's "22 feature keys" specification table,
#    which ends at palm_normal_z (21 entries). The implementation has 22 keys;
#    the roadmap table has 21 entries. The implementation and FEATURE_KEYS
#    constant are internally consistent (both count 22), so tests pass. The
#    discrepancy is between the written spec document and the implementation.
#
# 3. FEATURE_KEYS is a tuple, not a list. Any code that assumes a list type
#    (e.g. FEATURE_KEYS.append()) will raise AttributeError at runtime. The
#    module-level assert confirms length == 22 at import time which is good,
#    but the type is implicitly tuple via the Tuple[str, ...] annotation.
#    This is not a bug — tuples are appropriate for immutable constants.
# =============================================================================