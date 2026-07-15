# =============================================================================
# perception/gesture/feature_extractor.py
# AirSign — Hand Landmark Feature Extractor
# -----------------------------------------------------------------------------
# Converts a 21-point MediaPipe hand landmark array into a compact, named
# feature dictionary suitable for gesture classification and FSM evaluation.
#
# Design contract:
#   - Zero MediaPipe imports. Input is a plain Python list of 3-tuples.
#   - All distance features are normalised by palm_size to be scale-invariant
#     across different hand sizes and camera distances.
#   - extract_features() is the sole public entry point; all other functions
#     are module-level helpers and may be used independently in tests.
#   - Output dict always contains exactly 22 keys (see FEATURE_KEYS).
#   - No mutable global state. All functions are pure (no side effects).
#   - numpy is used only for cross-product computation in compute_palm_normal();
#     all other arithmetic is plain Python for minimal import overhead.
#
# Landmark index reference (MediaPipe Hands — 21 points):
#
#   Wrist:
#     0  WRIST
#
#   Thumb (4 joints):
#     1  THUMB_CMC   2  THUMB_MCP   3  THUMB_IP    4  THUMB_TIP
#
#   Index finger (4 joints):
#     5  INDEX_MCP   6  INDEX_PIP   7  INDEX_DIP   8  INDEX_TIP
#
#   Middle finger (4 joints):
#     9  MIDDLE_MCP  10 MIDDLE_PIP  11 MIDDLE_DIP  12 MIDDLE_TIP
#
#   Ring finger (4 joints):
#     13 RING_MCP    14 RING_PIP    15 RING_DIP    16 RING_TIP
#
#   Pinky finger (4 joints):
#     17 PINKY_MCP   18 PINKY_PIP   19 PINKY_DIP   20 PINKY_TIP
#
# Usage::
#
#   from perception.gesture.feature_extractor import extract_features
#
#   # landmarks is a list of 21 (x, y, z) tuples, normalised 0.0–1.0
#   features = extract_features(landmarks)
#
#   # features["pinch_distance"]   → float, normalised by palm size
#   # features["curl_index"]       → float, curl ratio for index finger
#   # features["fingers_extended_count"] → int, 0–5
# =============================================================================

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from shared.constants import (
    DEFAULT_PINCH_TOLERANCE,
    DISTANCE_EPSILON,
)

# ---------------------------------------------------------------------------
# Landmark index constants
# Defined at module level so any module can import them without re-typing.
# These correspond 1:1 with MediaPipe Hands landmark indices.
# ---------------------------------------------------------------------------

# Wrist
WRIST: int = 0

# Thumb
THUMB_CMC: int = 1
THUMB_MCP: int = 2
THUMB_IP:  int = 3
THUMB_TIP: int = 4

# Index finger
INDEX_MCP: int = 5
INDEX_PIP: int = 6
INDEX_DIP: int = 7
INDEX_TIP: int = 8

# Middle finger
MIDDLE_MCP: int = 9
MIDDLE_PIP: int = 10
MIDDLE_DIP: int = 11
MIDDLE_TIP: int = 12

# Ring finger
RING_MCP: int = 13
RING_PIP: int = 14
RING_DIP: int = 15
RING_TIP: int = 16

# Pinky finger
PINKY_MCP: int = 17
PINKY_PIP: int = 18
PINKY_DIP: int = 19
PINKY_TIP: int = 20

# Total expected landmarks
_NUM_LANDMARKS: int = 21

# ---------------------------------------------------------------------------
# Extension threshold
# A finger curl ratio above this value is considered "extended".
# Used to compute the boolean-ish extended flags and fingers_extended_count.
# Kept here (not in constants.py) because it is an internal classifier
# heuristic, not a tunable runtime parameter.
# ---------------------------------------------------------------------------
_EXTENSION_THRESHOLD: float = 0.9

# ---------------------------------------------------------------------------
# Ordered list of all 22 feature keys produced by extract_features().
# Used by classifier.py to build the fixed-column numpy array for the MLP,
# and by tests to assert the output dict is complete and consistent.
# ---------------------------------------------------------------------------
FEATURE_KEYS: Tuple[str, ...] = (
    "curl_index",
    "curl_middle",
    "curl_ring",
    "curl_pinky",
    "curl_thumb",
    "pinch_distance",
    "thumb_index_distance",
    "palm_size",
    "index_extended",
    "middle_extended",
    "ring_extended",
    "pinky_extended",
    "thumb_extended",
    "wrist_x",
    "wrist_y",
    "index_tip_x",
    "index_tip_y",
    "palm_normal_x",
    "palm_normal_y",
    "palm_normal_z",
    "fingers_extended_count",
    "all_fingers_extended",
)

# Confirm 22 keys at import time — catches accidental edits.
assert len(FEATURE_KEYS) == 22, (
    f"FEATURE_KEYS must contain exactly 22 entries, found {len(FEATURE_KEYS)}."
)

# Type alias
Landmark     = Tuple[float, float, float]
LandmarkList = List[Landmark]
FeatureDict  = Dict[str, float]


# =============================================================================
# Pure geometry helpers
# =============================================================================

def euclidean(a: Landmark, b: Landmark) -> float:
    """
    Compute the 3D Euclidean distance between two landmarks.

    Both landmarks are expected to be 3-tuples of normalised floats (x, y, z).
    The function operates purely on the three spatial axes without any
    coordinate-frame assumptions.

    Args:
        a: First point as ``(x, y, z)``.
        b: Second point as ``(x, y, z)``.

    Returns:
        Non-negative float distance. Returns 0.0 if ``a == b``.

    Example::

        euclidean((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))  # → 1.0
        euclidean((0.5, 0.5, 0.0), (0.5, 0.5, 0.0))  # → 0.0
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_palm_size(landmarks: LandmarkList) -> float:
    """
    Compute the palm size as the Euclidean distance from WRIST to MIDDLE_MCP.

    This distance is a stable, scale-invariant reference for the hand because:
    - The wrist–middle-MCP segment lies along the central axis of the palm.
    - It does not change with finger extension or curl (unlike tip-to-tip
      measurements).
    - It scales linearly with hand-to-camera distance, making it a reliable
      normaliser for all other distance features.

    Args:
        landmarks: 21-item list of ``(x, y, z)`` tuples.

    Returns:
        Positive float. In degenerate cases where the hand is perfectly flat
        or the two landmarks coincide, returns ``DISTANCE_EPSILON`` to prevent
        downstream division-by-zero.
    """
    size = euclidean(landmarks[WRIST], landmarks[MIDDLE_MCP])
    return max(size, DISTANCE_EPSILON)


def finger_curl_ratio(
    landmarks: LandmarkList,
    tip_idx:   int,
    pip_idx:   int,
    mcp_idx:   int,
) -> float:
    """
    Compute how extended (uncurled) a finger is using a distance ratio.

    Intuition:
        When a finger is fully extended, its tip is far from the wrist —
        approximately as far as or farther than its MCP joint.
        When fully curled, the tip folds back toward the palm, making the
        tip-to-wrist distance shorter than the MCP-to-wrist distance.

    Formula:
        curl_ratio = dist(tip, wrist) / (dist(mcp, wrist) + ε)

    Interpretation:
        > 1.0 : finger fully extended (tip further from wrist than MCP)
        ≈ 1.0 : finger partially extended
        < 0.6 : finger significantly curled
        ≈ 0.0 : finger tightly curled (tip nearly at wrist)

    The PIP index is accepted as a parameter for potential future use
    (e.g. computing intermediate joint angles) but is currently unused in
    this ratio-based implementation. It is kept in the signature to maintain
    a consistent interface across all finger calls.

    Args:
        landmarks: 21-item list of ``(x, y, z)`` tuples.
        tip_idx:   Landmark index of the fingertip (e.g. ``INDEX_TIP = 8``).
        pip_idx:   Landmark index of the PIP joint (accepted, currently unused).
        mcp_idx:   Landmark index of the MCP (knuckle) joint.

    Returns:
        Non-negative float. Typical range 0.0–1.6. Not clamped — values
        slightly above 1.5 are physically possible for hyper-extended fingers.
    """
    wrist     = landmarks[WRIST]
    tip       = landmarks[tip_idx]
    mcp       = landmarks[mcp_idx]

    dist_tip_wrist = euclidean(tip,  wrist)
    dist_mcp_wrist = euclidean(mcp,  wrist)

    return dist_tip_wrist / (dist_mcp_wrist + DISTANCE_EPSILON)


def compute_thumb_curl(landmarks: LandmarkList) -> float:
    """
    Compute the thumb extension ratio using a thumb-specific landmark chain.

    The thumb has a different kinematic structure from the four fingers:
    it opposes the palm rather than extending parallel to it. A direct
    tip-to-wrist / MCP-to-wrist ratio gives noisy results because the thumb
    CMC joint (base of thumb, index 1) rotates the entire thumb away from
    the palm axis.

    Instead, this function uses:
        curl_ratio = dist(THUMB_TIP, WRIST) / (dist(THUMB_IP, WRIST) + ε)

    where THUMB_IP (index 3) acts as the reference joint. When the thumb is
    fully extended, the tip travels past the IP joint relative to the wrist;
    when curled inward, the tip approaches or crosses behind the IP joint.

    Args:
        landmarks: 21-item list of ``(x, y, z)`` tuples.

    Returns:
        Non-negative float. Analogous range to ``finger_curl_ratio()``.
    """
    wrist    = landmarks[WRIST]
    tip      = landmarks[THUMB_TIP]
    ip_joint = landmarks[THUMB_IP]

    dist_tip_wrist = euclidean(tip,      wrist)
    dist_ip_wrist  = euclidean(ip_joint, wrist)

    return dist_tip_wrist / (dist_ip_wrist + DISTANCE_EPSILON)


def compute_palm_normal(landmarks: LandmarkList) -> np.ndarray:
    """
    Estimate the outward-facing palm normal vector as a unit 3D vector.

    The normal is computed from two vectors that lie in the palm plane:
        v1 = INDEX_MCP  − WRIST   (proximal axis, wrist → index knuckle)
        v2 = PINKY_MCP  − WRIST   (lateral axis,  wrist → pinky knuckle)

    The cross product ``v1 × v2`` gives a vector perpendicular to the palm
    plane. For a right hand with palm facing the camera, this vector points
    toward the camera (positive z direction in MediaPipe's normalised space).

    The result is normalised to a unit vector. If the two input vectors are
    parallel (degenerate hand pose), the function returns a zero vector rather
    than raising, to allow downstream callers to handle the edge case.

    Args:
        landmarks: 21-item list of ``(x, y, z)`` tuples.

    Returns:
        ``numpy.ndarray`` of shape ``(3,)``, dtype ``float64``.
        Unit vector in the direction of the outward palm normal, or
        ``np.zeros(3)`` for degenerate (colinear) landmark configurations.

    Note:
        The z-component of the normal in MediaPipe landmark space correlates
        with how directly the palm faces the camera. It is used by the FSM
        to detect a deliberate "palm push" gesture (large positive z-delta).
    """
    wrist     = np.array(landmarks[WRIST],      dtype=np.float64)
    index_mcp = np.array(landmarks[INDEX_MCP],  dtype=np.float64)
    pinky_mcp = np.array(landmarks[PINKY_MCP],  dtype=np.float64)

    v1 = index_mcp - wrist   # proximal palm axis
    v2 = pinky_mcp - wrist   # lateral palm axis

    normal = np.cross(v1, v2)

    norm_magnitude = np.linalg.norm(normal)
    if norm_magnitude < DISTANCE_EPSILON:
        # Degenerate: landmarks are colinear — return zero vector, not NaN.
        return np.zeros(3, dtype=np.float64)

    return normal / norm_magnitude


def _extension_flag(curl_ratio: float) -> float:
    """
    Convert a curl ratio to a binary-ish extension flag.

    Returns 1.0 if the curl_ratio exceeds ``_EXTENSION_THRESHOLD``,
    indicating the finger is considered extended; 0.0 otherwise.

    Kept as float (not bool) so the value can be fed directly into the
    MLP classifier's numpy feature array without type conversion.

    Args:
        curl_ratio: Output of ``finger_curl_ratio()`` or ``compute_thumb_curl()``.

    Returns:
        1.0 if extended, 0.0 if curled.
    """
    return 1.0 if curl_ratio > _EXTENSION_THRESHOLD else 0.0


# =============================================================================
# Primary public entry point
# =============================================================================

def extract_features(landmarks: LandmarkList) -> FeatureDict:
    """
    Convert a 21-point hand landmark array into the AirSign feature dict.

    This is the sole public entry point called by ``InferenceThread`` on
    every processed frame (after smoothing). The returned dict is consumed by:
    - ``GestureClassifier.classify()``   — static gesture recognition
    - ``GestureFSM.update()``            — temporal state machine logic
    - ``data/collector.py``              — training data recording

    All distance-based features are normalised by ``palm_size`` to be
    scale-invariant across different hand sizes and camera distances.

    Feature pipeline:
        1. Validate input (length == 21, all values finite).
        2. Compute palm_size as the wrist-to-middle-MCP distance.
        3. Compute per-finger curl ratios using tip/MCP/wrist distances.
        4. Compute thumb curl using its IP joint as reference.
        5. Compute pinch distance (THUMB_TIP to INDEX_TIP), normalised.
        6. Compute palm normal via cross product of palm-plane vectors.
        7. Derive binary extension flags from curl ratios.
        8. Assemble and return the 22-key dict.

    Args:
        landmarks: 21-item list of ``(x, y, z)`` tuples. All coordinate
                   values should be normalised to [0.0, 1.0] by MediaPipe.
                   The list must contain exactly 21 entries; each entry
                   must be a 3-element sequence of finite floats.

    Returns:
        Dict with exactly 22 keys as enumerated in ``FEATURE_KEYS``.
        All values are Python floats or ints (no numpy scalars) to
        ensure JSON-serialisability for the data collector CSV writer.

    Raises:
        ValueError: If ``landmarks`` does not contain exactly 21 entries.
        ValueError: If any landmark coordinate is non-finite (NaN or inf).

    Example::

        from perception.gesture.feature_extractor import extract_features

        # Simulated flat open palm (all landmarks at z=0)
        lms = [(float(i % 5) * 0.1, float(i // 5) * 0.1, 0.0)
               for i in range(21)]
        features = extract_features(lms)
        assert "pinch_distance" in features
        assert features["fingers_extended_count"] in range(6)
    """
    # ── 1. Input validation ──────────────────────────────────────────────────
    if len(landmarks) != _NUM_LANDMARKS:
        raise ValueError(
            f"extract_features() expects exactly {_NUM_LANDMARKS} landmarks, "
            f"got {len(landmarks)}. Ensure MediaPipe max_num_hands=1 and "
            f"that the landmark list is extracted before calling this function."
        )

    for idx, lm in enumerate(landmarks):
        if len(lm) != 3:
            raise ValueError(
                f"Landmark {idx} has {len(lm)} components; expected 3 (x, y, z)."
            )
        for ax, val in enumerate(lm):
            if not math.isfinite(val):
                axis_name = ("x", "y", "z")[ax]
                raise ValueError(
                    f"Non-finite value at landmark {idx}, axis '{axis_name}': "
                    f"{val!r}. Check MediaPipe output for degenerate frames."
                )

    # ── 2. Palm size (scale normaliser) ──────────────────────────────────────
    palm_size: float = compute_palm_size(landmarks)

    # ── 3. Per-finger curl ratios ─────────────────────────────────────────────
    curl_index: float = finger_curl_ratio(
        landmarks, tip_idx=INDEX_TIP,  pip_idx=INDEX_PIP,  mcp_idx=INDEX_MCP
    )
    curl_middle: float = finger_curl_ratio(
        landmarks, tip_idx=MIDDLE_TIP, pip_idx=MIDDLE_PIP, mcp_idx=MIDDLE_MCP
    )
    curl_ring: float = finger_curl_ratio(
        landmarks, tip_idx=RING_TIP,   pip_idx=RING_PIP,   mcp_idx=RING_MCP
    )
    curl_pinky: float = finger_curl_ratio(
        landmarks, tip_idx=PINKY_TIP,  pip_idx=PINKY_PIP,  mcp_idx=PINKY_MCP
    )

    # ── 4. Thumb curl (uses thumb-specific IP joint reference) ───────────────
    curl_thumb: float = compute_thumb_curl(landmarks)

    # ── 5. Pinch distance (THUMB_TIP ↔ INDEX_TIP, normalised by palm) ────────
    raw_pinch_distance: float = euclidean(
        landmarks[THUMB_TIP], landmarks[INDEX_TIP]
    )
    # Normalise by palm_size for scale invariance. A value < DEFAULT_PINCH_TOLERANCE
    # (0.15) indicates an active pinch gesture regardless of hand-camera distance.
    pinch_distance: float = raw_pinch_distance / palm_size

    # thumb_index_distance is an alias kept for FSM clarity — same value,
    # different semantic label used in FSM conditions.
    thumb_index_distance: float = pinch_distance

    # ── 6. Palm normal vector ─────────────────────────────────────────────────
    palm_normal: np.ndarray = compute_palm_normal(landmarks)

    # ── 7. Extension flags (float 0.0 or 1.0, MLP-compatible) ───────────────
    index_extended:  float = _extension_flag(curl_index)
    middle_extended: float = _extension_flag(curl_middle)
    ring_extended:   float = _extension_flag(curl_ring)
    pinky_extended:  float = _extension_flag(curl_pinky)
    thumb_extended:  float = _extension_flag(curl_thumb)

    # Sum of extended flags: 0 (fist) → 5 (open palm).
    # Kept as int for clarity in FSM rule conditions.
    fingers_extended_count: int = int(
        index_extended + middle_extended + ring_extended +
        pinky_extended + thumb_extended
    )

    # Convenience flag: True (1.0) only when all five fingers are extended.
    all_fingers_extended: float = 1.0 if fingers_extended_count == 5 else 0.0

    # ── 8. Assemble feature dict ──────────────────────────────────────────────
    # Landmark coordinate reads — raw normalised values (not screen-mapped).
    wrist_x:     float = float(landmarks[WRIST][0])
    wrist_y:     float = float(landmarks[WRIST][1])
    index_tip_x: float = float(landmarks[INDEX_TIP][0])
    index_tip_y: float = float(landmarks[INDEX_TIP][1])

    # Numpy scalar → Python float conversion for JSON/CSV serialisability.
    palm_normal_x: float = float(palm_normal[0])
    palm_normal_y: float = float(palm_normal[1])
    palm_normal_z: float = float(palm_normal[2])

    return {
        # ── Finger curl ratios ────────────────────────────────────────────────
        "curl_index":  curl_index,
        "curl_middle": curl_middle,
        "curl_ring":   curl_ring,
        "curl_pinky":  curl_pinky,
        "curl_thumb":  curl_thumb,

        # ── Pinch distance features ───────────────────────────────────────────
        "pinch_distance":       pinch_distance,
        "thumb_index_distance": thumb_index_distance,

        # ── Raw palm size (un-normalised, for diagnostic use) ─────────────────
        "palm_size": palm_size,

        # ── Extension flags (0.0 or 1.0) ─────────────────────────────────────
        "index_extended":  index_extended,
        "middle_extended": middle_extended,
        "ring_extended":   ring_extended,
        "pinky_extended":  pinky_extended,
        "thumb_extended":  thumb_extended,

        # ── Wrist position (raw normalised frame coordinates) ─────────────────
        "wrist_x": wrist_x,
        "wrist_y": wrist_y,

        # ── Index fingertip position (for swipe trajectory tracking in FSM) ───
        "index_tip_x": index_tip_x,
        "index_tip_y": index_tip_y,

        # ── Palm orientation normal vector ────────────────────────────────────
        "palm_normal_x": palm_normal_x,
        "palm_normal_y": palm_normal_y,
        "palm_normal_z": palm_normal_z,

        # ── Derived counts / convenience flags ───────────────────────────────
        "fingers_extended_count": fingers_extended_count,
        "all_fingers_extended":   all_fingers_extended,
    }