# =============================================================================
# perception/gesture/classifier.py
# AirSign — Gesture Classifier
# -----------------------------------------------------------------------------
# Accepts a feature dict from extract_features() and returns a
# (gesture_label, confidence) tuple. Two backends are provided:
#
#   RuleBasedClassifier — deterministic, zero-training, always available.
#                         Used as a fallback before a model has been trained
#                         and as a sanity baseline during development.
#
#   MLPClassifier        — loads a scikit-learn Pipeline trained by
#                         trainer.py. Higher accuracy on ambiguous poses
#                         but requires data/collector.py + trainer.py to
#                         have been run first.
#
#   GestureClassifier    — facade used by InferenceThread. Attempts to load
#                         the MLP backend; falls back to RuleBasedClassifier
#                         if no trained model file is present.
#
# Design contract:
#   - Feature vector column order for the MLP backend is derived exclusively
#     from FEATURE_KEYS (feature_extractor.py) — never hardcoded.
#   - All returned gesture labels are members of GESTURE_LABELS.
#   - Confidence values are always in [0.0, 1.0].
#   - Below MIN_ACTION_CONFIDENCE, both backends return "UNKNOWN".
# =============================================================================

from __future__ import annotations

import os
from typing import List, Tuple

import joblib
import numpy as np

from perception.gesture.feature_extractor import FEATURE_KEYS
from shared.constants import DEFAULT_PINCH_TOLERANCE, MIN_ACTION_CONFIDENCE
from shared.gesture_labels import GESTURE_LABELS

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_DIR: str = os.path.dirname(__file__)
_DEFAULT_MODEL_PATH: str = os.path.join(
    _MODULE_DIR, "models", "gesture_mlp.joblib"
)
_DEFAULT_ENCODER_PATH: str = os.path.join(
    _MODULE_DIR, "models", "label_encoder.joblib"
)

FeatureDict = dict


# =============================================================================
# Rule-Based Classifier
# =============================================================================

class RuleBasedClassifier:
    """
    Deterministic gesture classifier using explicit geometric thresholds.

    Operates purely on the feature dict produced by ``extract_features()``.
    Requires no model file and no training data — always available as a
    fallback. Rules are evaluated in a fixed priority order; the first rule
    that matches determines the output label and confidence.

    This classifier is intentionally conservative and interpretable: every
    decision can be explained in one sentence, which makes it useful both as
    a development-time baseline and as a safety fallback if the MLP model
    file is missing or fails to load.
    """

    def classify(self, features: FeatureDict) -> Tuple[str, float]:
        """
        Classify a feature dict into a gesture label and confidence score.

        Evaluation order (first match wins):
            1. PINCH       — pinch_distance below DEFAULT_PINCH_TOLERANCE.
            2. FIST        — zero fingers extended.
            3. OPEN_PALM   — all five fingers extended.
            4. POINT       — only index extended.
            5. TWO_FINGER  — index + middle extended, others curled.
            6. THUMBS_UP   — only thumb extended.
            7. UNKNOWN     — no rule matched with sufficient confidence.

        Args:
            features: Output dict of ``extract_features()``. Must contain
                      every key referenced by the rules below (this is
                      guaranteed by ``extract_features()``'s 22-key
                      contract).

        Returns:
            Tuple of ``(gesture_label, confidence)`` where ``gesture_label``
            is a member of ``GESTURE_LABELS`` and ``confidence`` is a float
            in ``[0.0, 1.0]``.
        """
        pinch_distance = features["pinch_distance"]
        fingers_extended_count = features["fingers_extended_count"]
        index_extended  = features["index_extended"]
        middle_extended = features["middle_extended"]
        ring_extended   = features["ring_extended"]
        pinky_extended  = features["pinky_extended"]
        thumb_extended  = features["thumb_extended"]

        # ── Rule 1: PINCH ────────────────────────────────────────────────
        if pinch_distance < DEFAULT_PINCH_TOLERANCE:
            # Confidence scales inversely with distance: a tighter pinch
            # (smaller distance) yields higher confidence. Clamped to [0, 1].
            confidence = 1.0 - (pinch_distance / DEFAULT_PINCH_TOLERANCE) * 0.4
            confidence = max(0.0, min(1.0, confidence))
            if confidence >= MIN_ACTION_CONFIDENCE:
                return "PINCH", confidence

        # ── Rule 2: FIST ─────────────────────────────────────────────────
        if fingers_extended_count == 0:
            return "FIST", 0.95

        # ── Rule 3: OPEN_PALM ────────────────────────────────────────────
        if fingers_extended_count == 5:
            return "OPEN_PALM", 0.95

        # ── Rule 4: POINT ────────────────────────────────────────────────
        if (
            index_extended == 1.0
            and middle_extended == 0.0
            and ring_extended == 0.0
            and pinky_extended == 0.0
        ):
            return "POINT", 0.90

        # ── Rule 5: TWO_FINGER ───────────────────────────────────────────
        if (
            index_extended == 1.0
            and middle_extended == 1.0
            and ring_extended == 0.0
            and pinky_extended == 0.0
        ):
            return "TWO_FINGER", 0.90

        # ── Rule 6: THUMBS_UP ────────────────────────────────────────────
        if (
            thumb_extended == 1.0
            and fingers_extended_count == 1
        ):
            return "THUMBS_UP", 0.85

        # ── Rule 7: No match ─────────────────────────────────────────────
        return "UNKNOWN", 0.0


# =============================================================================
# MLP Classifier
# =============================================================================

class MLPClassifier:
    """
    Trained MLP-based gesture classifier.

    Loads a scikit-learn ``Pipeline`` (StandardScaler → MLPClassifier)
    produced by ``perception/gesture/trainer.py``, along with the ordered
    list of class name strings the pipeline was trained on.

    The feature vector passed to the pipeline is built strictly from
    ``FEATURE_KEYS`` order — this guarantees the column layout used at
    inference time always matches the layout used during training, since
    both read from the same constant.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        encoder_path: str = _DEFAULT_ENCODER_PATH,
    ) -> None:
        """
        Load the trained pipeline and label encoder from disk.

        Args:
            model_path:   Path to the joblib-serialised sklearn Pipeline,
                          as saved by ``trainer.save_model()``.
            encoder_path: Path to the joblib-serialised list of class name
                          strings, as saved by ``trainer.save_model()``.

        Raises:
            FileNotFoundError: If either file does not exist. The message
                               instructs the caller to run
                               ``data/collector.py`` and
                               ``perception/gesture/trainer.py`` first.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MLP model file not found: {model_path!r}. "
                f"Run data/collector.py to collect training samples, then "
                f"perception/gesture/trainer.py to train and save a model."
            )
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"Label encoder file not found: {encoder_path!r}. "
                f"This file is produced alongside the model by "
                f"perception/gesture/trainer.py — re-run training if only "
                f"the model file is present."
            )

        self._model = joblib.load(model_path)
        self._label_classes: List[str] = list(joblib.load(encoder_path))

        if len(self._label_classes) == 0:
            raise ValueError(
                f"Label encoder at {encoder_path!r} is empty. "
                f"Re-run trainer.py — this indicates a corrupted or "
                f"incomplete training save."
            )

    def classify(self, features: FeatureDict) -> Tuple[str, float]:
        """
        Classify a feature dict using the trained MLP pipeline.

        Builds the feature vector in ``FEATURE_KEYS`` order, runs
        ``predict_proba()``, and returns the highest-probability class
        together with its probability. If that probability falls below
        ``MIN_ACTION_CONFIDENCE``, returns ``("UNKNOWN", probability)``
        instead — the caller can inspect the returned confidence to see
        how close the classifier was to a confident decision.

        Args:
            features: Output dict of ``extract_features()``. Must contain
                      every key in ``FEATURE_KEYS``.

        Returns:
            Tuple of ``(gesture_label, confidence)``. ``gesture_label`` is
            either one of the trained class names or ``"UNKNOWN"``.
            ``confidence`` is the model's predicted probability for the
            winning class, in ``[0.0, 1.0]``.

        Raises:
            KeyError: If ``features`` is missing any key in ``FEATURE_KEYS``.
        """
        feature_vector = np.array(
            [[features[key] for key in FEATURE_KEYS]], dtype=np.float64
        )

        probabilities = self._model.predict_proba(feature_vector)[0]
        best_idx = int(np.argmax(probabilities))
        best_label = self._label_classes[best_idx]
        best_confidence = float(probabilities[best_idx])

        if best_confidence < MIN_ACTION_CONFIDENCE:
            return "UNKNOWN", best_confidence

        return best_label, best_confidence

    def is_loaded(self) -> bool:
        """
        Return True if the model and encoder were loaded successfully.

        Since ``__init__`` raises on any load failure, a successfully
        constructed instance is always loaded. This method exists for
        API symmetry with ``GestureClassifier`` and for explicit
        readability at call sites.

        Returns:
            Always ``True`` for a successfully constructed instance.
        """
        return True

    @property
    def label_classes(self) -> List[str]:
        """
        Ordered list of gesture label strings the model was trained on.

        Returns:
            Copy of the internal label class list, safe to mutate.
        """
        return list(self._label_classes)


# =============================================================================
# Facade — used by InferenceThread
# =============================================================================

class GestureClassifier:
    """
    Facade classifier that prefers the trained MLP backend and transparently
    falls back to the deterministic rule-based backend if no trained model
    is available.

    This is the class instantiated by ``InferenceThread``. Callers should
    not need to know which backend is active except for status-bar display
    purposes, exposed via ``backend_type``.

    Attributes:
        backend_type (str): ``"mlp"`` if the trained model loaded
                            successfully, ``"rule_based"`` otherwise.
    """

    def __init__(
        self,
        use_mlp: bool = True,
        model_path: str = _DEFAULT_MODEL_PATH,
        encoder_path: str = _DEFAULT_ENCODER_PATH,
    ) -> None:
        """
        Initialise the facade, attempting the MLP backend first.

        Args:
            use_mlp:      If ``True``, attempt to load the MLP backend. If
                          loading fails for any reason (missing files,
                          corrupted data), silently falls back to the
                          rule-based backend and prints a warning. If
                          ``False``, skips the MLP attempt entirely and
                          uses the rule-based backend from the start.
            model_path:   Path to the trained MLP pipeline file, forwarded
                          to ``MLPClassifier`` if ``use_mlp`` is ``True``.
            encoder_path: Path to the label encoder file, forwarded to
                          ``MLPClassifier`` if ``use_mlp`` is ``True``.
        """
        self._mlp_backend: MLPClassifier | None = None
        self._rule_backend: RuleBasedClassifier = RuleBasedClassifier()
        self.backend_type: str = "rule_based"

        if use_mlp:
            try:
                self._mlp_backend = MLPClassifier(
                    model_path=model_path, encoder_path=encoder_path
                )
                self.backend_type = "mlp"
            except (FileNotFoundError, ValueError) as exc:
                print(
                    f"Warning: could not load MLP classifier ({exc}). "
                    f"Falling back to RuleBasedClassifier."
                )
                self._mlp_backend = None
                self.backend_type = "rule_based"

    def classify(self, features: FeatureDict) -> Tuple[str, float]:
        """
        Classify a feature dict using the currently active backend.

        Args:
            features: Output dict of ``extract_features()``.

        Returns:
            Tuple of ``(gesture_label, confidence)`` from whichever backend
            is currently active (see ``backend_type``).
        """
        if self.backend_type == "mlp" and self._mlp_backend is not None:
            return self._mlp_backend.classify(features)
        return self._rule_backend.classify(features)

    def switch_backend(self, use_mlp: bool) -> None:
        """
        Hot-swap the active backend at runtime without restarting the thread.

        Args:
            use_mlp: If ``True``, attempts to switch to (or load, if not
                     already loaded) the MLP backend. If loading fails,
                     remains on the rule-based backend and prints a warning.
                     If ``False``, switches to the rule-based backend
                     immediately (no loading required, always succeeds).
        """
        if not use_mlp:
            self.backend_type = "rule_based"
            return

        if self._mlp_backend is not None:
            self.backend_type = "mlp"
            return

        try:
            self._mlp_backend = MLPClassifier()
            self.backend_type = "mlp"
        except (FileNotFoundError, ValueError) as exc:
            print(
                f"Warning: could not switch to MLP classifier ({exc}). "
                f"Remaining on RuleBasedClassifier."
            )
            self.backend_type = "rule_based"