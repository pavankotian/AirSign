"""Mock perception producer for standalone Developer B UI validation.

This module simulates Developer A's ``InferenceThread`` output so the
entire Developer B stack (``ActionThread`` and every UI panel) can be
exercised end to end without a webcam, MediaPipe, OpenCV inference, or
a trained gesture classifier. :func:`mock_producer` generates
``RESULT_SCHEMA`` dictionaries on a background thread and pushes them
onto a ``queue.Queue`` exactly the way the real ``InferenceThread``
would.

Running this module directly (``python mocks/mock_perception.py
<scenario>``) launches the full application: it constructs a real
``SettingsManager``, a real ``MainWindow``, and a real
``ActionThread``, but wires that ``ActionThread`` to
:class:`_MockMouseInjector`, :class:`_MockKeyboardInjector`, and
:class:`_MockEventLogger` instead of the production
``MouseInjector``, ``KeyboardInjector``, and ``EventLogger``. This
harness therefore never moves the real OS cursor, never sends real
clicks or key presses, and never creates or modifies
``airsign_session.db`` -- every dispatched action is only logged via
the ``logging`` module, making it safe to run unattended for UI
validation.
"""

from __future__ import annotations

import logging
import queue
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw
from PyQt6.QtWidgets import QApplication

from application.action_thread import ActionThread
from application.config.bindings import load_bindings
from application.config.settings_manager import SettingsManager
from application.os_integration.coordinate_mapper import CoordinateMapper
from shared.constants import (
    DOUBLE_PINCH_MS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HAND_LOST_FRAMES,
    PINCH_HOLD_MS,
    RESULT_QUEUE_MAXSIZE,
    SWIPE_MIN_FRAMES,
    TARGET_FPS,
)
from ui.main_window import MainWindow

logger = logging.getLogger(__name__)

SCENARIOS: frozenset[str] = frozenset(
    {"demo", "idle", "jitter", "stress", "swipe", "pinch"}
)

_VIRTUAL_SCREEN_WIDTH = 1920
_VIRTUAL_SCREEN_HEIGHT = 1080

_PINCH_HOLD_FRAMES = max(1, round(PINCH_HOLD_MS / 1000.0 * TARGET_FPS))
_DOUBLE_PINCH_FRAMES = max(1, round(DOUBLE_PINCH_MS / 1000.0 * TARGET_FPS))
_HOVER_FRAMES = TARGET_FPS * 2
_SHORT_HOVER_FRAMES = max(1, TARGET_FPS // 2)
_SCROLL_FRAMES = TARGET_FPS * 2
_SWIPE_PENDING_FRAMES = SWIPE_MIN_FRAMES + 2
_HAND_LOST_DEMO_FRAMES = HAND_LOST_FRAMES + 5


@dataclass(frozen=True)
class _Phase:
    """One scripted segment of a scenario's repeating FSM timeline.

    Attributes:
        frames: Number of frames this phase lasts.
        fsm_state: The FSM state active for the duration of this
            phase.
        entry_event: The FSM event to report on this phase's first
            frame only, or ``None`` if no event fires on entry.
        static_gesture: The classified gesture label reported while
            this phase is active.
        confidence_range: Inclusive ``(low, high)`` bounds used to
            draw a random ``gesture_confidence`` value each frame.
        hand_detected: Whether a hand is considered present during
            this phase.
        cursor_target: Normalized ``(x, y)`` target position in
            ``[0.0, 1.0]`` the cursor simulator eases toward during
            this phase, or ``None`` to hold the simulator's current
            position steady.
        jitter_scale: Standard deviation, in normalized units, of the
            per-frame noise added to the raw (unfiltered) cursor
            position.
    """

    frames: int
    fsm_state: str
    entry_event: Optional[str]
    static_gesture: str
    confidence_range: tuple[float, float]
    hand_detected: bool
    cursor_target: Optional[tuple[float, float]]
    jitter_scale: float = 0.004


def _build_phase_table(scenario: str) -> list[_Phase]:
    """Builds the repeating list of scripted phases for a scenario.

    Args:
        scenario: One of the strings in :data:`SCENARIOS`.

    Returns:
        A list of :class:`_Phase` instances that
        :func:`mock_producer` cycles through indefinitely.

    Raises:
        ValueError: If ``scenario`` is not a member of
            :data:`SCENARIOS`.
    """
    if scenario == "idle":
        return [
            _Phase(
                frames=TARGET_FPS * 5,
                fsm_state="IDLE",
                entry_event=None,
                static_gesture="NONE",
                confidence_range=(0.0, 0.0),
                hand_detected=False,
                cursor_target=None,
            ),
        ]

    if scenario == "jitter":
        return [
            _Phase(
                frames=TARGET_FPS * 8,
                fsm_state="HOVER",
                entry_event="ENTER_HOVER",
                static_gesture="OPEN_PALM",
                confidence_range=(0.90, 0.99),
                hand_detected=True,
                cursor_target=(0.5, 0.5),
                jitter_scale=0.015,
            ),
        ]

    if scenario == "pinch":
        return [
            _Phase(
                frames=_HOVER_FRAMES,
                fsm_state="HOVER",
                entry_event="ENTER_HOVER",
                static_gesture="OPEN_PALM",
                confidence_range=(0.90, 0.99),
                hand_detected=True,
                cursor_target=(0.5, 0.45),
            ),
            _Phase(
                frames=_PINCH_HOLD_FRAMES,
                fsm_state="PINCH_START",
                entry_event="PINCH_DETECTED",
                static_gesture="PINCH",
                confidence_range=(0.85, 0.98),
                hand_detected=True,
                cursor_target=(0.5, 0.45),
                jitter_scale=0.002,
            ),
            _Phase(
                frames=TARGET_FPS,
                fsm_state="PINCH_HELD",
                entry_event="PINCH_HELD",
                static_gesture="PINCH",
                confidence_range=(0.85, 0.98),
                hand_detected=True,
                cursor_target=(0.5, 0.45),
                jitter_scale=0.002,
            ),
            _Phase(
                frames=_SHORT_HOVER_FRAMES,
                fsm_state="HOVER",
                entry_event="PINCH_RELEASED",
                static_gesture="OPEN_PALM",
                confidence_range=(0.90, 0.99),
                hand_detected=True,
                cursor_target=(0.5, 0.45),
            ),
            _Phase(
                frames=_PINCH_HOLD_FRAMES // 2,
                fsm_state="PINCH_START",
                entry_event="PINCH_DETECTED",
                static_gesture="PINCH",
                confidence_range=(0.85, 0.98),
                hand_detected=True,
                cursor_target=(0.5, 0.45),
                jitter_scale=0.002,
            ),
            _Phase(
                frames=max(1, _DOUBLE_PINCH_FRAMES - _PINCH_HOLD_FRAMES // 2),
                fsm_state="HOVER",
                entry_event="DOUBLE_PINCH",
                static_gesture="OPEN_PALM",
                confidence_range=(0.85, 0.98),
                hand_detected=True,
                cursor_target=(0.5, 0.45),
            ),
        ]

    if scenario == "swipe":
        directions = [
            ("SWIPE_LEFT", (0.2, 0.5)),
            ("SWIPE_RIGHT", (0.8, 0.5)),
            ("SWIPE_UP", (0.5, 0.2)),
            ("SWIPE_DOWN", (0.5, 0.8)),
        ]
        phases: list[_Phase] = [
            _Phase(
                frames=_SHORT_HOVER_FRAMES,
                fsm_state="HOVER",
                entry_event="ENTER_HOVER",
                static_gesture="OPEN_PALM",
                confidence_range=(0.90, 0.99),
                hand_detected=True,
                cursor_target=(0.5, 0.5),
            ),
        ]
        for swipe_event, target in directions:
            phases.append(
                _Phase(
                    frames=_SWIPE_PENDING_FRAMES,
                    fsm_state="SWIPE_PENDING",
                    entry_event="POINT_DETECTED",
                    static_gesture="POINT",
                    confidence_range=(0.85, 0.97),
                    hand_detected=True,
                    cursor_target=target,
                    jitter_scale=0.003,
                )
            )
            phases.append(
                _Phase(
                    frames=1,
                    fsm_state="SWIPE_COMMIT",
                    entry_event=swipe_event,
                    static_gesture="POINT",
                    confidence_range=(0.85, 0.97),
                    hand_detected=True,
                    cursor_target=target,
                )
            )
            phases.append(
                _Phase(
                    frames=_SHORT_HOVER_FRAMES,
                    fsm_state="HOVER",
                    entry_event="ENTER_HOVER",
                    static_gesture="OPEN_PALM",
                    confidence_range=(0.90, 0.99),
                    hand_detected=True,
                    cursor_target=(0.5, 0.5),
                )
            )
        return phases

    if scenario == "stress":
        gestures = [
            ("HOVER", "OPEN_PALM", None),
            ("PINCH_START", "PINCH", "PINCH_DETECTED"),
            ("PINCH_HELD", "PINCH", "PINCH_HELD"),
            ("SCROLL_MODE", "FIST", "FIST_ENTER"),
            ("SWIPE_PENDING", "POINT", "POINT_DETECTED"),
            ("SWIPE_COMMIT", "POINT", "SWIPE_RIGHT"),
        ]
        phases = []
        for state, gesture, event in gestures:
            phases.append(
                _Phase(
                    frames=random.randint(2, 5),
                    fsm_state=state,
                    entry_event=event,
                    static_gesture=gesture,
                    confidence_range=(0.30, 0.99),
                    hand_detected=True,
                    cursor_target=(
                        random.uniform(0.1, 0.9),
                        random.uniform(0.1, 0.9),
                    ),
                    jitter_scale=0.03,
                )
            )
        phases.append(
            _Phase(
                frames=HAND_LOST_FRAMES + 2,
                fsm_state="IDLE",
                entry_event="HAND_LOST",
                static_gesture="NONE",
                confidence_range=(0.0, 0.0),
                hand_detected=False,
                cursor_target=None,
            )
        )
        return phases

    if scenario == "demo":
        return [
            _Phase(
                frames=_HAND_LOST_DEMO_FRAMES,
                fsm_state="IDLE",
                entry_event=None,
                static_gesture="NONE",
                confidence_range=(0.0, 0.0),
                hand_detected=False,
                cursor_target=None,
            ),
            _Phase(
                frames=_HOVER_FRAMES,
                fsm_state="HOVER",
                entry_event="ENTER_HOVER",
                static_gesture="OPEN_PALM",
                confidence_range=(0.88, 0.99),
                hand_detected=True,
                cursor_target=(0.3, 0.4),
            ),
            _Phase(
                frames=_PINCH_HOLD_FRAMES,
                fsm_state="PINCH_START",
                entry_event="PINCH_DETECTED",
                static_gesture="PINCH",
                confidence_range=(0.85, 0.98),
                hand_detected=True,
                cursor_target=(0.3, 0.4),
                jitter_scale=0.002,
            ),
            _Phase(
                frames=TARGET_FPS,
                fsm_state="PINCH_HELD",
                entry_event="PINCH_HELD",
                static_gesture="PINCH",
                confidence_range=(0.85, 0.98),
                hand_detected=True,
                cursor_target=(0.3, 0.4),
                jitter_scale=0.002,
            ),
            _Phase(
                frames=_HOVER_FRAMES,
                fsm_state="HOVER",
                entry_event="PINCH_RELEASED",
                static_gesture="OPEN_PALM",
                confidence_range=(0.88, 0.99),
                hand_detected=True,
                cursor_target=(0.7, 0.6),
            ),
            _Phase(
                frames=_SCROLL_FRAMES,
                fsm_state="SCROLL_MODE",
                entry_event="FIST_ENTER",
                static_gesture="FIST",
                confidence_range=(0.85, 0.97),
                hand_detected=True,
                cursor_target=(0.7, 0.6),
                jitter_scale=0.002,
            ),
            _Phase(
                frames=_SHORT_HOVER_FRAMES,
                fsm_state="HOVER",
                entry_event="FIST_EXIT",
                static_gesture="OPEN_PALM",
                confidence_range=(0.88, 0.99),
                hand_detected=True,
                cursor_target=(0.4, 0.7),
            ),
            _Phase(
                frames=_SWIPE_PENDING_FRAMES,
                fsm_state="SWIPE_PENDING",
                entry_event="POINT_DETECTED",
                static_gesture="POINT",
                confidence_range=(0.85, 0.97),
                hand_detected=True,
                cursor_target=(0.85, 0.7),
                jitter_scale=0.003,
            ),
            _Phase(
                frames=1,
                fsm_state="SWIPE_COMMIT",
                entry_event="SWIPE_RIGHT",
                static_gesture="POINT",
                confidence_range=(0.85, 0.97),
                hand_detected=True,
                cursor_target=(0.85, 0.7),
            ),
            _Phase(
                frames=_HOVER_FRAMES,
                fsm_state="HOVER",
                entry_event="ENTER_HOVER",
                static_gesture="OPEN_PALM",
                confidence_range=(0.88, 0.99),
                hand_detected=True,
                cursor_target=(0.5, 0.5),
            ),
        ]

    raise ValueError(f"Unknown scenario: {scenario!r}; expected one of {SCENARIOS}")


class _CursorSimulator:
    """Generates smooth, screen-pixel-space raw and filtered cursor motion.

    The simulator eases a raw position toward a normalized target each
    frame, adds Gaussian jitter to the raw position, and maintains a
    separately smoothed (exponentially filtered) position. This stands
    in for Dev A's real landmark smoothing pipeline purely for
    generating visually realistic mock data.
    """

    def __init__(self, smoothing_alpha: float = 0.25, ease_factor: float = 0.12) -> None:
        """Initializes the simulator at the center of the virtual screen.

        Args:
            smoothing_alpha: Exponential smoothing factor applied to
                the raw position to produce the filtered position.
                Higher values track the raw signal more closely.
            ease_factor: Fraction of the remaining distance to the
                target covered per frame.
        """
        self._smoothing_alpha = smoothing_alpha
        self._ease_factor = ease_factor
        self._raw_x = _VIRTUAL_SCREEN_WIDTH / 2.0
        self._raw_y = _VIRTUAL_SCREEN_HEIGHT / 2.0
        self._filtered_x = self._raw_x
        self._filtered_y = self._raw_y

    def step(
        self,
        target: Optional[tuple[float, float]],
        jitter_scale: float,
        hand_detected: bool,
    ) -> tuple[Optional[tuple[int, int]], tuple[float, float, float, float]]:
        """Advances the simulation by one frame.

        Args:
            target: Normalized ``(x, y)`` target in ``[0.0, 1.0]``, or
                ``None`` to hold the current position steady.
            jitter_scale: Standard deviation, in normalized units, of
                the per-frame noise added to the raw position.
            hand_detected: Whether a hand is present this frame. When
                ``False``, the simulator's internal state is frozen
                and ``cursor_screen_pos`` is reported as ``None``.

        Returns:
            A ``(cursor_screen_pos, graph_values)`` tuple, where
            ``cursor_screen_pos`` is an integer pixel ``(x, y)`` tuple
            or ``None``, and ``graph_values`` is a
            ``(raw_x, raw_y, filtered_x, filtered_y)`` float tuple in
            pixel space.
        """
        if not hand_detected:
            return None, (
                self._raw_x,
                self._raw_y,
                self._filtered_x,
                self._filtered_y,
            )

        if target is not None:
            target_x = target[0] * _VIRTUAL_SCREEN_WIDTH
            target_y = target[1] * _VIRTUAL_SCREEN_HEIGHT
            self._raw_x += (target_x - self._raw_x) * self._ease_factor
            self._raw_y += (target_y - self._raw_y) * self._ease_factor

        noise_x = random.gauss(0.0, jitter_scale * _VIRTUAL_SCREEN_WIDTH)
        noise_y = random.gauss(0.0, jitter_scale * _VIRTUAL_SCREEN_HEIGHT)
        self._raw_x = min(max(self._raw_x + noise_x, 0.0), _VIRTUAL_SCREEN_WIDTH - 1)
        self._raw_y = min(max(self._raw_y + noise_y, 0.0), _VIRTUAL_SCREEN_HEIGHT - 1)

        self._filtered_x += (self._raw_x - self._filtered_x) * self._smoothing_alpha
        self._filtered_y += (self._raw_y - self._filtered_y) * self._smoothing_alpha

        cursor_screen_pos = (int(round(self._filtered_x)), int(round(self._filtered_y)))
        return cursor_screen_pos, (
            self._raw_x,
            self._raw_y,
            self._filtered_x,
            self._filtered_y,
        )


def _make_fake_landmarks(
    center_x_norm: float, center_y_norm: float
) -> list[tuple[float, float, float]]:
    """Builds a schema-valid but purely synthetic 21-point landmark list.

    Args:
        center_x_norm: Normalized hand center x-coordinate, in
            ``[0.0, 1.0]``.
        center_y_norm: Normalized hand center y-coordinate, in
            ``[0.0, 1.0]``.

    Returns:
        A list of 21 ``(x, y, z)`` tuples, each clamped to
        ``[0.0, 1.0]`` on the x and y axes, matching the shape
        MediaPipe Hands would produce.
    """
    landmarks: list[tuple[float, float, float]] = []
    for index in range(21):
        angle = (index / 21.0) * 2.0 * np.pi
        radius = 0.05 + 0.01 * (index % 3)
        x = min(max(center_x_norm + radius * np.cos(angle), 0.0), 1.0)
        y = min(max(center_y_norm + radius * np.sin(angle), 0.0), 1.0)
        z = 0.01 * (index % 5)
        landmarks.append((float(x), float(y), float(z)))
    return landmarks


def _make_preview_frame(
    frame_id: int,
    scenario: str,
    fsm_state: str,
    static_gesture: str,
) -> np.ndarray:
    """Creates a synthetic, visibly-updating preview frame.

    No OpenCV processing and no MediaPipe are used. A small solid
    background (its brightness pulsing with ``frame_id`` so the video
    panel is visibly live) is rendered with Pillow, with the current
    frame number, scenario, FSM state, and gesture drawn as plain text
    overlays, purely so a person watching the UI can see the frame is
    actively updating.

    Args:
        frame_id: The current frame counter; also used to vary the
            frame's background brightness over time.
        scenario: The active scenario name.
        fsm_state: The FSM state to display.
        static_gesture: The classified gesture label to display.

    Returns:
        A BGR frame as a ``(FRAME_HEIGHT, FRAME_WIDTH, 3)`` array of
        dtype ``uint8``.
    """
    brightness = 40 + (frame_id % 30)
    image = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), color=(brightness,) * 3)
    draw = ImageDraw.Draw(image)

    lines = [
        f"Frame: {frame_id}",
        f"Scenario: {scenario}",
        f"State: {fsm_state}",
        f"Gesture: {static_gesture}",
    ]
    line_height = 18
    for line_index, line in enumerate(lines):
        draw.text((10, 10 + line_index * line_height), line, fill=(0, 255, 0))

    rgb_array = np.array(image, dtype=np.uint8)
    bgr_array = np.ascontiguousarray(rgb_array[:, :, ::-1])
    return bgr_array


def _compute_rolling_fps(frame_timestamps: "deque[float]") -> float:
    """Computes a rolling frames-per-second value from recent timestamps.

    Args:
        frame_timestamps: A bounded deque of ``time.monotonic()``
            timestamps, one per recently produced frame.

    Returns:
        The measured frames per second over the span of
        ``frame_timestamps``, or ``0.0`` if fewer than two timestamps
        are available.
    """
    if len(frame_timestamps) < 2:
        return 0.0
    elapsed = frame_timestamps[-1] - frame_timestamps[0]
    if elapsed <= 0.0:
        return 0.0
    return (len(frame_timestamps) - 1) / elapsed


def mock_producer(
    result_queue: "queue.Queue[dict[str, Any]]",
    stop_event: threading.Event,
    scenario: str = "demo",
    target_fps: int = TARGET_FPS,
) -> None:
    """Generates and enqueues mock ``RESULT_SCHEMA`` frames until stopped.

    Runs entirely on the calling thread; callers should invoke this
    from a dedicated background thread. Frames are produced at
    approximately ``target_fps`` frames per second. If the queue is
    full (mirroring the real ``InferenceThread``'s
    ``RESULT_QUEUE_MAXSIZE``-bounded behavior), the newly generated
    frame is dropped rather than blocking.

    Args:
        result_queue: The queue to push generated result dictionaries
            onto.
        stop_event: A ``threading.Event`` checked every frame; setting
            it stops production promptly.
        scenario: One of the strings in :data:`SCENARIOS`, selecting
            which scripted FSM timeline to simulate.
        target_fps: Target production rate in frames per second.
            Defaults to :data:`shared.constants.TARGET_FPS`.

    Raises:
        ValueError: If ``scenario`` is not a member of
            :data:`SCENARIOS`.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario!r}; expected one of {SCENARIOS}")

    phases = _build_phase_table(scenario)
    cursor_sim = _CursorSimulator()
    frame_timestamps: "deque[float]" = deque(maxlen=30)
    frame_interval = 1.0 / target_fps if target_fps > 0 else 0.0

    frame_id = 0
    phase_index = 0
    frames_into_phase = 0

    logger.info("mock_producer starting (scenario=%s, target_fps=%s)", scenario, target_fps)

    while not stop_event.is_set():
        loop_start = time.monotonic()

        phase = phases[phase_index]
        is_phase_entry = frames_into_phase == 0
        fsm_event = phase.entry_event if is_phase_entry else None

        confidence = (
            random.uniform(*phase.confidence_range) if phase.hand_detected else 0.0
        )

        cursor_screen_pos, graph_values = cursor_sim.step(
            phase.cursor_target, phase.jitter_scale, phase.hand_detected
        )
        raw_x, raw_y, filtered_x, filtered_y = graph_values

        if phase.hand_detected and phase.cursor_target is not None:
            landmarks = _make_fake_landmarks(*phase.cursor_target)
        elif phase.hand_detected:
            landmarks = _make_fake_landmarks(0.5, 0.5)
        else:
            landmarks = None

        frame_timestamps.append(time.monotonic())
        measured_fps = _compute_rolling_fps(frame_timestamps)

        result: dict[str, Any] = {
            "timestamp": time.time(),
            "frame_id": frame_id,
            "annotated_frame": _make_preview_frame(
                frame_id, scenario, phase.fsm_state, phase.static_gesture
            ),
            "landmarks_raw": landmarks,
            "landmarks_filtered": landmarks,
            "cursor_screen_pos": cursor_screen_pos,
            "static_gesture": phase.static_gesture if phase.hand_detected else "NONE",
            "gesture_confidence": confidence,
            "fsm_state": phase.fsm_state,
            "fsm_event": fsm_event,
            "graph_data": {
                "raw_x": raw_x,
                "raw_y": raw_y,
                "filtered_x": filtered_x,
                "filtered_y": filtered_y,
            },
            "inference_fps": measured_fps,
            "hand_detected": phase.hand_detected,
        }

        try:
            result_queue.put_nowait(result)
        except queue.Full:
            logger.debug("result_queue full; dropping frame %d", frame_id)

        frame_id += 1
        frames_into_phase += 1
        if frames_into_phase >= phase.frames:
            frames_into_phase = 0
            phase_index = (phase_index + 1) % len(phases)

        elapsed = time.monotonic() - loop_start
        sleep_time = max(0.0, frame_interval - elapsed)
        stop_event.wait(sleep_time)

    logger.info("mock_producer exiting (scenario=%s)", scenario)


class _MockMouseInjector:
    """Logs mouse actions without ever touching the real OS cursor.

    Implements the same public interface as
    :class:`~application.os_integration.mouse_controller.MouseInjector`
    so it can be handed to ``ActionThread`` unmodified, but every
    method only logs the requested action -- no ``pyautogui`` or
    ``pynput`` calls are made, so no real cursor movement, clicks, or
    button holds ever occur.
    """

    def __init__(self) -> None:
        """Initializes the mock mouse injector with no button held."""
        self._is_pressed = False
        logger.info("_MockMouseInjector initialized (no real OS input will be sent)")

    def move(self, x: float, y: float) -> None:
        """Logs a requested cursor move instead of performing one.

        Args:
            x: Requested screen x-coordinate in pixels.
            y: Requested screen y-coordinate in pixels.
        """
        logger.debug("_MockMouseInjector.move(x=%s, y=%s)", x, y)

    def left_click(self) -> None:
        """Logs a requested left click instead of performing one."""
        logger.info("_MockMouseInjector.left_click()")

    def right_click(self) -> None:
        """Logs a requested right click instead of performing one."""
        logger.info("_MockMouseInjector.right_click()")

    def double_click(self) -> None:
        """Logs a requested double click instead of performing one."""
        logger.info("_MockMouseInjector.double_click()")

    def press(self) -> None:
        """Logs a requested button hold instead of performing one."""
        self._is_pressed = True
        logger.info("_MockMouseInjector.press()")

    def release(self) -> None:
        """Logs a requested button release instead of performing one."""
        self._is_pressed = False
        logger.info("_MockMouseInjector.release()")

    def scroll(self, amount: int) -> None:
        """Logs a requested scroll instead of performing one.

        Args:
            amount: Requested scroll amount.
        """
        logger.debug("_MockMouseInjector.scroll(amount=%s)", amount)

    def is_pressed(self) -> bool:
        """Returns whether a mock button-hold is currently active.

        Returns:
            True if :meth:`press` has been called without a matching
            :meth:`release`, False otherwise.
        """
        return self._is_pressed


class _MockKeyboardInjector:
    """Logs keyboard actions without ever sending real key events.

    Implements the same public interface as
    :class:`~application.os_integration.keyboard_controller.KeyboardInjector`
    so it can be handed to ``ActionThread`` unmodified, but every
    method only logs the requested action -- no ``pynput`` calls are
    made, so no real key presses ever occur.
    """

    def __init__(self) -> None:
        """Initializes the mock keyboard injector."""
        logger.info(
            "_MockKeyboardInjector initialized (no real OS input will be sent)"
        )

    def perform_action(self, action: str) -> None:
        """Logs a requested action instead of dispatching it.

        Args:
            action: The action identifier that would have been
                dispatched.
        """
        logger.info("_MockKeyboardInjector.perform_action(action=%s)", action)

    def press_key(self, key: Any) -> None:
        """Logs a requested key press instead of performing one.

        Args:
            key: The key that would have been pressed.
        """
        logger.debug("_MockKeyboardInjector.press_key(key=%s)", key)

    def release_key(self, key: Any) -> None:
        """Logs a requested key release instead of performing one.

        Args:
            key: The key that would have been released.
        """
        logger.debug("_MockKeyboardInjector.release_key(key=%s)", key)

    def tap_key(self, key: Any) -> None:
        """Logs a requested key tap instead of performing one.

        Args:
            key: The key that would have been tapped.
        """
        logger.debug("_MockKeyboardInjector.tap_key(key=%s)", key)


class _MockEventLogger:
    """Logs dispatched events without creating or writing to any database.

    Implements the same public interface as
    :class:`~application.db.event_logger.EventLogger` so it can be
    handed to ``ActionThread`` unmodified, but every method only logs
    via the ``logging`` module -- ``airsign_session.db`` is never
    created or modified while this mock is in use.
    """

    def __init__(self) -> None:
        """Initializes the mock event logger."""
        logger.info(
            "_MockEventLogger initialized (no SQLite database will be created)"
        )

    def log_event(
        self,
        timestamp: float,
        gesture: str,
        fsm_state: str,
        fsm_event: Optional[str],
        action: Optional[str],
        confidence: float,
        latency_ms: float,
    ) -> None:
        """Logs a dispatched event's details instead of writing a database row.

        Args:
            timestamp: Event time as a Unix epoch float.
            gesture: The classified gesture label active at this event.
            fsm_state: The FSM state active at this event.
            fsm_event: The FSM event that fired, or ``None``.
            action: The OS action dispatched, or ``None``.
            confidence: Classifier confidence in the range 0.0-1.0.
            latency_ms: End-to-end pipeline latency in milliseconds.
        """
        logger.info(
            "_MockEventLogger.log_event(timestamp=%s, gesture=%s, "
            "fsm_state=%s, fsm_event=%s, action=%s, confidence=%.2f, "
            "latency_ms=%.2f)",
            timestamp,
            gesture,
            fsm_state,
            fsm_event,
            action,
            confidence,
            latency_ms,
        )

    def get_all_events(self) -> list[dict[str, Any]]:
        """Returns an empty list, since no events are ever persisted.

        Returns:
            An empty list.
        """
        logger.debug("_MockEventLogger.get_all_events() -> []")
        return []

    def get_event_count(self) -> int:
        """Returns zero, since no events are ever persisted.

        Returns:
            ``0``.
        """
        return 0

    def get_session_summary(self) -> dict[str, Any]:
        """Returns an empty summary, since no events are ever persisted.

        Returns:
            A summary dictionary with zeroed totals and an empty
            ``gesture_counts`` mapping.
        """
        return {
            "total_events": 0,
            "average_confidence": 0.0,
            "average_latency_ms": 0.0,
            "gesture_counts": {},
        }

    def clear_session(self) -> None:
        """Logs a clear-session request; there is no data to clear."""
        logger.debug("_MockEventLogger.clear_session()")

    def close(self) -> None:
        """Logs that the mock event logger has been closed."""
        logger.info("_MockEventLogger.close()")


def main() -> None:
    """Launches the full Developer B application against mock perception data.

    Constructs a real ``SettingsManager``, ``MainWindow``,
    ``CoordinateMapper``, and ``ActionThread``, but wires that
    ``ActionThread`` to :class:`_MockMouseInjector`,
    :class:`_MockKeyboardInjector`, and :class:`_MockEventLogger`
    instead of the production classes, so no real OS input is ever
    sent and no session database is ever created. Wires
    ``MainWindow``'s ``AppSignals`` into the ``ActionThread``, starts
    a background ``mock_producer`` thread, shows the window, and runs
    the Qt event loop. Shuts down every thread cleanly when the window
    is closed.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    scenario = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if scenario not in SCENARIOS:
        logger.error(
            "Unknown scenario '%s'; expected one of %s", scenario, sorted(SCENARIOS)
        )
        sys.exit(1)

    result_queue: "queue.Queue[dict[str, Any]]" = queue.Queue(
        maxsize=RESULT_QUEUE_MAXSIZE
    )
    stop_event = threading.Event()

    producer_thread = threading.Thread(
        target=mock_producer,
        args=(result_queue, stop_event, scenario),
        name="MockPerceptionProducer",
        daemon=True,
    )
    producer_thread.start()

    config_path = Path(__file__).resolve().parent.parent / "config.json"
    settings_manager = SettingsManager(config_path=config_path)
    load_bindings(settings_manager)

    app = QApplication(sys.argv)
    window = MainWindow(settings_manager)

    coordinate_mapper = CoordinateMapper(settings_manager)
    mouse_injector = _MockMouseInjector()
    keyboard_injector = _MockKeyboardInjector()
    event_logger = _MockEventLogger()

    action_thread = ActionThread(
        result_queue=result_queue,
        stop_event=stop_event,
        settings_manager=settings_manager,
        coordinate_mapper=coordinate_mapper,
        mouse_injector=mouse_injector,
        keyboard_injector=keyboard_injector,
        event_logger=event_logger,
        app_signals=window.app_signals,
    )
    action_thread.start()

    def _shutdown() -> None:
        """Signals every background thread to stop and waits briefly."""
        logger.info("Shutting down mock perception harness")
        stop_event.set()
        action_thread.stop()
        producer_thread.join(timeout=2.0)
        event_logger.close()

    app.aboutToQuit.connect(_shutdown)

    window.show()
    exit_code = app.exec()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
