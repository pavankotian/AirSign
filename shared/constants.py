# =============================================================================
# shared/constants.py
# AirSign — Shared Numeric Constants & Default Configuration Values
# -----------------------------------------------------------------------------
# All numeric thresholds, timing values, and hardware defaults used across
# both the Dev A (perception) and Dev B (application/UI) tracks live here.
#
# Any value that appears in more than one file MUST be defined here and
# imported — never duplicated as a local literal.
#
# FROZEN: Changing a value here changes behavior across the entire system.
# Document the reason for any tuning change in a git commit message.
#
# Organisation:
#   1. Gesture timing & FSM thresholds      (Dev A primary)
#   2. Action dispatch cooldowns            (Dev B primary)
#   3. 1€ Filter defaults                   (Dev A primary)
#   4. Tracking & coordinate mapping        (Dev A primary, Dev B reads)
#   5. Camera & capture hardware            (Dev A primary)
#   6. Queue sizing                         (both tracks)
#   7. Confidence gating                    (both tracks)
#   8. UI / graph display                   (Dev B primary)
# =============================================================================

# =============================================================================
# 1. Gesture Timing & FSM Thresholds
#    Used by: perception/gesture/fsm.py
# =============================================================================

# Milliseconds a PINCH gesture must be held continuously before the FSM
# transitions from PINCH_START → PINCH_HELD (drag mode activation).
# Below this threshold, a pinch-release is treated as a short click.
PINCH_HOLD_MS: int = 400

# Milliseconds within which a second pinch-release cycle must occur after
# the first PINCH_RELEASED event to trigger a DOUBLE_PINCH (right-click).
# Measured from the moment of the first PINCH_RELEASED timestamp.
DOUBLE_PINCH_MS: int = 500

# Normalized displacement (0.0–1.0 in landmark coordinate space) that the
# index fingertip must travel from its SWIPE_PENDING entry position to
# trigger a SWIPE_* commit event. Axis with greater displacement wins.
# Tuning: increase to require more deliberate swipes; decrease for sensitivity.
SWIPE_THRESHOLD: float = 0.15

# Minimum number of consecutive frames POINT gesture must be held before
# the FSM enters SWIPE_PENDING. Prevents accidental swipe entry on brief
# frame-level POINT detections during other gestures.
SWIPE_MIN_FRAMES: int = 8

# Minimum milliseconds that must elapse between two firings of the same
# FSM event. Prevents rapid-fire event emission during gesture hold phases.
# Applied per-event, not globally — different events have independent timers.
GESTURE_DEBOUNCE_MS: int = 150

# Number of consecutive frames with gesture == "NONE" required before the
# FSM transitions from any active state to IDLE and fires HAND_LOST.
# At 30fps: 10 frames = ~333ms of no hand before mode resets.
HAND_LOST_FRAMES: int = 10

# Number of consecutive frames a non-dominant gesture must persist before
# the FSM treats it as a state-exit trigger (e.g., FIST_EXIT from SCROLL_MODE,
# or SWIPE_PENDING abandonment). Prevents jitter-induced false exits.
STATE_EXIT_FRAMES: int = 4

# =============================================================================
# 2. Action Dispatch Cooldowns
#    Used by: application/os_integration/mouse_controller.py
#             application/os_integration/keyboard_controller.py
# =============================================================================

# Minimum milliseconds between consecutive left or right click injections.
# Prevents OS-level click spam if PINCH_RELEASED fires on back-to-back frames.
COOLDOWN_CLICK_MS: int = 300

# Minimum milliseconds between consecutive scroll injection calls.
# Lower value = smoother scroll; too low risks OS scroll queue flooding.
COOLDOWN_SCROLL_MS: int = 80

# Minimum milliseconds between consecutive keyboard macro injections.
# Applies to: prev_slide, next_slide, volume_up/down, media_play_pause.
COOLDOWN_KEYPRESS_MS: int = 200

# =============================================================================
# 3. One-Euro Filter Defaults
#    Used by: perception/filtering/one_euro.py
#             perception/filtering/smoother.py
# These are the startup defaults. Live values are read from config.json and
# updated via InferenceThread.update_filter_params() when sliders change.
# =============================================================================

# Minimum cutoff frequency (Hz). Controls jitter at rest.
# Higher value → more jitter removed at rest, but slightly more lag.
# Recommended range: 0.5–2.0 for cursor tracking.
DEFAULT_FILTER_MINCUTOFF: float = 1.0

# Speed coefficient. Controls how much the cutoff frequency adapts to motion speed.
# Higher value → less lag on fast movements, more jitter on slow movements.
# Recommended range: 0.001–0.01 for cursor tracking at 30fps.
DEFAULT_FILTER_BETA: float = 0.007

# Derivative filter cutoff frequency (Hz). Controls smoothing of the velocity
# estimate used to compute the adaptive cutoff. Usually fixed at 1.0.
# Do not expose this in the UI — it rarely needs tuning.
DEFAULT_FILTER_DCUTOFF: float = 1.0

# =============================================================================
# 4. Tracking & Coordinate Mapping
#    Used by: perception/filtering/smoother.py (cursor mapping)
#             perception/gesture/feature_extractor.py (pinch threshold)
# =============================================================================

# Multiplier applied to normalized cursor coordinates before screen mapping.
# Values < 1.0 reduce the effective cursor range (requires more hand movement).
# Values > 1.0 expand the effective range (requires less hand movement).
# This is the startup default; live value is read from config.json.
DEFAULT_SENSITIVITY: float = 0.85

# Normalized distance between THUMB_TIP and INDEX_TIP below which the
# PINCH gesture is classified as active. Measured relative to palm_size.
# Smaller value → requires tighter pinch; larger → more forgiving.
# This is the startup default; live value is read from config.json.
DEFAULT_PINCH_TOLERANCE: float = 0.15

# Default active zone within the camera frame used for cursor mapping.
# Expressed as normalized fractions of frame width/height (0.0–1.0).
# Landmarks outside this zone are still tracked but cursor is clamped.
DEFAULT_ACTIVE_ZONE: dict = {
    "x1": 0.05,
    "y1": 0.05,
    "x2": 0.95,
    "y2": 0.95,
}

# Index of the MediaPipe hand landmark used as the cursor control point.
# 8 = INDEX_FINGER_TIP — the most precise and natural pointer.
# Do not change without updating feature_extractor.py and smoother.py.
CURSOR_LANDMARK_INDEX: int = 8

# Small epsilon added to distance denominators to prevent division by zero
# in feature extraction when landmarks collapse (e.g., very close to camera).
DISTANCE_EPSILON: float = 1e-6

# =============================================================================
# 5. Camera & Capture Hardware
#    Used by: perception/capture_thread.py
# =============================================================================

# OpenCV device index for the webcam. 0 = first detected camera.
# Override via config.json camera.device_index for multi-camera setups.
DEFAULT_CAMERA_INDEX: int = 0

# Target capture resolution. MediaPipe Hands is optimized for 640×480 input.
# Higher resolutions increase CPU load without proportional accuracy gain.
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480

# Target frames per second requested from cv2.VideoCapture.
# Actual achieved FPS may be lower depending on USB bandwidth and lighting.
# The 1€ Filter is initialized with this value as its freq parameter.
TARGET_FPS: int = 30

# Number of frames used in the rolling FPS average displayed in the status bar.
FPS_ROLLING_WINDOW: int = 30

# =============================================================================
# 6. Queue Sizing
#    Used by: perception/capture_thread.py    (produces to frame_queue)
#             perception/inference_thread.py  (consumes frame_queue,
#                                             produces to result_queue)
#             application/action_thread.py    (consumes result_queue)
# =============================================================================

# Maximum number of raw frames buffered between CaptureThread and InferenceThread.
# Set to 2: allows one frame in transit + one being processed, no more.
# If full: CaptureThread drops the oldest frame. Never grows the backlog.
CAPTURE_QUEUE_MAXSIZE: int = 2

# Maximum number of result dicts buffered between InferenceThread and ActionThread.
# Set to 1: ActionThread always processes the most recent gesture state.
# If full: InferenceThread discards the new result (NOT the queued one —
# ActionThread may be mid-read on the queued result).
RESULT_QUEUE_MAXSIZE: int = 1

# Timeout in seconds for queue.get() calls in blocking loops.
# Prevents threads from hanging forever if the upstream thread stops.
QUEUE_GET_TIMEOUT: float = 0.1

# =============================================================================
# 7. Confidence Gating
#    Used by: perception/gesture/classifier.py  (UNKNOWN threshold)
#             application/action_thread.py      (action dispatch gate)
# =============================================================================

# Minimum classifier confidence (0.0–1.0) required for Dev A to emit a
# non-UNKNOWN gesture label, AND for Dev B to dispatch an OS action.
# Frames below this threshold are treated as UNKNOWN and produce no events.
# Tuning: lower to improve responsiveness in poor lighting; raise to reduce
# false-positive actions.
MIN_ACTION_CONFIDENCE: float = 0.75

# Minimum number of consecutive frames a gesture must be the dominant
# classification before the FSM treats it as a confirmed gesture.
# Used by GestureFSM._dominant_gesture() window parameter.
GESTURE_CONFIRMATION_FRAMES: int = 6

# =============================================================================
# 8. UI / Graph Display
#    Used by: ui/smoothing_graph.py   (Dev B)
#             ui/gesture_panel.py     (Dev B)
#             application/db/         (Dev B)
# =============================================================================

# Number of frames visible in the rolling cursor smoothing graph.
# At 30fps: 150 frames = 5 seconds of history visible at any time.
GRAPH_WINDOW_FRAMES: int = 150

# Maximum number of gesture event rows retained in the GesturePanel action log.
# Oldest entries are trimmed when this count is exceeded.
ACTION_LOG_MAX_ROWS: int = 100

# Duration in milliseconds for the GesturePanel border flash animation
# triggered on each FSM event. Short enough to be non-distracting.
GESTURE_FLASH_MS: int = 200

# Duration in milliseconds before the PDF export toast auto-dismisses.
EXPORT_TOAST_MS: int = 4000