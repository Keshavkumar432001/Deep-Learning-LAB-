import numpy as np

# Module 3: Gesture Recognition
# Takes the raw landmark data from Module 2 and matches it against
# known gesture patterns. Outputs a gesture label + confidence value.

# Gesture labels
GESTURE_NONE           = "none"
GESTURE_VOLUME_CTRL    = "volume_control"
GESTURE_BRIGHTNESS_CTRL= "brightness_control"
GESTURE_MOUSE_MOVE     = "mouse_move"
GESTURE_MOUSE_CLICK    = "mouse_click"
GESTURE_SCROLL_UP      = "scroll_up"
GESTURE_SCROLL_DOWN    = "scroll_down"


class GestureRecognizer:
    def __init__(self):
        # Thresholds tuned by trial — adjust if your hand size is different
        self.PINCH_THRESHOLD  = 35   # fingers "together" distance in px
        self.SPREAD_THRESHOLD = 120  # fingers clearly "apart"

        # Smoothing buffer — averages last N distances to reduce jitter
        self._dist_buffer = []
        self._buffer_size = 5

    def _smooth_distance(self, dist):
        self._dist_buffer.append(dist)
        if len(self._dist_buffer) > self._buffer_size:
            self._dist_buffer.pop(0)
        return np.mean(self._dist_buffer)

    def recognize(self, lmList, finger_states, dist_thumb_index, dist_index_middle):
        # No hand detected
        if not lmList or not finger_states:
            return GESTURE_NONE, 0.0

        thumb, index, middle, ring, pinky = finger_states

        # --- Gesture: Volume / Brightness Control ---
        # Thumb and index are the only fingers involved (rest can be anything)
        # We detect which mode based on which other fingers are extended
        
        # Pinch between thumb (4) and index (8)
        smooth_ti = self._smooth_distance(dist_thumb_index)

        # Volume: index up, middle up, ring+pinky down
        if index == 1 and middle == 1 and ring == 0 and pinky == 0:
            conf = self._distance_to_confidence(smooth_ti, 15, 200)
            return GESTURE_VOLUME_CTRL, conf

        # Brightness: only index up, rest down (or all down with thumb active)
        if index == 1 and middle == 0 and ring == 0 and pinky == 0:
            conf = self._distance_to_confidence(smooth_ti, 15, 200)
            return GESTURE_BRIGHTNESS_CTRL, conf

        # --- Gesture: Mouse Move / Click ---
        # Index and middle both up = mouse mode
        # When those two fingers pinch together = click
        if index == 1 and middle == 1 and thumb == 0 and ring == 0 and pinky == 0:
            if dist_index_middle < self.PINCH_THRESHOLD:
                return GESTURE_MOUSE_CLICK, 0.95
            return GESTURE_MOUSE_MOVE, 0.90

        # --- Gesture: Scroll ---
        # All 4 fingers up (no thumb) = scroll mode
        # Wrist tilt would be ideal but for simplicity:
        # index+middle+ring up = scroll up, add pinky = scroll down
        if index == 1 and middle == 1 and ring == 1 and pinky == 0:
            return GESTURE_SCROLL_UP, 0.85

        if index == 1 and middle == 1 and ring == 1 and pinky == 1:
            return GESTURE_SCROLL_DOWN, 0.85

        return GESTURE_NONE, 0.0

    def _distance_to_confidence(self, dist, min_d, max_d):
        # Rough confidence based on how far from ambiguous mid-range we are
        # Near min or max = high confidence, middle = lower
        norm = (dist - min_d) / (max_d - min_d)
        norm = max(0.0, min(1.0, norm))
        # U-shaped: confident at extremes
        confidence = abs(norm - 0.5) * 2
        return round(confidence, 2)


# Gesture display names for the UI overlay
GESTURE_LABELS = {
    GESTURE_NONE:            "",
    GESTURE_VOLUME_CTRL:     "Volume Control",
    GESTURE_BRIGHTNESS_CTRL: "Brightness Control",
    GESTURE_MOUSE_MOVE:      "Mouse Move",
    GESTURE_MOUSE_CLICK:     "Mouse Click",
    GESTURE_SCROLL_UP:       "Scroll Up",
    GESTURE_SCROLL_DOWN:     "Scroll Down",
}
