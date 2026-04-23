import numpy as np
import pyautogui
import platform
import subprocess

# Module 4: Gesture Commander
# Receives the recognized gesture label + the raw landmark data
# and executes the appropriate system command.
# Keeps all the messy OS-level stuff isolated here.

pyautogui.FAILSAFE = False

SCREEN_W, SCREEN_H = pyautogui.size()

# Camera frame dimensions — must match what Module 1 is using
CAM_W, CAM_H = 640, 480


class GestureCommander:
    def __init__(self):
        self._os = platform.system()
        self._last_scroll_y = None

    # ------------------------------------------------------------------ #
    #  Volume
    # ------------------------------------------------------------------ #
    def set_volume(self, dist_px, min_px=15, max_px=200):
        vol = int(np.interp(dist_px, (min_px, max_px), (0, 100)))
        vol = max(0, min(100, vol))

        if self._os == "Linux":
            subprocess.call(
                ["amixer", "-D", "pulse", "sset", "Master", f"{vol}%"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        elif self._os == "Darwin":   # macOS
            subprocess.call(["osascript", "-e", f"set volume output volume {vol}"])
        elif self._os == "Windows":
            # pycaw works best on Windows; using pyautogui hotkeys as fallback
            # For a full implementation swap this with pycaw
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                # pycaw uses -65.25 to 0.0 dB range
                dB = np.interp(vol, [0, 100], [-65.25, 0.0])
                volume.SetMasterVolumeLevel(dB, None)
            except ImportError:
                pass   # pycaw not installed, skip silently

        return vol

    # ------------------------------------------------------------------ #
    #  Brightness
    # ------------------------------------------------------------------ #
    def set_brightness(self, dist_px, min_px=15, max_px=200):
        brightness = int(np.interp(dist_px, (min_px, max_px), (0, 100)))
        brightness = max(0, min(100, brightness))

        try:
            import screen_brightness_control as sbc
            sbc.set_brightness(brightness, display=0)
        except Exception:
            pass   # sbc might not be installed or supported on this display

        return brightness

    # ------------------------------------------------------------------ #
    #  Mouse
    # ------------------------------------------------------------------ #
    def move_mouse(self, finger_x, finger_y):
        # Map from camera coordinates to screen coordinates
        mx = int(np.interp(finger_x, (0, CAM_W), (0, SCREEN_W)))
        my = int(np.interp(finger_y, (0, CAM_H), (0, SCREEN_H)))
        pyautogui.moveTo(mx, my, duration=0.08)
        return mx, my

    def mouse_down(self):
        pyautogui.mouseDown(button='left')

    def mouse_up(self):
        pyautogui.mouseUp(button='left')

    # ------------------------------------------------------------------ #
    #  Scroll
    # ------------------------------------------------------------------ #
    def scroll(self, direction):
        # direction: "up" or "down"
        clicks = 3
        if direction == "up":
            pyautogui.scroll(clicks)
        else:
            pyautogui.scroll(-clicks)

    # ------------------------------------------------------------------ #
    #  Overlay helpers  (draw current value on frame)
    # ------------------------------------------------------------------ #
    def draw_volume_bar(self, frame, vol_pct):
        import cv2 as cv
        # Bar on the right side
        bar_x, bar_y_top, bar_h = 580, 50, 300
        filled_h = int(np.interp(vol_pct, (0, 100), (0, bar_h)))

        # Background
        cv.rectangle(frame, (bar_x, bar_y_top), (bar_x + 30, bar_y_top + bar_h),
                     (50, 50, 50), cv.FILLED)
        # Fill level
        cv.rectangle(frame, (bar_x, bar_y_top + bar_h - filled_h),
                     (bar_x + 30, bar_y_top + bar_h), (0, 200, 255), cv.FILLED)
        cv.putText(frame, f"{vol_pct}%", (bar_x - 5, bar_y_top + bar_h + 25),
                   cv.FONT_HERSHEY_PLAIN, 1.5, (0, 200, 255), 2)
        return frame

    def draw_brightness_bar(self, frame, bright_pct):
        import cv2 as cv
        bar_x, bar_y_top, bar_h = 540, 50, 300
        filled_h = int(np.interp(bright_pct, (0, 100), (0, bar_h)))

        cv.rectangle(frame, (bar_x, bar_y_top), (bar_x + 30, bar_y_top + bar_h),
                     (50, 50, 50), cv.FILLED)
        cv.rectangle(frame, (bar_x, bar_y_top + bar_h - filled_h),
                     (bar_x + 30, bar_y_top + bar_h), (0, 255, 180), cv.FILLED)
        cv.putText(frame, f"{bright_pct}%", (bar_x - 5, bar_y_top + bar_h + 25),
                   cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 180), 2)
        return frame

    def draw_mouse_coords(self, frame, mx, my):
        import cv2 as cv
        cv.putText(frame, f"X:{mx} Y:{my}", (10, 80),
                   cv.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 0), 2)
        return frame
