import cv2 as cv
import time

# Module 1: Webcam Input & Preprocessing
# Handles reading raw frames from the webcam and converting them
# into a usable format for the gesture pipeline.

class WebcamCapture:
    def __init__(self, cam_index=0, width=640, height=480):
        self.cap = cv.VideoCapture(cam_index)
        self.width = width
        self.height = height

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        self.pTime = 0

        if not self.cap.isOpened():
            raise IOError("Cannot open webcam. Check if it's connected.")

    def read_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None

        # Flip so it feels like a mirror — more natural for gesture control
        frame = cv.flip(frame, 1)
        return frame

    def preprocess(self, frame):
        # Resize just in case the cam doesn't respect the set() values
        frame = cv.resize(frame, (self.width, self.height))

        # Slight gaussian blur to reduce noise before hand detection
        # kernel 3x3 is enough — we don't want to lose edge detail
        blurred = cv.GaussianBlur(frame, (3, 3), 0)

        return blurred

    def overlay_fps(self, frame):
        cTime = time.time()
        if self.pTime != 0:
            fps = 1 / (cTime - self.pTime)
            cv.putText(frame, f"FPS: {int(fps)}", (10, 40),
                       cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        self.pTime = cTime
        return frame

    def release(self):
        self.cap.release()
        cv.destroyAllWindows()
