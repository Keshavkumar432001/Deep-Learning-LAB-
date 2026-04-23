import cv2 as cv
import mediapipe as mp
import math

# Module 2: Gesture Interception
# Detects the hand in the frame, extracts landmark positions,
# and computes raw gesture features (finger distances, angles, etc.)
# that the recognizer can work with.

# MediaPipe hand landmark indices — keeping these here for easy reference
# 0=WRIST, 4=THUMB_TIP, 8=INDEX_TIP, 12=MIDDLE_TIP, 16=RING_TIP, 20=PINKY_TIP
# 5=INDEX_MCP, 9=MIDDLE_MCP, 13=RING_MCP, 17=PINKY_MCP

WRIST       = 0
THUMB_TIP   = 4
INDEX_TIP   = 8
MIDDLE_TIP  = 12
RING_TIP    = 16
PINKY_TIP   = 20
THUMB_MCP   = 2
INDEX_MCP   = 5
MIDDLE_MCP  = 9
RING_MCP    = 13
PINKY_MCP   = 17


class GestureInterceptor:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.6):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, frame, draw=True):
        # MediaPipe needs RGB
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks and draw:
            for handlms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, handlms, self.mpHands.HAND_CONNECTIONS)

        return frame

    def get_landmarks(self, frame, hand_index=0):
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if hand_index >= len(self.results.multi_hand_landmarks):
                return lmList

            hand = self.results.multi_hand_landmarks[hand_index]
            h, w, _ = frame.shape

            for idx, lm in enumerate(hand.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                lmList.append((idx, px, py))

        return lmList

    def get_distance(self, lmList, id1, id2):
        if not lmList or max(id1, id2) >= len(lmList):
            return 0, None, None
        x1, y1 = lmList[id1][1], lmList[id1][2]
        x2, y2 = lmList[id2][1], lmList[id2][2]
        dist = math.hypot(x2 - x1, y2 - y1)
        midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
        return dist, (x1, y1), (x2, y2)

    def get_finger_states(self, lmList):
        # Returns which fingers are up as a list [thumb, index, middle, ring, pinky]
        # 1 = up, 0 = down
        if not lmList or len(lmList) < 21:
            return []

        fingers = []

        # Thumb — compare x coords since it's horizontal
        if lmList[THUMB_TIP][1] < lmList[THUMB_MCP][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other four fingers — tip y should be above (less than) their MCP y
        tip_ids = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        mcp_ids = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

        for tip, mcp in zip(tip_ids, mcp_ids):
            if lmList[tip][2] < lmList[mcp][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def draw_connection(self, frame, p1, p2, midpoint, active=False):
        cv.circle(frame, p1, 8, (255, 12, 136), cv.FILLED)
        cv.circle(frame, p2, 8, (255, 12, 136), cv.FILLED)
        cv.line(frame, p1, p2, (255, 0, 255), 2)
        color = (0, 255, 0) if active else (255, 12, 136)
        cv.circle(frame, midpoint, 8, color, cv.FILLED)
        return frame