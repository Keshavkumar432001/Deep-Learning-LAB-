import cv2 as cv
import sys

from module1_capture     import WebcamCapture
from module2_interceptor import GestureInterceptor, THUMB_TIP, INDEX_TIP, MIDDLE_TIP
from module3_recognizer  import (GestureRecognizer, GESTURE_LABELS,
                                  GESTURE_VOLUME_CTRL, GESTURE_BRIGHTNESS_CTRL,
                                  GESTURE_MOUSE_MOVE, GESTURE_MOUSE_CLICK,
                                  GESTURE_SCROLL_UP, GESTURE_SCROLL_DOWN)
from module4_commander   import GestureCommander


def draw_gesture_label(frame, gesture, conf):
    label = GESTURE_LABELS.get(gesture, "")
    if label:
        text = f"{label}  ({int(conf*100)}%)"
        cv.putText(frame, text, (10, 460),
                   cv.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 0), 2)


def main():
    # --- Init all four modules ---
    capture     = WebcamCapture(cam_index=0, width=640, height=480)
    interceptor = GestureInterceptor(max_hands=1, detection_conf=0.7)
    recognizer  = GestureRecognizer()
    commander   = GestureCommander()

    print("Gesture Control running — press 'q' to quit")
    print("  Two fingers up + pinch  →  Volume")
    print("  One finger up  + pinch  →  Brightness")
    print("  Index+Middle up (no thumb) →  Mouse (pinch to click)")
    print("  Index+Middle+Ring →  Scroll Up")
    print("  All four fingers up    →  Scroll Down")

    mouse_held = False

    while True:
        # ----------------------------------------------------------------
        # MODULE 1 — Capture & preprocess frame
        # ----------------------------------------------------------------
        frame = capture.read_frame()
        if frame is None:
            print("Camera read failed.")
            break

        frame = capture.preprocess(frame)

        # ----------------------------------------------------------------
        # MODULE 2 — Intercept hand landmarks
        # ----------------------------------------------------------------
        frame = interceptor.find_hands(frame, draw=True)
        lmList = interceptor.get_landmarks(frame)

        if lmList:
            # Compute the distances we actually need
            dist_ti, p_thumb, p_index  = interceptor.get_distance(lmList, THUMB_TIP, INDEX_TIP)
            dist_im, p_index2, p_mid   = interceptor.get_distance(lmList, INDEX_TIP, MIDDLE_TIP)
            finger_states              = interceptor.get_finger_states(lmList)

            # ----------------------------------------------------------------
            # MODULE 3 — Recognize the gesture
            # ----------------------------------------------------------------
            gesture, confidence = recognizer.recognize(
                lmList, finger_states, dist_ti, dist_im
            )

            draw_gesture_label(frame, gesture, confidence)

            # ----------------------------------------------------------------
            # MODULE 4 — Execute the corresponding command
            # ----------------------------------------------------------------
            if gesture == GESTURE_VOLUME_CTRL and p_thumb and p_index:
                vol = commander.set_volume(dist_ti)
                midpoint = ((p_thumb[0]+p_index[0])//2, (p_thumb[1]+p_index[1])//2)
                interceptor.draw_connection(frame, p_thumb, p_index, midpoint,
                                            active=(dist_ti < 35))
                frame = commander.draw_volume_bar(frame, vol)

            elif gesture == GESTURE_BRIGHTNESS_CTRL and p_thumb and p_index:
                bright = commander.set_brightness(dist_ti)
                midpoint = ((p_thumb[0]+p_index[0])//2, (p_thumb[1]+p_index[1])//2)
                interceptor.draw_connection(frame, p_thumb, p_index, midpoint,
                                            active=(dist_ti < 35))
                frame = commander.draw_brightness_bar(frame, bright)

            elif gesture in (GESTURE_MOUSE_MOVE, GESTURE_MOUSE_CLICK):
                ix, iy = lmList[INDEX_TIP][1], lmList[INDEX_TIP][2]
                mx, my = commander.move_mouse(ix, iy)
                frame  = commander.draw_mouse_coords(frame, mx, my)

                if gesture == GESTURE_MOUSE_CLICK:
                    if not mouse_held:
                        commander.mouse_down()
                        mouse_held = True
                    cv.putText(frame, "CLICK", (10, 110),
                               cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                else:
                    if mouse_held:
                        commander.mouse_up()
                        mouse_held = False

            elif gesture == GESTURE_SCROLL_UP:
                commander.scroll("up")
                cv.putText(frame, "^^ Scrolling Up", (10, 110),
                           cv.FONT_HERSHEY_PLAIN, 2, (0, 220, 255), 2)

            elif gesture == GESTURE_SCROLL_DOWN:
                commander.scroll("down")
                cv.putText(frame, "vv Scrolling Down", (10, 110),
                           cv.FONT_HERSHEY_PLAIN, 2, (0, 180, 255), 2)

        else:
            # No hand in frame — make sure mouse button isn't stuck down
            if mouse_held:
                commander.mouse_up()
                mouse_held = False

        frame = capture.overlay_fps(frame)
        cv.imshow("Gesture Control", frame)

        key = cv.waitKey(5) & 0xFF
        if key == ord('q') or key == 27:
            break

    capture.release()
    sys.exit(0)


if __name__ == "__main__":
    main()
