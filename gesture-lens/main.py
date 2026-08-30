import cv2
import pyvirtualcam
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HandTrackerModule import handDetector

import mediapipe as mp
import time

EXPECTED_WIDTH = 1280
EXPECTED_HEIGHT = 720

cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, EXPECTED_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, EXPECTED_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))

print("Camera:", cap.isOpened())
for _ in range(3):
    cap.read()

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if actual_w != EXPECTED_WIDTH or actual_h != EXPECTED_HEIGHT:
    print(f"Actual resolution {actual_w} {actual_h}")
    print("Provided size is not excepted")

cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("preview", 800, 600)

detector = handDetector()

with pyvirtualcam.Camera(height=720, width=1280, fps=30, device="/dev/video3") as cam:
    print(f"virtual camera running: {cam.device}")
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.flip(frame, 1)


        # cv2.putText(
        #     frame,
        #     "GestureLens",
        #     (5, 5),
        #     cv2.FONT_HERSHEY_COMPLEX,
        #     1,
        #     (0, 255, 0),
        #     2,
        #     cv2.LINE_AA,
        # )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts = int(time.time() * 1000)

        detector.landmarker.detect_async(mp_image, ts)
        img = detector.findHands(frame)

        # cam.send(rgb_frame)
        # cam.sleep_until_next_frame()

        cv2.imshow("preview", frame)

        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
