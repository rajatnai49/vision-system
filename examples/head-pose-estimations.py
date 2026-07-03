import time

import cv2

import mediapipe as mp
import numpy as np
from numpy._core import dtype

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
RunningMode = mp.tasks.vision.RunningMode

latest_result = None
model_path = "/home/rajatnai49/projects/vision-system/face_landmarker.task"


def landmarker_result_callback(result, output_image, timestampms):
    global latest_result
    latest_result = result
    # print(result)


def estimate_headpose(frame, result):
    if result is None or not result.face_landmarks:
        print("no result")
        return
    img_h, img_w, _ = frame.shape
    face_2d = []
    face_3d = []
    focal_length = 1 * img_w

    camera_matrix = np.array(
        [
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1],
        ]
    )

    for face_landmarks in result.face_landmarks:
        for idx, landmark in enumerate(face_landmarks):
            if (
                idx == 1
                or idx == 33
                or idx == 61
                or idx == 199
                or idx == 263
                or idx == 291
            ):
                x, y = (int(landmark.x * img_w), int(landmark.y * img_h))
                face_2d.append([x, y])
                face_3d.append([x, y, landmark.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    distrotion_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rvect, _ = cv2.solvePnP(face_3d, face_2d, camera_matrix, distrotion_matrix)

    rmat, _ = cv2.Rodrigues(rvect)

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    (pitch, yaw, roll) = (angles[0] * 360, angles[1] * 360, angles[2] * 360)

    if pitch > 15:
        print("Looking up")
    elif pitch < -15:
        print("Looking down")
    elif yaw > 10:
        print("Looking right")
    elif yaw < -10:
        print("Looking left")
    else:
        print("Straight")


cap = cv2.VideoCapture(0)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=landmarker_result_callback,
)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

start_time = time.time()

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("unable to capture")
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("Frame", frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(data=rgb, image_format=mp.ImageFormat.SRGB)

        ts = int((time.time() - start_time) * 1000)

        landmarker.detect_async(mp_image, ts)

        estimate_headpose(frame, latest_result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
