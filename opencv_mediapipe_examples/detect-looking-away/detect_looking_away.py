import time

import cv2

import mediapipe as mp
import numpy as np
from hard_smoother import HardSmoother
from ema_smoother import EmaSmoother

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
RunningMode = mp.tasks.vision.RunningMode

latest_result = None
model_path = "/home/rajatnai49/projects/vision-system/face_landmarker.task"


def landmarker_result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


def detect_looking_away(frame, result) -> bool:
    if result is None or not result.face_landmarks:
        return True

    img_h, img_w, _ = frame.shape
    face_2d = []
    face_3d = []

    focal_length = 1 * img_w

    camera_mat = np.array(
        [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
    )

    distrotion_mat = np.zeros((4, 1), dtype=np.float64)

    for faces_landmarks in result.face_landmarks:
        for idx, lm in enumerate(faces_landmarks):
            if (
                idx == 1
                or idx == 61
                or idx == 33
                or idx == 199
                or idx == 263
                or idx == 291
            ):
                x = int(lm.x * img_w)
                y = int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    success, rvec, _ = cv2.solvePnP(face_3d, face_2d, camera_mat, distrotion_mat)

    if not success:
        return False

    rmat, _ = cv2.Rodrigues(rvec)

    angles = cv2.RQDecomp3x3(rmat)[0]

    pitch, yaw, _ = (a * 360 for a in angles)

    if pitch > 12 or pitch < -12 or yaw > 12 or yaw < -12:
        return True

    return False


cap = cv2.VideoCapture(0)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=landmarker_result_callback,
)

start_time = time.time()
hardSmoother = HardSmoother()
emaSmoother = EmaSmoother()

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("nothing to capture")
            break

        frame = cv2.flip(frame, 1)

        cv2.imshow("Frame", frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int((time.time() - start_time) * 1000)

        landmarker.detect_async(mp_image, ts)

        detection = detect_looking_away(frame, latest_result)
        hard_detection = hardSmoother.update(detection)
        ema_detection = emaSmoother.update(detection)

        print(f"Normal Detection {detection}")
        print(f"Hard Smoother {hard_detection}")
        print(f"Ema Smoother {ema_detection}")
        print("----")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
