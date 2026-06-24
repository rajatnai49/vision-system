import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import RunningMode

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
FaceLandmarker = mp.tasks.vision.FaceLandmarker

latest_result = None

model_path = "/home/rajatnai49/projects/vision-system/face_landmarker.task"

def set_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

    print(f"face landmarker result: {result}")


def draw_landmarks(frame, result):
    if result is None or not result.face_landmarks:
        return frame
    h, w, _ = frame.shape
    for face_landmarks in result.face_landmarks:
        for landmark in face_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    return frame

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=set_result
)

start_time = time.time()
cap = cv2.VideoCapture(0)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Nothing to capture")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - start_time) * 1000)

        landmarker.detect_async(mp_image, ts)

        frame = draw_landmarks(frame, latest_result)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
