import argparse
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

parser = argparse.ArgumentParser(description="Video input demo")
model_path = (
    "/home/rajatnai49/projects/vision-system/blaze_face_full_range_sparse.tflite"
)

parser.add_argument(
    "--video",
    type=str,
    default=None,
    help="Path to video file (leave empty for webcam)",
)

args = parser.parse_args()


def print_result(result, output_image, timestamp_ms):
    if result.detections:
        print(f"[{timestamp_ms}]: detected result - {len(result.detections)}")
    else:
        print("no result detected")


def compute_iou(box_a, box_b):
    """box: x, y, w, h"""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b

    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1, inter_x2 = max(ax1, bx1), min(ax2, bx2)
    inter_y1, inter_y2 = max(ay1, by1), min(ay2, by2)

    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = ((aw * ah) + (bw * bh)) - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def nms(detections, iou_threshold=0.5):
    """detections: list of (x, y, w, h, score)"""
    if not detections:
        return []

    # sort descending based on the score
    detections = sorted(detections, key=lambda d: d[4], reverse=True)

    kept = []

    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [
            d for d in detections if compute_iou(best[:4], d[:4]) < iou_threshold
        ]

    return kept


def sliding_window(detector, frame, start_time, window_size=400, stride=250):
    h, w = frame.shape[:2]
    all_detections = []
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            crop = frame[y : y + window_size, x : x + window_size]
            resized = cv2.resize(crop, (128, 128))
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            ts = int((time.time() - start_time) * 1000)
            result = detector.detect_for_video(
                mp_image, ts
            )

            # print_result(result, mp_image, ts)

            if result.detections:
                scale_x = window_size / 128
                scale_y = window_size / 128

                for det in result.detections:
                    bbox = det.bounding_box

                    orig_x = int(bbox.origin_x * scale_x) + x
                    orig_y = int(bbox.origin_y * scale_y) + y
                    orig_w = int(bbox.width * scale_x)
                    orig_h = int(bbox.height * scale_y)

                    if orig_w < 40 or orig_h < 40:
                        continue

                    confidence_score = det.categories[0].score
                    all_detections.append((orig_x, orig_y, orig_w, orig_h, confidence_score))

    return nms(all_detections, iou_threshold=0.4)


options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.6,
)

if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera or given file not found")
    exit()

start_time = time.time()

with vision.FaceDetector.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Capture Failed")
            break

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
#         ts = int((time.time() - start_time) * 1000)

#         # detector.detect_async(mp_image, ts)
#         result = detector.detect_for_video(mp_image, ts)


        detections = sliding_window(detector, frame, start_time)
        print(f"Faces in current frame: {len(detections)}")

        for (x,y,w,h,score) in detections:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)


        cv2.imshow("Frame", frame)
        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()


