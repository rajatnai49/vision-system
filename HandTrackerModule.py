import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarksConnections
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)


class handDetector:
    module_path = "/home/rajatnai49/projects/vision-system/hand_landmarker.task"

    def __init__(self, num_hands=2) -> None:
        self.result: HandLandmarkerResult | None = None
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.module_path),
            running_mode=VisionTaskRunningMode.LIVE_STREAM,
            num_hands=num_hands,
            result_callback=self._save_result,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def _save_result(
        self, result: HandLandmarkerResult, output_img: mp.Image, timestamp_ms: int
    ):
        self.result = result

    def findHands(self, img, draw=True):
        mp_hands = mp.tasks.vision.HandLandmarksConnections
        mp_drawing = mp.tasks.vision.drawing_utils
        mp_drawing_styles = mp.tasks.vision.drawing_styles

        if self.result:
            for hands in self.result.hand_landmarks:
                # points = []
                # for lm in hands:
                #     cx, cy = int(lm.x * w), int(lm.y * h)
                #     points.append((cx, cy))

                if draw:
                    mp_drawing.draw_landmarks(
                        img,
                        hands,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
        return img
