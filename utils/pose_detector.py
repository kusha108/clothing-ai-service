import cv2
import mediapipe as mp

#  NEW API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

MODEL_PATH = "pose_landmarker.task"


def detect_pose(image_path):
    try:
        image = cv2.imread(image_path)

        if image is None:
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE
        )

        detector = PoseLandmarker.create_from_options(options)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        return {
            "left_shoulder": {
                "x": left_shoulder.x,
                "y": left_shoulder.y
            },
            "right_shoulder": {
                "x": right_shoulder.x,
                "y": right_shoulder.y
            }
        }

    except Exception as e:
        print("Pose Error:", e)
        return None