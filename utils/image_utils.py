import cv2
import numpy as np
from .color_extractor import get_skin_tone

#  NEW MEDIAPIPE IMPORT
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

MODEL_PATH = "pose_landmarker.task"


#  BODY TYPE DETECTION (UNCHANGED)
def detect_body_type(image):
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE
        )

        detector = PoseLandmarker.create_from_options(options)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)

        if not result.pose_landmarks:
            return "slim"

        landmarks = result.pose_landmarks[0]

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        waist_width = abs(left_hip.x - right_hip.x)

        ratio = shoulder_width / (waist_width + 1e-5)

        if ratio > 1.4:
            return "athletic"
        elif ratio > 1.1:
            return "slim"
        else:
            return "heavy"

    except Exception as e:
        print("Body Detection Error:", e)
        return "slim"


# NEW: BASIC GENDER DETECTION (simple heuristic)
def detect_gender(body):
    # simple logic (you can upgrade later with ML)
    if body == "athletic":
        return "male"
    elif body == "slim":
        return "unisex"
    else:
        return "female"


# 🔹 GENERATE RECOMMENDATIONS (IMPROVED)
def generate_recommendations(skin, body, gender):
    recommendations = []

    # STYLE BASED ON BODY
    if body == "slim":
        style = "oversized"
    elif body == "athletic":
        style = "fitted"
    else:
        style = "loose-fit"

    # COLORS BASED ON SKIN
    if skin == "light":
        colors = ["black", "navy", "maroon"]
    elif skin == "medium":
        colors = ["olive", "beige", "brown"]
    else:
        colors = ["white", "yellow", "red"]

    # NEW: CATEGORY BASED ON GENDER
    if gender == "female":
        top = "tops"
        bottom = "skirts"
    else:
        top = "t-shirts"
        bottom = "jeans"

    recommendations.extend([
        {"category": top, "style": style, "colors": colors},
        {"category": "hoodies", "style": "oversized", "colors": colors},
        {"category": bottom, "style": "fitted", "colors": colors},
        {"category": "joggers", "style": "fitted", "colors": colors},
        {"category": "shoes", "style": "casual", "colors": ["black", "white"]},
        {"category": "caps", "style": "streetwear", "colors": colors}
    ])

    return recommendations


#  MAIN FUNCTION (UPDATED)
def analyze_image(path):
    try:
        image = cv2.imread(path)

        if image is None:
            return {
                "skinTone": "medium",
                "bodyType": "slim",
                "gender": "male",
                "recommendations": []
            }

        #  DETECTION
        skin = get_skin_tone(image)
        body = detect_body_type(image)

        #  NEW
        gender = detect_gender(body)

        recommendations = generate_recommendations(skin, body, gender)

        return {
            "skinTone": skin,
            "bodyType": body,
            "gender": gender,   #  IMPORTANT
            "recommendations": recommendations
        }

    except Exception as e:
        return {
            "skinTone": "medium",
            "bodyType": "slim",
            "gender": "male",
            "recommendations": [],
            "error": str(e)
        }