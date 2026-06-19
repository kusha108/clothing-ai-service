from fastapi import FastAPI, File, UploadFile
import shutil

from utils.image_utils import analyze_image
from utils.pose_detector import detect_pose   #  NEW

app = FastAPI()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    #  EXISTING AI
    result = analyze_image(path)

    #  ADD POSE DETECTION
    pose = detect_pose(path)

    result["pose"] = pose   # ✅ ADD THIS

    return result