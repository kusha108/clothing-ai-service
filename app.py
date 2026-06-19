from fastapi import FastAPI, File, UploadFile
import shutil
import os

from utils.image_utils import analyze_image
from utils.pose_detector import detect_pose   #  NEW

app = FastAPI()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_image(path)
    pose = detect_pose(path)
    result["pose"] = pose

    if os.path.exists(path):
        os.remove(path)

    return result