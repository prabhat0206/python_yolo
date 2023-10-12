from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
from PIL import Image
from typing import List
from datetime import datetime
from io import BytesIO
from requests import get

device = "cuda"
model = YOLO('yolov8m.pt')
model = model.to(device = device)

def personDetect(image):
  image = image.convert("RGB")
  output = model(image, classes=[0], conf = 0.2, verbose = False)
  time = datetime.now().strftime("%H%M%S.%f")
  if len(output[0].boxes.cls) > 0:
    print("{0} Person exists {1}".format(time, True))
    return True
  print("{0} Person exists {1}".format(time, False))
  return False

app = FastAPI()


@app.post("/yolo-speed-check")
async def yolo_speed_check(images: List[UploadFile]):
    for image in images:
        img = Image.open(image.file)
        personDetect(img)
    return {"speed": 1}


@app.get("/api/v1/person-detect-url")
async def person_detect_url(url: str):
    response = get(url)
    img = Image.open(BytesIO(response.content))
    return {"person": personDetect(img)}
    
