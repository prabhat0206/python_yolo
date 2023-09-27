from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
from PIL import Image
from typing import List
from datetime import datetime

device = "cpu"
model = YOLO('yolov8m.pt')
model = model.to(device = device)

def personDetect(image):
  image = image.convert("RGB")
  output = model(image, classes=[0])
  time = datetime.now().strftime("%H%M%S.%f")
  if len(output[0].boxes.cls) > 0:
    print("{0} Person exists {1}".format(time, True))
    return True
  print("{0} Person exists {1}".format(time, False))
  return False

app = FastAPI()


@app.post("/yolo-speed-check")
def yolo_speed_check(images: List[UploadFile]):
    for image in images:
        img = Image.open(image.file)
        person_exist = personDetect(img)
        return {"is_person_exist": person_exist}
    return {"is_person_exist": False}

