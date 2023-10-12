from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
from PIL import Image
from typing import List
import torch
from datetime import datetime
import os

device = "cuda"
model = YOLO('yolov8m.pt')
model = model.to(device=device)

# Initialize a counter variable
run_counter = 0

def personDetect(image):
    # image = image.convert("RGB")
    # output = model(image, classes=[0])
    with torch.no_grad():
        outputs = model(image, classes=[0], conf = 0.2, verbose = False)
    result = []
    for output in outputs:
        if len(output.boxes.cls) > 0:
            result.append(True)
        else:
            result.append(False)

    return result

app = FastAPI()

@app.post("/yolo-speed-check")
async def yolo_speed_check(images: List[UploadFile]):
    global run_counter  # Access the counter variable
    batch = []
    for image in images:
        with Image.open(image.file) as img:
            batch.append(img.convert("RGB"))
        image.file.close()
    
    person_exist_list = personDetect(batch)

    # Increment the counter
    run_counter += 1

    # Check if the counter has reached 100, and clear CUDA cache if so
    print(run_counter)
    if run_counter >= 100:
        torch.cuda.empty_cache()
        run_counter = 0  # Reset the counter

    return person_exist_list
