 
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
from PIL import Image
from typing import List
import torch
import io
import cv2
import numpy as np
from datetime import datetime
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# device = "cuda"
# model = YOLO('yolov8s.pt')
# model = model.to(device=device)
model = torch.hub.load("ultralytics/yolov5", "yolov5l")
model.conf = 0.2  
model.classes = [0]  
model.amp = True 

# Initialize a counter variable
run_counter = 0
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
def personDetect(image):
    # image = image.convert("RGB")
    # output = model(image, classes=[0])
    with torch.no_grad():
        outputs = model(image)
    
    result = []
    
    for results in outputs:
        res = results[:, 4:5] * results[:, 5:6]
        final = (res > 0.2).any()
        if final:
            result.append(True)
        else:
            result.append(False)

    return result

app = FastAPI()

@app.post("/yolo-speed-check")
def yolo_speed_check(images: List[UploadFile]):
    global run_counter  # Access the counter variable
    batch = []
    for image in images:
        # im_bytes = im_file.read()
        # im = Image.open(io.BytesIO(image.file))
        with Image.open(image.file) as img:
            frame = img.convert("RGB")
            
        frame = letterbox(np.array(frame), new_shape = (640, 640))[0]

        frame = frame.transpose((2, 0, 1)) # HWC to CHW, BGR to RGB
        frame = np.ascontiguousarray(frame)  # contiguous
        # Perform inference on the frame
        frame = torch.from_numpy(frame).to(device  ="cuda").float()

        frame /= 255
        frame = frame[None]
        
        batch.append(frame)
      # Process the results (e.g., draw bounding boxes)
      # processed_frame = results.render()[0]
      
      
    batch = torch.cat(batch, dim=0)
    person_exist_list = personDetect(batch)

    # Increment the counter
    run_counter += 1

    # Check if the counter has reached 100, and clear CUDA cache if so
    print(run_counter)
    if run_counter >= 50:
        # torch.cuda.empty_cache()
        run_counter = 0  # Reset the counter

    return person_exist_list
