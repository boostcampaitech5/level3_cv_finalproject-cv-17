from typing import Tuple

import streamlit as st
import torch
import yaml

from util_funcs import transform_image


DET_SAVE_PATH = "/opt/ml/eye_phone_streamlit/detection_results"

from ultralytics import YOLO
from PIL import Image
import time

# @st.cache
@st.cache_data
# def load_det_model(config_file) -> YOLO:
def load_det_model(config_file):
    
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=config["model_path"], force_reload=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model.load_state_dict(torch.load(config["model_path"], map_location=device))

    return model


## YOLOv8 ['left_eye', 'right_eye'] / (640, 640)
def get_detection(model, image_path, start_time):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tensor = transform_image(image_bytes=image_bytes).to(device)
    # outputs = model.forward(tensor)
    image = Image.open(image_path)
    image = image.resize((768,480)) # doesn't matter 

    print('^^^^^^^^')
    outputs = model(image)
    print(outputs)
    # print(outputs.xyxy)
    print(outputs.xyxy[0])
    
    # boxes = outputs.boxes
    # class_ = boxes.cls
    # bounding_boxes = boxes.xyxy
    # class_ = outputs.xyxy[0]
    # for box, cls in zip(bounding_boxes, class_):
    #     cls = int(cls.item())
    #     box = [int(item.item()) for item in box]
    #     x1, y1, x2, y2 = box

    #     save_path = DET_SAVE_PATH +'/'+ str(cls) +'.jpg'
        
    #     cropped_image = image.crop((x1, y1, x2, y2))
    #     st.image(cropped_image)
    #     cropped_image.save(save_path)
    
    finish_time = time.time()
    print("detection time per image (save 2 images)", finish_time - start_time)
    return finish_time    
    
