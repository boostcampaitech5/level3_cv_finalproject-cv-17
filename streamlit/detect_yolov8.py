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

    model = YOLO(config["model_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.load_state_dict(torch.load(config["model_path"], map_location=device))

    return model


## YOLOv8 ['left_eye', 'right_eye'] / (640, 640)
## YOLOv8 ['Leye_O', 'Leye_C', Reye_O', Reye_C'] / (640,384)
def get_detection(model, image_path, start_time):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tensor = transform_image(image_bytes=image_bytes).to(device)
    # outputs = model.forward(tensor)
    image = Image.open(image_path)
    #image = image.resize((640, 640)) # doesn't matter 

    outputs = model(image)[0]
    # print('####')
    # print(outputs)
    boxes = outputs.boxes
    class_ = boxes.cls
    bounding_boxes = boxes.xyxy
    for box, cls in zip(bounding_boxes, class_):
        cls = int(cls.item())
        box = [int(item.item()) for item in box]
        x1, y1, x2, y2 = box

        save_path = DET_SAVE_PATH +'/'+ str(cls) +'.jpg'
        
        dx=x2-x1
        dy=y2-y1
        x_1=x1-0.1*dx
        y_1=y1-0.8*dy
        x_2=x2+0.1*dx
        y_2=y2+1.7*dy
        cropped_image = image.crop((x_1, y_1, x_2, y_2))
        # image_size =[x2-x1,y2-y1]
        # after_image_size=[x_2-x_1,y_2-y_1]
        # st.text(image_size)
        # st.text(after_image_size)
        st.image(cropped_image)
        cropped_image.save(save_path)
    
    finish_time = time.time()
    print("detection time per image (save 2 images)", finish_time - start_time)
    return finish_time        

