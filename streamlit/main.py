import streamlit as st
from confirm_button_hack import cache_on_button_press

# import torch
# import torchvision
import yaml

# from predict import get_prediction, load_model
from detect import get_detection, load_det_model
from segment import get_segmentation, load_seg_model

import os
import io
from PIL import Image
import cv2
import glob
import time

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


SAVE_PATH = "/opt/ml/eye_phone_streamlit/detection_results"

def run_detection(config_file):

    st.header("run eye detection..")

    model = load_det_model(config_file)
    
    # with open(config_file) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])
    
    if uploaded_file:
        start_time = time.time()
        image = Image.open(uploaded_file)
        with st.spinner("Running Eye detection..."):
            get_detection(model, image, start_time)
            

def run_segmentation(config_file):

    st.header("run eye segmentation..")

    model = load_seg_model(config_file)
    model.cuda()
    model.eval()

    # with open(config_file) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    image_paths = glob.glob(SAVE_PATH + "/*")
    for image_path in image_paths:
        start_time = time.time()
        image = cv2.imread(image_path)
        with st.spinner("Running Eye segmentation..."):
            result = get_segmentation(model, image, start_time)
        

# root_password = 'password'

# @cache_on_button_press('Authenticate')
# def authenticate(password) ->bool:
#     print(type(password))
#     return password == root_password

# password = st.text_input('password', type="password")

# if authenticate(password):
#     st.success('You are authenticated!')
#     main()
# else:
#     st.error('The password is invalid.')


if __name__ == "__main__":

    # def clear_cache():
    #     st.cache_data.clear()
    # st.sidebar.button("Refresh Program",on_click=clear_cache)
    st.cache_data.clear()

    folder = glob.glob(SAVE_PATH + "/*")
    for file in folder:
        os.remove(file)

    # config_det_yolov5.yaml / config_det_yolov8.yaml
    run_detection("config_det_yolov8.yaml")
    # config_seg_ritnet.yaml / config_seg_unet++_mobilenet.yaml
    run_segmentation("config_seg_unet++_mobilenet.yaml")
