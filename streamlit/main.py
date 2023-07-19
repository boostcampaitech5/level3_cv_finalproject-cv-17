import streamlit as st
from confirm_button_hack import cache_on_button_press

# import torch
# import torchvision
import yaml

# from detect import get_detection, load_det_model
# from segment import get_segmentation, load_seg_model

import os
import io
import cv2
import glob
import time

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


DET_SAVE_PATH = "/opt/ml/eye_phone_streamlit/detection_results"
SEG_SAVE_PATH = "/opt/ml/eye_phone_streamlit/segmentation_results"
POSE_SAVE_PATH = "/opt/ml/eye_phone_streamlit/pose_results"
MASK_SAVE_PATH = "/opt/ml/eye_phone_streamlit/mask_result"

def run_detection(config_file):

    st.subheader("run eye detection..")

    model = load_det_model(config_file)
    
    # with open(config_file) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])
    
    if uploaded_file:
        start_time = time.time()
        # image = Image.open(uploaded_file)
        with st.spinner("Running Eye detection..."):
            # get_detection(model, image, start_time)
            get_detection(model, uploaded_file, start_time)
            

def run_segmentation(config_file):

    st.subheader("run eye segmentation..")

    model = load_seg_model(config_file)
    model.eval()

    # with open(config_file) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    image_paths = glob.glob(DET_SAVE_PATH + "/*")
    # for image_path in image_paths:
    #     start_time = time.time()
    #     # image = cv2.imread(image_path)
    #     with st.spinner("Running Eye segmentation..."):
    #         # result = get_segmentation(model, image, start_time)
    #         result = get_segmentation(model, image_path, start_time)
    start_time = time.time()
    with st.spinner("Running Eye segmentation..."):
        # result = get_segmentation(model, image, start_time)
        result = get_segmentation(model, image_paths, start_time)


def run_pose_estimation(model_path):

    st.subheader("run pose estimation..")

    model = load_pose_model(model_path)

    # with open(config_file) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    image_paths = glob.glob(DET_SAVE_PATH + "/*")
    # for image_path in image_paths:
    #     start_time = time.time()
    #     # image = cv2.imread(image_path)
    #     with st.spinner("Running Eye segmentation..."):
    #         # result = get_segmentation(model, image, start_time)
    #         result = get_segmentation(model, image_path, start_time)
    start_time = time.time()
    with st.spinner("Running Eye point estimation..."):
        # result = get_segmentation(model, image, start_time)
        result = get_pose_estimation(model, image_paths, start_time)
        

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

    folder = glob.glob(DET_SAVE_PATH + "/*")
    for file in folder:
        os.remove(file)
    folder = glob.glob(SEG_SAVE_PATH + "/*")
    for file in folder:
        os.remove(file)
    folder = glob.glob(POSE_SAVE_PATH + "/*")
    for file in folder:
        os.remove(file)

    # from detect_yolov8 import get_detection, load_det_model
    # det_config = "config_det_yolov8.yaml"

    from detect_yolov8 import get_detection, load_det_model
    det_config = "config_det_yolov8_with_close.yaml"

    # from detect_yolov5 import get_detection, load_det_model
    # det_config = "config_det_yolov5.yaml"
    
    # from segment_unetplusplus import get_segmentation, load_seg_model
    # seg_config = "config_seg_unet++_mobilenet.yaml"
    
    #segment_unetplusplus / segment_ritnet
    from segment_unetplusplus import get_segmentation, load_seg_model
    seg_config = "config_seg_unet++_mobilenet.yaml"

    from pose_est import get_pose_estimation, load_pose_model
    pose_model_path = "./assets/pose/pose_yolov8.pt"

    from post_processing import post_processing
    image_directory = "/opt/ml/eye_phone_streamlit/mask_result"
    
    # config_det_yolov5.yaml / config_det_yolov8.yaml
    run_detection(det_config)
    # config_seg_ritnet.yaml / config_seg_unet++_mobilenet.yaml
    run_segmentation(seg_config)
    # config_seg_ritnet.yaml / config_seg_unet++_mobilenet.yaml
    #run_pose_estimation(pose_model_path)
    post_processing(image_directory)