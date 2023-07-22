import os
import io
import cv2
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time


POSE_SAVE_PATH = "/opt/ml/streamlit/pose_results"
center = [55,28]

@st.cache_data
def load_pose_model(model_path):
    pose_model = YOLO(model_path)
    return pose_model


def get_pose_estimation(model, image_paths, start_time):

    predicts = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for index, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        results = model.predict(source=image)[0]
        # boxes = results.boxes
        # class_  = boxes.cls
        # bounding_boxes = boxes.xyxy

        keypoints = results.keypoints.xy
        keypoints = [point.tolist() for point in keypoints]

        print('>> pose estimation time per image (image %d) : ' %(index+1), time.time() - start_time)

        fig, ax = plt.subplots()
        # fig, ax = plt.subplots(figsize=(0.2,0.2))

        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        for keypoint in keypoints[0]:
            #print(keypoint)
            x, y = keypoint
            ax.plot(x, y, 'ro')

        # Figure 객체를 이미지로 변환
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_eye = buffer.getvalue()

        save_path = POSE_SAVE_PATH +'/'+ str(index) +'.jpg'
        
        st.image(image_eye, use_column_width=True)
        fig.savefig(save_path, format='png')
        # plt.imsave(save_path, image_eye)
    
        predicts.append(keypoints)

    return predicts