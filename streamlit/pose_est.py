import os
import io
import cv2
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from ultralytics import YOLO

SAVE_PATH = "/opt/ml/streamlit__/detection_results"
center = [55,28]
@st.cache_data
def load_pose_model(model_path):
    pose_model = YOLO(model_path)
    return pose_model


def get_pose_estimation(model, image):
    results = model.predict(source=image)[0]
    keypoints = results.keypoints.xy
    keypoints = [point.tolist() for point in keypoints]

    fig, ax = plt.subplots()
    #fig, ax = plt.subplots(figsize=(0.2,0.2))

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

    # 스트림릿에서 이미지로 표시
    st.image(image_eye, use_column_width=True)
        
    
    #st.pyplot(fig)
    

