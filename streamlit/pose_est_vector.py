import os
import cv2
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from ultralytics import YOLO

####미완####

SAVE_PATH = "/opt/ml/streamlit__/detection_results"
center = [55,28]
@st.cache_data
def load_pose_model(model_path):
    pose_model = YOLO(model_path)
    return pose_model

def calculate_eye_center(eye_points): ######
    x = np.mean(eye_points[:, 0])
    y = np.mean(eye_points[:, 1])
    return int(x), int(y)

def calculate_vector(center, current_point): ########
    return (current_point[0] - center[0], current_point[1] - center[1])

def visualize_eye_vectors(image, bounding_boxes, keypoints): #######
    fig, ax = plt.subplots(figsize=(10, 5))  
    
    for idx, bounding_box in enumerate(bounding_boxes):
        x_min, y_min, x_max, y_max = map(int, bounding_box)
        eye_image = image[y_min:y_max, x_min:x_max]

        # 눈의 중심 좌표 계산
        keypoint_ = keypoints[idx]
        
        pupil_point = keypoint_[0]
        print(keypoint_)
        # 벡터 계산
        vector = calculate_vector(center, pupil_point)

        # 이미지와 눈의 중심 좌표, 벡터 시각화
        ax.imshow(eye_image)
        ax.scatter(pupil_point[0], pupil_point[1], color='green', s=2)
        ax.scatter(center[0], center[1], color='blue')
        ax.arrow(center[0], center[1], vector[0], vector[1], color='red', head_width=1.5, head_length=1.5)
        #ax.set_title(f"이미지 {idx}")

        # Streamlit에 이미지 출력
        st.pyplot(fig)
        fig.clear()
        
def get_pose_estimation(model, image):
    results = model.predict(source=image)[0]
    boxes = results.boxes
    class_  = boxes.cls
    keypoints = results.keypoints.xy
    keypoints = [point.tolist() for point in keypoints]
    bounding_boxes = boxes.xyxy

    # Matplotlib를 사용하여 이미지와 경계 상자 표시
    fig, ax = plt.subplots()
    #ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # for box, cls in zip(bounding_boxes, class_):
    #     cls = int(cls.item())
    #     print("Class:", cls)
    #     box = [int(item.item()) for item in box]
    #     print(box)
    #     x_min, y_min, x_max, y_max = box
    #     print("Box:", x_min, y_min, x_max, y_max)
    #     rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
    #                          fill=False, edgecolor='green', linewidth=2)
    #     ax.add_patch(rect)
    #     ax.text(x_min, y_min - 5, f'Class: {cls}', color='green')

    #st.pyplot(fig)

    # 키포인트 시각화
    #fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for keypoint in keypoints[0]:
        #print(keypoint)
        x, y = keypoint
        ax.plot(x, y, 'ro')

    
    st.pyplot(fig)
