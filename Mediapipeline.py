import os
import cv2
import time
import numpy as np
import mediapipe as mp

# import streamlit as st


def Mediapipeline(image_path):

    frame = cv2.imread(image_path)
    frame = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
    mp_face_mesh = mp.solutions.face_mesh
    start_time = time.time()
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
    ) as face_mesh:
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        # getting width and height or frame
        img_h, img_w = frame.shape[:2]

    mesh_points = np.array(
        [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

    # cv2.resize(frame, (50, 50))
    # st.image(frame) # remove for using server.py directly

    return mesh_points
