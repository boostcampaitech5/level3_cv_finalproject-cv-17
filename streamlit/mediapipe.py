import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
mp_face_mesh = mp.solutions.face_mesh
 
# 홍채 좌표 리스트
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# 눈꺼풀 좌표 리스트
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]


def visualize_eyes(image):
    frame = np.array(image)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:  
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        img_h, img_w, _ = frame.shape

    if results.multi_face_landmarks[0].landmark: ## 눈이 인식 되었을 때
        mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

        cv2.polylines(frame, [mesh_points[LEFT_IRIS]], True, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)

        ## Y좌표 최대 최소값 구하기
        
        left_eye_points = mesh_points[LEFT_EYE]
        L_max_x = np.max(left_eye_points[11:13, 0])
        L_min_x = np.min(left_eye_points[3:6, 0])
        L_max_y = np.max(left_eye_points[3:5, 1])
        L_min_y = np.min(left_eye_points[11:13, 1])

        right_eye_points = mesh_points[RIGHT_EYE]
        R_max_x = np.max(right_eye_points[11:13, 0])
        R_min_x = np.min(right_eye_points[3:6, 0])   
        R_max_y = np.max(right_eye_points[3:5, 1])
        R_min_y = np.min(right_eye_points[11:13, 1])

        ## CENTER구하기
        R_points = np.array(mesh_points[RIGHT_IRIS])
        R_x_mean = np.mean(R_points[:, 0])
        R_y_mean = np.mean(R_points[:, 1])
        R_center_point = (int(R_x_mean), int(R_y_mean))

        L_points = np.array(mesh_points[LEFT_IRIS])
        L_x_mean = np.mean(L_points[:, 0])
        L_y_mean = np.mean(L_points[:, 1])
        L_center_point = (int(L_x_mean), int(L_y_mean))

        # OpenCV에서 읽어온 BGR 이미지를 RGB로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 중앙점 시각화
        radius = 2  # 원의 반지름
        cv2.circle(image_rgb, R_center_point, radius, (255, 0, 0), -1)
        cv2.circle(image_rgb, L_center_point, radius, (255, 0, 0), -1)

        # 이미지 시각화
        st.image(image_rgb)
        st.text(f"Right Center: ({int(R_x_mean)}, {int(R_y_mean)})")
        st.text(f"Left Center: ({int(L_x_mean)}, {int(L_y_mean)})")
        
        
        if L_max_x-L_min_x > L_max_y-L_min_y: 
            st.text("똑바로 누움")
            st.text(f"L : max-min = {L_max_y-L_min_y}")
            st.text(f"R : max-min = {R_max_y-R_min_y}")
            
            if L_max_y-L_min_y < 5 and R_max_y-R_min_y < 5 : ## L_max_y-L_min_y < 영점잡았을 때 max-min 값의 n% 로 수정
                st.text('close')
            else:
                st.text('open')
        else: ## 옆으로 누울 때 (90도 회전)  ###########################수정 ##################################
            st.text("옆으로 누움")
            st.text(f"L : max-min = {L_max_x-L_min_x}")
            st.text(f"R : max-min = {R_max_x-R_min_x}")
            
            if L_max_x-L_min_x < 5 and R_max_x-R_min_x < 5 : ## L_max_y-L_min_y < 영점잡았을 때 max-min 값의 n% 로 수정
                st.text('close')
            else:
                st.text('open')
                
    else: # 이미지 인식 실패 시 잠시 기다린 후 다시 이미지 받기
        st.warning("No face detected. Please wait for 0.1 seconds and try again.")
        time.sleep(0.1) ##FPS에 따라 시간 조정
        visualize_eyes(image)