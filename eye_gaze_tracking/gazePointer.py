#%%

import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

from getGesture import get_gesture
import streamlit as st


# 홍채 좌표 리스트
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# 눈꺼풀 좌표 리스트
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 ]

# 카메라 설정
camera_pos = np.array([0, 0, 0]) # 카메라 중심 좌표


def rotate_y(angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    return np.array([[cos_theta, 0, -sin_theta],
                     [0, 1, 0],
                     [sin_theta, 0, cos_theta]])

def rotate_x(angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, cos_theta, -sin_theta],
                     [0, sin_theta, cos_theta]])

def map_3d_to_2d(image, mesh_points): # red_dot_pos
    # eye_x, eye_y, eye_z, screen_width, screen_height

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape 
    print('width, height : ', width, height)

    global LEFT_IRIS, LEFT_EYE, RIGHT_IRIS, RIGHT_EYE
    global camera_pos

    all_eye_coords = [(landmark.x, landmark.y, landmark.z) for landmark in mesh_points] # 모든좌표 불러오기
    left_eye_coords = [all_eye_coords[idx] for idx in LEFT_IRIS]
    right_eye_coords = [all_eye_coords[idx] for idx in RIGHT_IRIS]

    # 눈 중점 계산
    # left_eye_center = np.mean(left_eye_coords, axis=0)
    # right_eye_center = np.mean(right_eye_coords, axis=0)
    left_eye_center = calculate_eye_center(left_eye_coords) # 눈 중점 계산
    right_eye_center = calculate_eye_center(right_eye_coords)

    gaze_vector = np.array(left_eye_center) - np.array(right_eye_center) # 두 동공의 위치를 연결
    gaze_vector = gaze_vector / np.linalg.norm(gaze_vector) # gaze vector를 정규화하여 단위 벡터로 만들기
    print("Gaze Vector :", gaze_vector)

    eye_direction = gaze_vector - camera_pos # 눈의 방향 벡터 계산
    pitch = np.degrees(np.arcsin(eye_direction[1] / np.linalg.norm(eye_direction))) # 카메라와 눈의 방향 벡터 사이의 각도 계산 -> degree로 변환
    yaw = np.degrees(np.arctan2(eye_direction[0], eye_direction[2]))

    # 카메라 회전 변환 계산
    # 카메라의 시선 방향 벡터 (camera_lookat) 계산
    target_x, target_y, target_z = gaze_vector[0], gaze_vector[1], gaze_vector[2]
    camera_lookat = np.array([target_x, target_y, target_z]) - np.array(camera_pos)
    # camera_dir = np.array([0, 0, -1])
    camera_dir = np.array(camera_lookat) - np.array(camera_pos)
    camera_dir = camera_dir / np.linalg.norm(camera_dir)
    rotation_matrix = rotate_y(yaw) @ rotate_x(pitch)
    camera_dir = rotation_matrix @ camera_dir

    # 카메라 행렬 구하기
    # camera_lookat = camera_pos + camera_dir
    view_matrix = np.array([
        [*np.cross([0, 1, 0], camera_dir), -np.dot(np.cross([0, 1, 0], camera_dir), camera_pos)],
        [0, 1, 0, 0],
        [*-camera_dir, np.dot(camera_dir, camera_pos)],
        [0, 0, 0, 1]
    ])

    # 3D 좌표를 카메라 기준 좌표계로 변환
    eye_world = np.array([target_x, target_y, target_z, 1])
    eye_camera = np.dot(view_matrix, eye_world)

    # Perspective Divide (투영 변환)
    if eye_camera[3] != 0:
        eye_camera /= eye_camera[3]

    # 2D 화면 좌표로 변환
    # screen_x = int((eye_camera[0] + 1) * 0.5 * screen_width)
    # screen_y = int((1 - (eye_camera[1] + 1) * 0.5) * screen_height)
    screen_x = int((eye_camera[0] + 1) * 0.5 * width) # red_dot_pos[0]
    screen_y = int((1 - (eye_camera[1] + 1) * 0.5) * height) # red_dot_pos[1]

    cv2.putText(image, f'Pitch: {pitch:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f'Yaw: {yaw:.2f}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(image, (int(screen_x), int(screen_y)), 10, (0, 0, 255), -1)


    # OpenCV에서 읽어온 BGR 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb)
    # 이미지 시각화
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Eye Visualization')
    plt.show()

    print(f"화면 좌표: ({screen_x}, {screen_y})")

    return screen_x, screen_y


def eye_gaze_tracking(frame, mesh_points):
    print("*** eye_tracking ***")
    # if not frame:
    #     print("<< video >>")
    #     video_capture = cv2.VideoCapture(0)
    #     red_dot_position = [int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2, int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2]
    #     while True:
    #         ret, frame = video_capture.read()
    #         if not ret:
    #             break
    #         # result_frame = map_3d_to_2d(frame, red_dot_position, mesh_points)
    #         # cv2.imshow('Eye Tracking Result', result_frame)
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     break
    #         screen_x, screen_y = map_3d_to_2d(frame, red_dot_position, mesh_points)

    # red_dot_position = [int(frame.shape[0]) // 2, int(frame.shape[1]) // 2]
    # result_frame = map_3d_to_2d(frame, red_dot_position, mesh_points)
    # cv2.imshow('Eye Tracking Result', result_frame)
    image = cv2.imread(frame)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = cv2.flip(image, 1)
    screen_x, screen_y = map_3d_to_2d(image, mesh_points)
    
    return screen_x, screen_y


def calculate_eye_center(eye_points):
    x_coords = [x for x, y, z in eye_points]
    y_coords = [y for x, y, z in eye_points]
    z_coords = [z for x, y, z in eye_points]
    eye_center = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords), sum(z_coords) / len(z_coords))
    return eye_center


def get_pupil_from_iris(points):
    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]
    center = (int(sum(x_points) / len(points)), int(sum(y_points) / len(points))) # tuple
    return center



if __name__ == "__main__":

    mp_face_mesh = mp.solutions.face_mesh
    frame_path = '/opt/ml/Mediapipe/test_unni_cardinal/KakaoTalk_20230723_124644889_03.jpg'

    input_frame = cv2.imread(frame_path)
    # print(input_frame.shape)

    # get base position
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:
        input_frame = cv2.flip(input_frame, 1)
        rgb_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        #getting width and height or frame
        img_h, img_w = input_frame.shape[:2]

    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
    # print(results.multi_face_landmarks)
    # print(mesh_points)
    
    # pupils = get_eye_movement(mesh_points, input_frame)
    left_pupil = get_pupil_from_iris(mesh_points[LEFT_IRIS])
    right_pupil = get_pupil_from_iris(mesh_points[RIGHT_IRIS])

    # gesture = get_gesture(mesh_points, input_frame)

    # cv2.polylines(input_frame, [mesh_points[LEFT_IRIS]], True, (0,0,255), 1, cv2.LINE_AA)
    # cv2.polylines(input_frame, [mesh_points[RIGHT_IRIS]], True, (0,0,255), 1, cv2.LINE_AA)
    # cv2.polylines(input_frame, [mesh_points[LEFT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
    # cv2.polylines(input_frame, [mesh_points[RIGHT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
    # cv2.drawMarker(input_frame, left_pupil, (255,255,255), line_type = cv2.LINE_AA)
    # cv2.drawMarker(input_frame, right_pupil, (255,255,255), line_type = cv2.LINE_AA)


    # eye gaze tracking
    # eye_gaze_tracking(input_frame, results.multi_face_landmarks[0].landmark)
    eye_gaze_tracking(None, results.multi_face_landmarks[0].landmark)
# %%
