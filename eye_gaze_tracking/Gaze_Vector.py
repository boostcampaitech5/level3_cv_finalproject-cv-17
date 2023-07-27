import cv2
import mediapipe as mp
import numpy as np
from math import atan2, degrees


mp_face_mesh = mp.solutions.face_mesh

def clip_point_to_screen(point, width, height):
    x, y = point
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return x, y

def calculate_eye_center(eye_points):
    x_coords = [x for x, y, z in eye_points]
    y_coords = [y for x, y, z in eye_points]
    z_coords = [z for x, y, z in eye_points]
    eye_center = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords), sum(z_coords) / len(z_coords))
    return eye_center

def calculate_eye_gaze_direction(eye_center, gaze_vector):
    gaze_x, gaze_y, gaze_z = gaze_vector
    pitch = degrees(atan2(gaze_z, gaze_x)) # 위아래 판단
    yaw = degrees(atan2(gaze_y, gaze_x))   # 좌우 판단
    return pitch, yaw

def convert_to_absolute_coordinates(relative_coords, width, height):
    absolute_coords = []
    for x, y, z in relative_coords:
        absolute_x = int(x * width)
        absolute_y = int(y * height)
        absolute_coords.append((absolute_x, absolute_y))
    return absolute_coords

def eye_tracking(image, red_dot_pos):
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape 

    with mp_face_mesh.FaceMesh(static_image_mode=False,refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            left_eye_landmarks = [474, 475, 476, 477]
            right_eye_landmarks = [469, 470, 471, 472]
            
            all_eye_coords = [(landmark.x, landmark.y, landmark.z) for landmark in results.multi_face_landmarks[0].landmark] # 모든좌표 불러오기
            left_eye_coords = [all_eye_coords[idx] for idx in left_eye_landmarks]
            right_eye_coords = [all_eye_coords[idx] for idx in right_eye_landmarks]
            
            left_eye_coords_absolute = convert_to_absolute_coordinates(left_eye_coords, width, height)      # 상대좌표를 절대좌표로 변환
            right_eye_coords_absolute = convert_to_absolute_coordinates(right_eye_coords, width, height) 

            left_eye_center = calculate_eye_center(left_eye_coords) # 눈의 중심구하기
            right_eye_center = calculate_eye_center(right_eye_coords)

            gaze_vector = (right_eye_center[0] - left_eye_center[0], right_eye_center[1] - left_eye_center[1], right_eye_center[2] - left_eye_center[2]) # 임시 수정 필요...
            gaze_vector = np.array(gaze_vector)

            pitch, yaw = calculate_eye_gaze_direction(left_eye_center, gaze_vector)

            cv2.polylines(image, [np.array(left_eye_coords_absolute, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.polylines(image, [np.array(right_eye_coords_absolute, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(image, f'Pitch: {pitch:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Yaw: {yaw:.2f}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            
            red_dot_pos[0] += int(yaw)//10
            red_dot_pos[1] += int(pitch)//10

            red_dot_pos[0], red_dot_pos[1] = clip_point_to_screen(red_dot_pos, width, height)
            cv2.circle(image, tuple(red_dot_pos), 10, (0, 0, 255), -1)

    return image


video_capture = cv2.VideoCapture(0)


red_dot_position = [int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2, int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2]

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    result_frame = eye_tracking(frame, red_dot_position)
    cv2.imshow('Eye Tracking Result', result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
