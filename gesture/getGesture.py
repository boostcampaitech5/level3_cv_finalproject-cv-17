import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# from coordinate import Coordinate
# from shapely.geometry import Polygon


# 홍채 좌표 리스트
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# 눈꺼풀 좌표 리스트
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 ]

# BLINK_COUNT = 0
# BASE_POSITION = {}

# 아래 끝 2초간 보기 -> 스크롤 아래
# 위 끝 2초간 보기 -> 스크롤 위
# 오른쪽 끝 2초간 보기 -> 앞으로 가기
# 왼쪽 끝 2초간 보기 -> 뒤로 가기


def get_pupil_from_iris(points):
    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]
    center = (int(sum(x_points) / len(points)), int(sum(y_points) / len(points))) # tuple
    return center


def get_gesture(mesh_points):
    global LEFT_IRIS, RIGHT_IRIS, LEFT_EYE, RIGHT_EYE
    # mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
    # print(mesh_points) 
    
    # get current pupils
    left_pupil = get_pupil_from_iris(mesh_points[LEFT_IRIS])
    right_pupil = get_pupil_from_iris(mesh_points[RIGHT_IRIS])
    print('left_pupil\n', left_pupil)
    print('right_pupil\n', right_pupil)

    # if Polygon(mesh_points[LEFT_IRIS]).area > Polygon(mesh_points[LEFT_EYE]).area:
    #     gesture = 'left_eye_closed' # L_cor = 0
    #     print(gesture)
    # if Polygon(mesh_points[RIGHT_IRIS]).area > Polygon(mesh_points[RIGHT_EYE]).area:
    #     gesture = 'right_eye_closed' # R_cor = 0
    #     print(gesture)
    # else:
    #     # get gesture
    gesture = limit_of_gesture(mesh_points, left_pupil, right_pupil)
    
    return gesture
    
    
def limit_of_gesture(mesh_points, left_pupil, right_pupil):
    
    # LEFT_EYE_CORDINAL = [362, 374, 263, 386] # left / bottom / right / top
    # RIGHT_EYE_CORDINAL = [33, 145, 133, 159] # left / bottom / right / top
    # LEFT_IRIS = [474, 475, 476, 477] # right / bottom / left / top
    # RIGHT_IRIS = [469, 470, 471, 472] # right / bottom / left / top

    ## left eye 기준
    leftEyeCardinals = {
        'eye' : {
            'left' : mesh_points[362], # left_eye[0],
            'bottom' : mesh_points[374], # left_eye[4], # 12
            'right' : mesh_points[263], # left_eye[8],
            'top' : mesh_points[386], # left_eye[12], # 4
        },
        'iris' : {
            'left' : mesh_points[476], # left_iris[2],
            'bottom' : mesh_points[475], # left_iris[1], # 3
            'right' : mesh_points[474], # left_iris[0],
            'top' : mesh_points[477], # 1
        }
    }
    rightEyeCardinals = {
        'eye' : {
            'left' : mesh_points[33], # right_eye[0],
            'bottom' : mesh_points[145], # right_eye[4], # 12
            'right' : mesh_points[133], # right_eye[8],
            'top' : mesh_points[159], # right_eye[12], # 4
        },
        'iris' : {
            'left' : mesh_points[471], # right_iris[2],
            'bottom' : mesh_points[470], # right_iris[1], # 3
            'right' : mesh_points[469], # right_iris[0],
            'top' : mesh_points[472], # right_iris[3], # 1
        }
    }
    print('leftEyeCardinals\n', leftEyeCardinals)
    print('rightRyeCardinals\n', rightEyeCardinals)

    eye_move = {
        'look_left' : 5, # 'backward',
        'look_right' : 6, # 'forward',
        'look_up' : 2, # 'scroll_up',
        'look_down' : 1, # 'scroll_down',
        # 'closed' : 'blink',
        # 'look_up_with_blink' : 'auto_scroll_up',
        # 'look_down_with_blink' : 'auto_scroll_down',
    }

    left_eye_center_x = int((leftEyeCardinals['eye']['left'][0] + leftEyeCardinals['eye']['right'][0]) / 2)
    left_eye_center_y = int((leftEyeCardinals['eye']['top'][1] + leftEyeCardinals['eye']['bottom'][1]) / 2)
    print(left_eye_center_x)
    print(left_eye_center_y)
    # cv2.drawMarker(input_frame, leftEyeCardinals['eye']['left'], (0, 225, 225), line_type = cv2.LINE_AA)
    
    right_eye_center_x = int((rightEyeCardinals['eye']['left'][0] + rightEyeCardinals['eye']['right'][0]) / 2)
    right_eye_center_y = int((rightEyeCardinals['eye']['top'][1] + rightEyeCardinals['eye']['bottom'][1]) / 2)
    print(right_eye_center_x)
    print(right_eye_center_y)
    # cv2.drawMarker(input_frame, rightEyeCardinals['eye']['right'], (0, 225, 225), line_type = cv2.LINE_AA)

    gesture = []

    # look_left (left_end : center_x = 3 : 2) - left_eye / x좌표 기준
    print('\nleft')
    left_move_limit_x = (leftEyeCardinals['eye']['right'][0]*2 + left_eye_center_x*3) / 5
    print(left_pupil[0], left_move_limit_x)
    # if left_pupil[0] <= left_move_limit_x:
    if left_move_limit_x -5.8 <= left_pupil[0]:
        gesture.append(eye_move['look_left'])

    # look_right (right_end : center_x = 3 : 2) - right_eye / x좌표 기준
    print('\nright')
    right_move_limit_x = (right_eye_center_x*3 + rightEyeCardinals['eye']['left'][0]*2) / 5
    print(right_pupil[0], right_move_limit_x)
    if right_pupil[0] <= right_move_limit_x +5.8:
    # if (right_move_limit_x -3 <= right_pupil[0]) and (right_pupil[0] <= right_move_limit_x +3):
        gesture.append(eye_move['look_right'])

    # look_up (distance)
    print('\nup')
    # dis_x, dis_y = Coordinate(leftEyeCardinals['eye'], rightEyeCardinals['eye'], left_pupil, right_pupil)
    r_foot_xp, r_foot_yp, r_pupil_distance = get_distance(rightEyeCardinals['eye']['left'], rightEyeCardinals['eye']['right'], right_pupil)
    r_foot_xb, r_foot_yb, r_bottom_distance = get_distance(rightEyeCardinals['eye']['left'], rightEyeCardinals['eye']['right'], rightEyeCardinals['eye']['bottom'])
    l_foot_xp, l_foot_yp, l_pupil_distance = get_distance(leftEyeCardinals['eye']['left'], leftEyeCardinals['eye']['right'], left_pupil)
    l_foot_xb, l_foot_yb, l_bottom_distance = get_distance(leftEyeCardinals['eye']['left'], leftEyeCardinals['eye']['right'], leftEyeCardinals['eye']['bottom'])
    print(r_pupil_distance, r_bottom_distance)
    print(l_pupil_distance, r_bottom_distance)
    if l_pupil_distance >= l_bottom_distance or r_pupil_distance >= r_bottom_distance: # and (left_move_limit_x < left_pupil[0] or left_pupil[0] < right_move_limit_x):
        if not gesture:
            gesture.append(eye_move['look_up'])
    
    # # up (eye_bottom < iris_bottom) - left_eye / y좌표 기준 
    # print('up')
    # print(leftEyeCardinals['iris']['bottom'][1], leftEyeCardinals['eye']['bottom'][1])
    # if leftEyeCardinals['iris']['bottom'][1] >= leftEyeCardinals['eye']['bottom'][1] : # and left_pupil[1] > left_eye_center_y:
    #     gesture.append(eye_move['look_up'])

    # down (eye_height < iris_height 5분의 3) - left_eye / 길이 기준
    print('\ndown')
    eye_height = abs(leftEyeCardinals['eye']['top'][1] - leftEyeCardinals['eye']['bottom'][1])
    iris_height = abs(leftEyeCardinals['iris']['top'][1] - leftEyeCardinals['iris']['bottom'][1])
    print(eye_height, iris_height / 5 * 3, iris_height)
    if eye_height < (iris_height / 5 * 3) or (r_pupil_distance < 3 or l_pupil_distance < 3):
        gesture.append(eye_move['look_down'])

    print(gesture)
    if gesture:
        return gesture[0]
    else:
        return 0 # None


    
def get_distance(start_point, end_point, dest_point):
    
    # 선분의 방향벡터 계산
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    # 점 (a, b)와 선분의 시작점 (x1, y1) 사이의 벡터 계산
    ax = dest_point[0] - start_point[0]
    ay = dest_point[1] - start_point[1]

    # 점 (a, b)와 선분의 방향벡터의 내적 계산
    dot = ax * dx + ay * dy

    # 내적 값을 이용하여 수선의 발을 계산
    t = dot / (dx * dx + dy * dy)
    x_foot = start_point[0] + dx * t
    y_foot = start_point[1] + dy * t

    # 주어진 점 (a, b)와 수선의 발 사이의 거리 계산
    distance = ((dest_point[0] - x_foot) ** 2 + (dest_point[1] - y_foot) ** 2) ** 0.5

    # cv2.line(input_frame, start_point, end_point, (0,0,255), 3, cv2.LINE_AA)
    # cv2.line(input_frame, dest_point, (int(x_foot), int(y_foot)), (0,255,255), 5, cv2.LINE_AA)
    # cv2.drawMarker(input_frame, (int(x_foot), int(y_foot)), (0,255,255), 3, line_type = cv2.LINE_AA)

    return x_foot, y_foot, distance
    

