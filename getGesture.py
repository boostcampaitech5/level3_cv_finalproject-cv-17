import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math
# import streamlit as st

# from coordinate import Coordinate
# from shapely.geometry import Polygon


# 홍채 좌표 리스트
LEFT_IRIS = [ 474, 475, 476, 477 ]
RIGHT_IRIS = [ 469, 470, 471, 472 ]

# 눈꺼풀 좌표 리스트
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 ]

# auto
CEF_COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 2 # 3

# shut down
CEF_COUNTER_FOR_CLOSE = 0
CLOSED_EYES_FRAME_FOR_CLOSE = 4 #7

PREV_GESTURE = 0
UP_COUNT = 0
DOWN_COUNT = 0


# 아래 끝 보기 -> scroll_down
# 위 끝 보기 -> scroll_up
# 오른쪽 끝 보기 -> forward
# 왼쪽 끝 보기 -> backward
# 아래 끝 5 frame 이상 보기 -> auto_scroll_down
# 위 끝 5 frame 이상 보기 -> auto_scroll_up
# 눈 3 frame 이상 감기 -> stop_auto_scroll
# 눈 5 frame 이상 감기 -> shut_down


# 두 점 사이의 거리 
def get_distance_between_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance


# 두 점을 잇는 직선과 점 사이의 거리 / 수선의 발
def get_distance_between_line_and_point(start_point, end_point, dest_point):
    
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



# Blinking Ratio
def blink_ratio(left_eye_cardinals, right_eye_cardinals):

    # right eye
    # horizontal line 
    rh_right = right_eye_cardinals['right']
    rh_left = right_eye_cardinals['left']
    # vertical line 
    rv_top = right_eye_cardinals['top']
    rv_bottom = right_eye_cardinals['bottom']
    # draw lines on right eyes 

    # left eye 
    # horizontal line 
    lh_right = left_eye_cardinals['right']
    lh_left = left_eye_cardinals['left']

    # vertical line 
    lv_top = left_eye_cardinals['top']
    lv_bottom = left_eye_cardinals['bottom']

    rhDistance = get_distance_between_points(rh_right, rh_left)
    rvDistance = get_distance_between_points(rv_top, rv_bottom)

    lvDistance = get_distance_between_points(lv_top, lv_bottom)
    lhDistance = get_distance_between_points(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    
    return ratio 


def count_blink(left_eye_cardinals, right_eye_cardinals):

    global CEF_COUNTER, CLOSED_EYES_FRAME, TOTAL_BLINKS
    global CEF_COUNTER_FOR_CLOSE, CLOSED_EYES_FRAME_FOR_CLOSE
    is_blink = False
    is_off = False

    ratio = blink_ratio(left_eye_cardinals, right_eye_cardinals)
    # print(f'Ratio : {round(ratio,2)}')
    if ratio > 5.5: # 5.5
        CEF_COUNTER += 1
        CEF_COUNTER_FOR_CLOSE += 1
        print(f'closed !  (Ratio : {round(ratio,2)} / CEF_COUNTER : {CEF_COUNTER} / CEF_COUNTER_FOR_CLOSE : {CEF_COUNTER_FOR_CLOSE})')
    else:
        print('opened ! (check closed time)')
        if CEF_COUNTER_FOR_CLOSE > CLOSED_EYES_FRAME_FOR_CLOSE: # 4
            print(f'CEF_COUNTER_FOR_CLOSE: {CEF_COUNTER_FOR_CLOSE}')
            is_off = True
        elif CEF_COUNTER > CLOSED_EYES_FRAME: # 4
            print(f'CEF_COUNTER: {CEF_COUNTER}')
            TOTAL_BLINKS += 1
            is_blink = True
        CEF_COUNTER = 0        
        CEF_COUNTER_FOR_CLOSE = 0
    
    # print(f'CEF_COUNTER: {CEF_COUNTER}')
    print(f'TOTAL_BLINKS: {TOTAL_BLINKS}')
    return is_blink, is_off




def get_gesture(mesh_points):
    global LEFT_IRIS, RIGHT_IRIS, LEFT_EYE, RIGHT_EYE
    # mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
    # print(mesh_points) 
    
    # get current pupils
    left_pupil = get_pupil_from_iris(mesh_points[LEFT_IRIS])
    right_pupil = get_pupil_from_iris(mesh_points[RIGHT_IRIS])
    # print('left_pupil\n', left_pupil)
    # print('right_pupil\n', right_pupil)

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


def get_pupil_from_iris(points):
    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]
    center = (int(sum(x_points) / len(points)), int(sum(y_points) / len(points))) # tuple
    return center

    
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
    # print('leftEyeCardinals\n', leftEyeCardinals)
    # print('rightRyeCardinals\n', rightEyeCardinals)

    eye_move = {
        'look_down' : 1, # 'scroll_down',
        'look_up' : 2, # 'scroll_up',
        'look_right' : 5, # 'forward',
        'look_left' : 6, # 'backward',
        'look_down_cont' : 7, # 'auto_scroll_down',
        'look_up_cont' : 8, # 'auto_scroll_up',
        'blink' : 9, # 'stop_auto_scroll',
        'blink_cont' : 10, # 'shut_down'
    }

    left_eye_center_x = int((leftEyeCardinals['eye']['left'][0] + leftEyeCardinals['eye']['right'][0]) / 2)
    left_eye_center_y = int((leftEyeCardinals['eye']['top'][1] + leftEyeCardinals['eye']['bottom'][1]) / 2)
    # print(left_eye_center_x)
    # print(left_eye_center_y)
    # cv2.drawMarker(input_frame, leftEyeCardinals['eye']['left'], (0, 225, 225), line_type = cv2.LINE_AA)
    
    right_eye_center_x = int((rightEyeCardinals['eye']['left'][0] + rightEyeCardinals['eye']['right'][0]) / 2)
    right_eye_center_y = int((rightEyeCardinals['eye']['top'][1] + rightEyeCardinals['eye']['bottom'][1]) / 2)
    # print(right_eye_center_x)
    # print(right_eye_center_y)
    # cv2.drawMarker(input_frame, rightEyeCardinals['eye']['right'], (0, 225, 225), line_type = cv2.LINE_AA)

    # limit
    # left
    left_move_limit_x = (leftEyeCardinals['eye']['right'][0]*2 + left_eye_center_x*3) / 5
    # right
    right_move_limit_x = (right_eye_center_x*3 + rightEyeCardinals['eye']['left'][0]*2) / 5
    # top
    r_foot_xp, r_foot_yp, r_pupil_distance = get_distance_between_line_and_point(rightEyeCardinals['eye']['left'], rightEyeCardinals['eye']['right'], right_pupil)
    r_foot_xb, r_foot_yb, r_bottom_distance = get_distance_between_line_and_point(rightEyeCardinals['eye']['left'], rightEyeCardinals['eye']['right'], rightEyeCardinals['eye']['bottom'])
    l_foot_xp, l_foot_yp, l_pupil_distance = get_distance_between_line_and_point(leftEyeCardinals['eye']['left'], leftEyeCardinals['eye']['right'], left_pupil)
    l_foot_xb, l_foot_yb, l_bottom_distance = get_distance_between_line_and_point(leftEyeCardinals['eye']['left'], leftEyeCardinals['eye']['right'], leftEyeCardinals['eye']['bottom'])
    # bottom
    eye_height = abs(leftEyeCardinals['eye']['top'][1] - leftEyeCardinals['eye']['bottom'][1])
    iris_height = abs(leftEyeCardinals['iris']['top'][1] - leftEyeCardinals['iris']['bottom'][1])
    
    
    global PREV_GESTURE, UP_COUNT, DOWN_COUNT
    gesture = []

    # look_left (left_end : center_x = 3 : 2) - left_eye / x좌표 기준
    if left_move_limit_x - 5.8 <= left_pupil[0]:
        print('\n>> left <<')
        # print(left_pupil[0], left_move_limit_x)
        # if l_pupil_distance < l_bottom_distance and r_pupil_distance < r_bottom_distance: ### top
        #     if eye_height >= (iris_height / 5 * 3) and (r_pupil_distance >= 3 and l_pupil_distance >= 3): ### bottom
        gesture.append(eye_move['look_left'])

    # look_right (right_end : center_x = 3 : 2) - right_eye / x좌표 기준
    if right_pupil[0] <= right_move_limit_x + 5.8:
        print('\n>> right <<')
        # print(right_pupil[0], right_move_limit_x)
        # if l_pupil_distance < l_bottom_distance and r_pupil_distance < r_bottom_distance: ### top
        #     if eye_height >= (iris_height / 5 * 3) and (r_pupil_distance >= 3 and l_pupil_distance >= 3): ### bottom
        gesture.append(eye_move['look_right'])

    # look_up (distance)
    if l_pupil_distance >= l_bottom_distance + 1.2 or r_pupil_distance >= r_bottom_distance + 1.2:
        if left_pupil[0] < left_move_limit_x - 5.8 and right_move_limit_x + 5.8 < right_pupil[0]: ### left / right
            if eye_height >= (iris_height / 5 * 3) and (r_pupil_distance >= 3 and l_pupil_distance >= 3): # bottom
                if len(gesture) == 0:
                    print('\n>> up <<')
                    # st.write(l_pupil_distance, l_bottom_distance)
                    # st.write(r_pupil_distance, r_bottom_distance)
                    print('(PREV_GESTURE) ', PREV_GESTURE)
                    if PREV_GESTURE == 2:
                        UP_COUNT += 1
                    if UP_COUNT > 4:
                        gesture.append(eye_move['look_up_cont'])
                        UP_COUNT = 0
                    # print(r_pupil_distance, r_bottom_distance)
                    # print(l_pupil_distance, l_bottom_distance)
                    # is_blink, is_off = count_blink(leftEyeCardinals['eye'], rightEyeCardinals['eye'])
                    # print('is_blink ?', is_blink)
                    # if is_blink:
                    #     gesture.append(eye_move['look_up_with_blink'])
                    #     print(eye_move['look_up_with_blink'])
                    #     st.write('--> auto_look_up')
                    #     # on_auto = True
                    else:
                        gesture.append(eye_move['look_up'])


    # down (eye_height < iris_height 5분의 3) - left_eye / 길이 기준
    if eye_height < (iris_height / 5 * 3) or (r_pupil_distance < 3 or l_pupil_distance < 3):
        if left_pupil[0] < left_move_limit_x -5.8 and right_move_limit_x + 5.8 < right_pupil[0]: ### left / right
            if l_pupil_distance < l_bottom_distance and r_pupil_distance < r_bottom_distance: ### top
                if eye_height > 7:
                    if len(gesture) == 0:
                        print('\n>> down <<')
                        # st.write(eye_height, ' < ', (iris_height / 5 * 3))
                        # st.write(r_pupil_distance, ' < 3 / ', l_pupil_distance, '< 3')
                        print('(PREV_GESTURE) ', PREV_GESTURE)
                        if PREV_GESTURE == 1:
                            DOWN_COUNT += 1
                        if DOWN_COUNT > 4:
                            gesture.append(eye_move['look_down_cont'])
                            DOWN_COUNT = 0
                        # print(eye_height, iris_height / 5 * 3, iris_height)
                        # is_blink, is_off = count_blink(leftEyeCardinals['eye'], rightEyeCardinals['eye'])
                        # print('is_blink ?', is_blink)
                        # if is_blink:
                        #     gesture.append(eye_move['look_down_with_blink'])
                        #     print(eye_move['look_down_with_blink'])
                        #     st.write('--> auto_look_down')
                        #     # on_auto = True
                        else:
                            gesture.append(eye_move['look_down'])

    
    is_blink, is_off = count_blink(leftEyeCardinals['eye'], rightEyeCardinals['eye'])
    print(is_blink, is_off)
    if is_off: # 프로그램 종료할것인지
        gesture.append(eye_move['blink_cont'])
        print(eye_move['blink_cont'])
        # st.write('--> ask_shut_down')
    elif is_blink: # 자동 스크롤
        gesture.append(eye_move['blink'])
        print(eye_move['blink'])
        # st.write('--> stop_auto_scroll')

    print(gesture)
    if gesture:
        PREV_GESTURE = gesture[0]
        return gesture[0]
    else:
        return 0 # None
    


if __name__ == '__main__':

    mp_face_mesh = mp.solutions.face_mesh
    
    # get new input
    input_frame_path = 'test_image.jpg'

    input_frame = cv2.imread(input_frame_path)
    print(input_frame.shape)

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

    # print(results.multi_face_landmarks)

    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
    # print(mesh_points)
    
    gesture = get_gesture(mesh_points, input_frame)
    
    left_pupil = get_pupil_from_iris(mesh_points[LEFT_IRIS])
    right_pupil = get_pupil_from_iris(mesh_points[RIGHT_IRIS])

    cv2.polylines(input_frame, [mesh_points[LEFT_IRIS]], True, (0,0,255), 1, cv2.LINE_AA)
    cv2.polylines(input_frame, [mesh_points[RIGHT_IRIS]], True, (0,0,255), 1, cv2.LINE_AA)
    cv2.polylines(input_frame, [mesh_points[LEFT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
    cv2.polylines(input_frame, [mesh_points[RIGHT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
    cv2.drawMarker(input_frame, left_pupil, (255,255,255), line_type = cv2.LINE_AA)
    cv2.drawMarker(input_frame, right_pupil, (255,255,255), line_type = cv2.LINE_AA)
    
    # left_iris = []
    # right_iris = []
    # for i in range(4):
    #     left_iris.append((mesh_points[LEFT_EYE][i]))
    #     right_iris.append((mesh_points[RIGHT_EYE][i]))
    # # print(left_iris)
    # # print(right_iris)
    # for index, color in enumerate([(255,0,0),(0,255,0),(0,0,255),(255,255,255)]):
    #     cv2.drawMarker(input_frame, left_iris[index], color, line_type = cv2.LINE_AA)
    #     cv2.drawMarker(input_frame, right_iris[index], color, line_type = cv2.LINE_AA)

    # OpenCV에서 읽어온 BGR 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

    # 이미지 시각화
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Eye Visualization')
    plt.show()