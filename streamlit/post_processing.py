import cv2
import copy
import math
import numpy as np
from collections import deque
import streamlit as st
import os


def post_processing(mask_image_path):
    image_paths = []
    
    # 이미지 파일 경로 리스트 가져오기
    image_paths = sorted([os.path.join(mask_image_path, filename) for filename in os.listdir(mask_image_path)])
    
    if not image_paths:  # 이미지 디렉토리에 이미지가 없을 경우
        print("No images found in the directory.")
        return None, None
    

    # 첫 번째 이미지와 마지막 이미지의 파일 경로 가져오기
    first_image_path = image_paths[0]
    last_image_path = image_paths[-1]

    # 이미지 불러오기
    first_image = cv2.imread(first_image_path)
    last_image = cv2.imread(last_image_path)

    # AfterTreatment 함수 호출
    first_result, last_result = AfterTreatment(first_image, last_image)
    st.text(first_result)
    st.text(last_result)
    return first_result, last_result

def AfterTreatment(image_left, image_right):
    size = (image_left.shape,image_right.shape)
    image_left = cv2.resize(image_left, (size[0][1]//4, size[0][0]//4))
    image_right= cv2.resize(image_right,(size[1][1]//4, size[1][0]//4))
    
    t_left  = np.transpose(image_left,  (2, 0, 1))
    t_right = np.transpose(image_right, (2, 0, 1))
    
    left_eyelid = t_left[0]
    left_iris   = t_left[1]
    left_pupil = t_left[2]
    right_eyelid = t_right[0]
    right_iris   = t_right[1]
    right_pupil = t_right[2]
    
    left_eyelid  = left_eyelid  +left_iris  +left_pupil
    right_eyelid = right_eyelid +right_iris +right_pupil
    left_eyelid  = FindClass(left_eyelid)
    right_eyelid = FindClass(right_eyelid)
    
    left_iris = left_iris  +left_pupil
    right_iris= right_iris +right_pupil
    left_iris  = FindClass(left_iris)
    right_iris = FindClass(right_iris)
    
    left_edge = EyelidMask2Edge(left_eyelid)
    right_edge= EyelidMask2Edge(right_eyelid)
    
    left_center = PupilMask2ECenter(left_iris)
    right_center= PupilMask2ECenter(right_iris)
    
    left_edge, right_edge, left_center, right_center =\
        Stabilize(left_edge, right_edge, left_center, right_center)
    
    left_xy = coordinate(left_edge, left_center)
    right_xy= coordinate(right_edge, right_center)
    
    return left_xy, right_xy

        
        
def BFS_FindClass():
    M,N = size
    
    while stack:
        i,j = stack.popleft()
        
        for dx,dy in direct:
            x = i +dx
            y = j +dy
            if 0<=x and x<N and 0<=y and y<M:
                if img_[y][x] == 1 and note[-1][y][x] == 0:
                    img_[y][x] = 0
                    note[-1][y][x] = 1
                    len_[0] += 1
                    stack.append([x,y])
            

def FindClass(img):
    img = np.clip(img, 0, 1)
    
    global img_, size, stack, direct, len_, note, n_note
    img_ = np.copy(img)
    size = img.shape
    stack = deque([])
    direct = [[1,0], [-1,0], [0,1], [0,-1]]
    note = []
    n_note = []

    M,N = size
    idx = 0
    for j in range(M):
        for i in range(N):
            if img_[j][i] ==1 :
                stack.append([i,j])
                note.append([[0 for n in range(N)] for m in range(M)])
                len_ = [1]
                img_[j][i] = 0
                note[-1][j][i] = 1
                BFS_FindClass()
                n_note.append([idx, len_[0]])
                idx +=1
                
    n_note.sort(key = lambda x: -x[1])
    image_pupil = note[n_note[0][0]]
    image_pupil = np.array(image_pupil)

    return image_pupil


def EyelidMask2Edge(image_eyelid):
    M,N = image_eyelid.shape
    sum_list = []
    block = np.array([i for i in range(M)])
    for n in range(N):
        if image_eyelid[:,n].sum() == 0:
            sum_ = 1
        else:
            sum_ = image_eyelid[:,n].sum()

        ans = image_eyelid[:,n] *block
        ans = round(ans.sum() /sum_)
        sum_list.append(ans)
    
    point = []
    for x,y in enumerate(sum_list):
        point.append([x,y])
    point = [x for x in point if x[1] != 0]
    
    return point[0], point[-1]


def PupilMask2ECenter(image_pupil):
    M,N =image_pupil.shape
    sum_ = image_pupil.sum()

    sum_n = 0
    sum_m = 0

    block_N = np.array([i for i in range(M)])
    block_M = np.array([i for i in range(N)])

    for n in range(N):
        sum_n += image_pupil[:,n] *block_N
    for m in range(M):
        sum_m += image_pupil[m,:] *block_M

    center_n = round(sum_n.sum() /sum_)
    center_m = round(sum_m.sum() /sum_)

    return center_n, center_m


def Stabilize(edge_left, edge_right, center_left, center_right):
    a,b = edge_left[0]
    c,d = edge_left[1]
    x,y = edge_right[0]
    w,z = edge_right[1]

    A = (a-c)*(z-y)
    B = (b-d)*(z-y) + (a-c)*(x-w)
    C = (b-d)*(x-w)
    
    
    T_1 = (-B + (B**2 -4*A*C)**(0.5)) /(2*A)
    T_2 = (-B - (B**2 -4*A*C)**(0.5)) /(2*A)

    th_1 = math.atan(T_1)
    th_2 = math.atan(T_2)
    if abs(th_1) > abs(th_2):
        th = th_2
    else:
        th = th_1

    S = math.sin(th)
    C = math.cos(th)

    T_points = []
    all_list = [edge_left, edge_right, [center_left], [center_right]]
    for graph in all_list:
        pnt = []
        for x,y in graph:
            val = [round(x*C -y*S), round(x*S +y*C)]
            pnt.append(val)
        T_points.append(pnt)
        
    T_points[2] = T_points[2][0]
    T_points[3] = T_points[3][0]
    
    return T_points


def coordinate(edge_eyelid, center_pupil):
    x,y = center_pupil
    x1,y1 = edge_eyelid[0]
    x2,y2 = edge_eyelid[1]
    L = ((x1 -x2)**2 +(y1 -y2)**2)**(0.5) /2

    xm = (x1+x2)/2
    ym = (y1+y2)/2

    x_ = (x-xm)/L
    y_ = (y-ym)/L

    return round(x_,4), round(y_,4)