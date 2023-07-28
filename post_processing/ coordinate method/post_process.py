#After_treatment.py 

import math
import streamlit as st


def Coordinate(mesh_points):
    # 홍채 좌표 리스트
    LEFT_IRIS  = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    # 눈꺼풀 좌표 리스트
    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ] #362, 263// 386 374
    RIGHT_EYE=[  33,   7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 ] #133,  33// 159 145

    L_srt = mesh_points[362][:]
    R_srt = mesh_points[133][:]
    L_end = mesh_points[263][:]
    R_end = mesh_points[33][:]
    L_cen = mesh_points[473][:]
    R_cen = mesh_points[468][:]

    L_srt, L_end, L_cen, R_srt, R_end, R_cen = Parallelize(L_srt, L_end, L_cen, R_srt, R_end, R_cen)

    L_closed, R_closed = CheckClosed(mesh_points)
    if L_closed: L_cor = 0
    else       : L_cor = Coordinate_part(L_srt, L_end, L_cen) 
    if R_closed: R_cor = 0
    else       : R_cor = Coordinate_part(R_srt, R_end, R_cen)
    
    return L_cor, R_cor


def Parallelize(L_srt, L_end, L_cen, R_srt, R_end, R_cen):
    try:
        srt_slope = (L_end[1] - R_end[1]) / (L_end[0] - R_end[0])
    except ZeroDivisionError:
        srt_slope = 0
    try:
        end_slope = (L_end[1] - R_end[1]) / (L_end[0] - R_end[0])
    except ZeroDivisionError:
        end_slope = 0
    slope = (srt_slope +end_slope)/2
    
    theta = math.atan(slope)
    
    S = math.sin(-theta)
    C = math.cos(-theta)

    T_points = []
    all_list = [L_srt, L_end, L_cen, R_srt, R_end, R_cen]
    for x,y in all_list:
        val = [x*C -y*S, x*S +y*C]
        T_points.append(val)
        
    return T_points


def Coordinate_part(srt, end, cen):
    x,y = cen
    x1,y1 = srt
    x2,y2 = end
    L = length(srt, end)

    xm = (x1+x2)/2
    ym = (y1+y2)/2

    x_ = (x-xm)/L
    y_ = (y-ym)/L

    return round(x_,4), round(y_,4)


def length(point1, point2):
    x1,y1 = point1
    x2,y2 = point2
    L = ((x1 -x2)**2 +(y1 -y2)**2)**(0.5) /2
    
    return L


def CheckClosed(mesh_points):
    L_eye_height  = length(mesh_points[386], mesh_points[374])
    R_eye_height  = length(mesh_points[159], mesh_points[145])
    L_iris_height = length(mesh_points[475], mesh_points[477])
    R_iris_height = length(mesh_points[470], mesh_points[472])
    
    
    if L_eye_height*5 > L_iris_height:
        L_closed = 0
    else:
        L_closed = 1
        
    if R_eye_height*5 > R_iris_height:
        R_closed = 0
    else:
        R_closed = 1
        
    return L_closed, R_closed

#-------------------------------------------------

def SetZeropoint(zeropoint, input_coord, record):
    #zeropoint = [[center], [left], [top], [right], [bottom]]
    avg_coord = AvgCoord(input_coord)
    avg_x, avg_y = avg_coord

    #최고값 갱신
    if avg_x < zeropoint[1]:
        zeropoint[1] = avg_x
    if avg_y > zeropoint[2]:
        zeropoint[2] = avg_y
    if avg_x > zeropoint[3]:
        zeropoint[3] = avg_x
    if avg_y < zeropoint[4]:
        zeropoint[4] = avg_y
    
    if record[-1] == 0 and record[-2] == 0:
        zeropoint[0] = avg_coord
            
    return zeropoint

    
def AvgCoord(input_coord):
    L_cor, R_cor = input_coord
    x = (L_cor[0] + R_cor[0])/2
    y = (L_cor[1] + R_cor[1])/2
    avg_coord = [x ,y]
    
    return avg_coord

#-------------------------------------------------

def Record():
    return 0



#-------------------------------------------------

def GetGesture(zeropoint, input_coord):
    #zeropoint = [[center], [left], [top], [right], [bottom]]
    avg_coord = AvgCoord(input_coord)
    st.write("avg_coord : ", avg_coord)
    st.write(avg_coord[0], avg_coord[1])
    
    len_cen_row    =  abs(avg_coord[0] -zeropoint[0][0])
    len_cen_col    =  abs(avg_coord[1] -zeropoint[0][1])
    len_left   =  abs(avg_coord[0] -zeropoint[1])
    len_top    =  abs(avg_coord[1] -zeropoint[2])
    len_right  =  abs(avg_coord[0] -zeropoint[3])
    len_bottom =  abs(avg_coord[1] -zeropoint[4])


    cor = [0,0]

    if len_left  *1.3 < len_cen_row:
        cor[0] =-1
    if len_right *1.3 < len_cen_col:
        cor[0] = 1
    if len_top   *1.3 < len_cen_row:
        cor[1] = 1
    if len_bottom*1.3 < len_cen_col:
        cor[1] =-1
    st.write("cor : ", cor)
    
    gesture = 0
    if cor == [-1, 0]: # look_left
        gesture = 5
    if cor == [ 1, 0]: # look_right
        gesture = 6
    if cor == [ 0, 1]: # look_up
        gesture = 2
    if cor == [ 0,-1]: # look_down
        gesture = 1

    
    return gesture
        
