import os
import cv2
import time
import numpy as np
import socket
import base64
import mediapipe as mp
import streamlit as st
from PIL import Image
from tqdm.auto import tqdm
from io import BytesIO

from collections import deque
from Mediapipeline import Mediapipeline
from post_process import Coordinate, SetZeropoint, GetGesture

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


def main():
    # 여기부터는 소켓 통신에 이용되는 곳입니다.
    st.title("Android - Streamlit Connection Test")

    host, port = '0.0.0.0', 30008

    server_sock = socket.socket(socket.AF_INET)
    server_sock.bind((host, port))
    server_sock.listen()
    st.write("최초 접속 기다리는 중...")

    zeropoint = [[0,0]] +[1,-1,-1,1]
    record = deque([0]*10)
    

    while True:
        client_sock, addr = server_sock.accept()
        st.write('연결 완료 : ', addr)
        st.write("\n----------------------------------------\n")

        st.write("client에서 보내는 데이터 기다리는 중...")

        # 이 부분에서 client에서 보내는 이미지를 버퍼 단위로 tst.jpg 이미지에 씁니다.
        image_path = 'tst.jpg'
        try:
            with open(image_path, 'wb') as filename:
                while True:
                    buf = client_sock.recv(1024)  # 버퍼 크기 증가
                    if not buf:
                        break
                    filename.write(buf)
            st.write("이미지 수신 완료")
        except Exception as e:
            st.write("이미지 수신 및 저장 중 오류 발생:", e)

            
        try:
            mesh_points = Mediapipeline(image_path)
        except:
            st.write('error : Mediapipeline')
            gesture = 0   
        else:
            st.write('complete : Mediapipe')
            input_coord = Coordinate(mesh_points)
            st.write('input_coor : ', input_coord)
            
            #눈 감았는 지 확인 
            if input_coord[0] != 0 and input_coord[1] != 0 :
                zeropoint = SetZeropoint(zeropoint, input_coord, record)
                st.write('zeropoint : ', zeropoint)
                gesture = GetGesture(zeropoint, input_coord)
                record.append(1)
                record.popleft()
                
            else:
                gesture = 0
                record.append(0)
                record.popleft()
                    

        data = gesture
        if data == 5:
            st.write("제스처 결과: 뒤로가기 전송\n")
        elif data == 6:
            st.write("제스처 결과: 앞으로 가기 전송\n")
        elif data == 1:
            st.write("제스처 결과: 스크롤 다운 전송\n")
        elif data == 2:
            st.write("제스처 결과: 스크롤 업 전송\n")
        else:
            st.write("서버에서 안드로이드로 " + str(data) + " 전송\n")

        client_sock.send(data.to_bytes(1, byteorder='little'))
        st.write("----------------------------------------\n\n")

    server_sock.close()
    st.write("server socket close\n\n")

main()