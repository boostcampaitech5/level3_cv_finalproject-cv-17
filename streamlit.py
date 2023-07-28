import os
import cv2
import time
import numpy as np
import base64
import mediapipe as mp
import streamlit as st
from PIL import Image
from tqdm.auto import tqdm
from io import BytesIO
import socket

from Mediapipeline import Mediapipeline
# from .gesture.getGesture_ver2 import get_gesture
from getGesture import get_gesture


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


def main():
    # 여기부터는 소켓 통신에 이용되는 곳입니다.
    st.title("Android - Streamlit Connection Test")

    # host = '0.0.0.0' # '101.101.218.26'
    # ports = [30008, 30009]
    # for port in ports:
    #     try:
    #         server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #         server_sock.bind((host, port))
    #         server_sock.listen()
    #         st.write(f"port {port}")
    #         st.write("최초 접속 기다리는 중...")
    #         break
    #     except OSError as e:
    #         st.write(f"포트 {port}를 사용할 수 없습니다. 다른 포트를 시도합니다.")

    host, port = '0.0.0.0', 30008

    server_sock = socket.socket(socket.AF_INET)
    server_sock.bind((host, port))
    server_sock.listen()
    st.write("최초 접속 기다리는 중...")

    while True:
        client_sock, addr = server_sock.accept()
        client_sock.settimeout(1)
        st.write('연결 완료 : ', addr)
        st.write("\n----------------------------------------\n")

        st.write("client에서 보내는 데이터 기다리는 중...")

        # 이 부분에서 client에서 보내는 이미지를 버퍼 단위로 tst.jpg 이미지에 씁니다.
        try:
            filename = open('tst.jpg', 'wb')
            while True:
                buf = client_sock.recv(1024)
                if not buf:
                    break
                filename.write(buf)
            filename.close()
        except:
            st.write("receive timeout")

        # buf = b''
        # try:
        #     filename = open('tst.jpg', 'wb')
        #     step = 1024
        #     while True:
        #         data = client_sock.recv(step)
        #         buf += data
        #         if len(buf) == 1024:
        #             break
        #         elif len(buf) < 1024:
        #             step = 1024 - len(buf)
        #         filename.write(buf)
        # except Exception as e:
        #     print(e)
        # print(buf[:1024])
        # filename.close()

        # 이미지 받기
        image_path = './tst.jpg'
        try:
            # point = gaze_pointer(image_path)

            # mediapipe를 사용하여 output 생성
            mesh_points = Mediapipeline(image_path)

            # 동공 위치 추적 & 최종 제스처 생성
            # scroll_up / scroll_down / forward / backward
            # auto_scroll_up / auto_scroll_down / stop_auto_scroll / shut_down
            gesture = get_gesture(mesh_points)

            data = gesture  # int형 자료형으로 최종 제스처 결과 출력 # 
        except Exception as e:
            print("오류", e)
            data = 0 # 이미지를 받아오지 못하면 0으로 출력

        client_sock.sendall(data.to_bytes(1, byteorder='little'))
        st.write("서버에서 안드로이드로 " + str(data) + " 전송\n")
        st.write("----------------------------------------\n\n")

        # client_sock.shutdown(socket.SHUT_RDWR)
        client_sock.close()

    server_sock.close()
    st.write("server socket close\n\n")

main()