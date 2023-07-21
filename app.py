import streamlit as st
import os
import cv2
import time
import numpy as np
import socket
import base64
from PIL import Image

from tqdm.auto import tqdm

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


def receive_all(sock, count):
    buf = b''
    while True:
        newbuf = sock.recv(1024)
        if not newbuf: 
            break
        buf += newbuf
    return buf


def main():
    st.title("Android - Streamlit Connection Test")

    host, port = '125.135.3.93', 30008

    server_sock = socket.socket(socket.AF_INET)
    server_sock.bind((host, port))
    server_sock.listen(1)
    st.write("최초 접속 기다리는 중...")
    client_sock, addr = server_sock.accept()
    st.write('연결 완료 : ', addr)
    st.write("\n----------------------------------------\n")
    while True:
        st.write("client에서 보내는 데이터 기다리는 중...")
        filename = open('tst.jpg', 'wb')
        while True:
            buf = client_sock.recv(1024)
            if not buf:
                break
            filename.write(buf)
        filename.close()

        st.write("안드로이드에서 보낸 데이터: ")
        cv2.imread("tst.jpg")
        cv2.imshow('color_img', "tst.jpg")
        cv2.waitKey()
        st.write("\n\n\n\n")

        data2 = int(input("보낼 값(0이면 종료) : "))
        if data2 == 0:
            break

        client_sock.send(data2.to_bytes(1, byteorder='little'))
        st.write("서버에서 안드로이드로 " + str(data2) + " 전송\n")
        st.write("----------------------------------------\n\n")

    client_sock.close()
    st.write("client socket close")
    server_sock.close()
    st.write("server socket close\n\n")


main()