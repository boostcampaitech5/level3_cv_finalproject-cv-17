import socket

from Mediapipeline import Mediapipeline
from .gesture.getGesture_ver2 import get_gesture


def main():
    # 여기부터는 소켓 통신에 이용되는 곳입니다.

    host, port = '0.0.0.0', 30008

    server_sock = socket.socket(socket.AF_INET)
    server_sock.bind((host, port))
    server_sock.listen()
    print("최초 접속 기다리는 중...")

    try:
        while True:
            client_sock, addr = server_sock.accept()
            print('연결 완료 : ', addr)
            print("\n----------------------------------------\n")

            print("client에서 보내는 데이터 기다리는 중...")

            # 이 부분에서 client에서 보내는 이미지를 버퍼 단위로 tst.jpg 이미지에 씁니다.
            filename = open('tst.jpg', 'wb')
            while True:
                buf = client_sock.recv(1024)
                if not buf:
                    break
                filename.write(buf)
            filename.close()

            print("이미지 수신 완료")

            # mediapipe를 사용하여 output 생성
            try:
                mesh_points = Mediapipeline('tst.jpg')
                gesture = get_gesture(mesh_points)
                data = gesture  # int형 자료형으로 최종 제스처 결과 출력
            except:
                data = 0

            if data == 5: # 'look_left'
                print("제스처 결과: 앞으로 가기 전송\n") # 'forward'
            elif data == 6: # 'look_right'
                print("제스처 결과: 뒤로 가기 전송\n") # 'backward'
            elif data == 1: # 'look_down'
                print("제스처 결과: 스크롤 다운 전송\n") # 'scroll_down'
            elif data == 2: # 'look_up'
                print("제스처 결과: 스크롤 업 전송\n") # 'scroll_up'
            elif data == 7: # 'look_down_cont'
                print("제스처 결과: 자동 스크롤 다운 전송\n") # 'auto_scroll_down'
            elif data == 8: # 'look_up_cont'
                print("제스처 결과: 자동 스크롤 업 전송\n") # 'auto_scroll_up'
            elif data == 9: # 'blink'
                print("제스처 결과: 자동 스크롤 중지 전송\n") # 'stop_auto_scroll'
            elif data == 10: # 'blink_cont'
                print("제스처 결과: 프로그램 종료 전송\n") # 'shut_down'
            else:
                print("서버에서 안드로이드로 " + str(data) + " 전송\n")

            client_sock.sendall(data.to_bytes(1, byteorder='little'))
            print("----------------------------------------\n\n")

            client_sock.close()
    except:
        pass

    server_sock.close()


main()