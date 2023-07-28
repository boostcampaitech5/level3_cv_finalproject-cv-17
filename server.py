import socket

from Mediapipeline import Mediapipeline
# from .gesture.getGesture_ver2 import get_gesture
from getGesture import get_gesture


def main():
    # 여기부터는 소켓 통신에 이용되는 곳입니다.

    host, port = '0.0.0.0', 30008

    server_sock = socket.socket(socket.AF_INET)
    server_sock.bind((host, port))
    server_sock.listen()
    print("최초 접속 기다리는 중...")

    
    while True:
        client_sock, addr = server_sock.accept()
        client_sock.settimeout(1) # maybe 700 ms on app will be okay
        print('연결 완료 : ', addr)
        print("\n----------------------------------------\n")

        print("client에서 보내는 데이터 기다리는 중...")

        # 이 부분에서 client에서 보내는 이미지를 버퍼 단위로 tst.jpg 이미지에 씁니다.
        filename = open('tst.jpg', 'wb')
        try:
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

                if data == 0:
                    print("제스처 결과: 작동 없음 (정면 응시)\n")
            except:
                data = 0
                print("mediapipeline 오류 발생(얼굴 인식 불가?)")

            if data == 1: # 'look_down'
                print("제스처 결과: 스크롤 다운\n") # 'scroll_down'
            elif data == 2: # 'look_up'
                print("제스처 결과: 스크롤 업\n") # 'scroll_up'
            elif data == 5: # 'look_right'
                print("제스처 결과: 앞으로 가기\n") # 'forward'
            elif data == 6: # 'look_left
                print("제스처 결과: 뒤로 가기\n") # 'backward'
            elif data == 7: # 'look_down_cont'
                print("제스처 결과: 자동 스크롤 다운\n") # 'auto_scroll_down'
            elif data == 8: # 'look_up_cont'
                print("제스처 결과: 자동 스크롤 업\n") # 'auto_scroll_up'
            elif data == 9: # 'blink'
                print("제스처 결과: 자동 스크롤 종료\n") # 'stop_auto_scroll'
            elif data == 10: # 'blink_cont'
                print("제스처 결과: 프로그램 종료\n") # 'shut_down'
                client_sock.send(data.to_bytes(1, byteorder='little'))
                break
            else:
                print("서버에서 안드로이드로 " + str(data) + " 전송\n")

            client_sock.sendall(data.to_bytes(1, byteorder='little'))
            
        except:
            print("receive timeout")
            filename.close()

        print("----------------------------------------\n\n")
        client_sock.close()
    

    server_sock.close()
    print("server socket close\n\n")

main()