# 프로젝트 소개

<aside>
💻 **EyePhone**

</aside>

- 현대인의 스마트폰 사용 시간이 지속적으로 증가하는 주세에 있어, 스마트폰을 보다 편하게 조작하기 위한 기능을 개발하고자 한다. 이를 통해 몸이 불편하거나, 자세가 고정된 상황에서도 스마트폰을 사용이 가능하도록 한다.
- 스마트폰 카메라로 인에서 안구의 위치를 탐지하고, 그 중 동공의 위치를 통해 시선의 움직임을 파악한다. 해당 움직임을 기준으로 설정한 화면 동작 기능을 수행한다.

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/126538540/65e82ce6-45e0-459c-84b0-95e6cfba6e6f)


# 팀 소개

| 강대호 | 모델 테스트, 데이터 후처리 |
| --- | --- |
| 강정우 | 모델 테스트, mediapipe, streamlit 구현 |
| 박혜나 | 데이터 서치, 데이터 후처리, streamlit 구현 |
| 원유석 | 데이터 서치, 앱 구축 |
| 서지훈 | 모델 테스트, 앱 구축 |
| 정대훈 | 모델 테스트, mediapipe 구현 |

# 프로젝트 타임라인

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/126538540/3ff0f498-2523-44e3-82c2-ec3bb84d1f37)


# 모델

### **Mediapipe**

 

  **모델 소개**

- Mediapipe는 Google에서 개발한 오픈 소스 프레임 워크로, 머신 러닝 기반의 비디오 및 오디오 처리를 위한 다양한 라이브러리를 제공한다
- Object detection, segmentation, pose detection, face detection 등 다양한 vision task solution을 제공한다.
- 본 프로젝트에서 Mediapipe의 **face landmark detection** 모델을 눈과 홍채를 추적하는데 사용했습니다.
- Android, Web, Python 환경에서도 동작이 가능 합니다.

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/126538540/fb2c632c-c9e2-4f6d-b6bb-883b0595755c)
![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/126538540/dc2535af-232b-4454-869a-8be421262064)

    **모델 설명**

- Face landmark 검출 – Face detection model, **Face mesh model**, blendshape prediction model
- 전체 이미지에서 얼굴 위치를 계산하는 검출기와 검출된 얼굴 위치에서 3D표면을 예측하는 3D 얼굴 랜드마크 모델이 함께 작동합니다.
- FaceMesh는 실시간으로 468개의 3D 얼굴 랜드마크를 추정합니다.
- Attention Mesh 구조를 통해 특징이 가장 작은 부분을 집중적으로 검출하여 랜드마크 생성합니다.

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/126538540/87329448-4651-4e88-81a4-ea74260d068e)


# 서비스 아키텍처

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/126538540/c47ce51b-9eb0-4fb3-b8e8-126b5ce193be)


- 휴대폰의 전면 카메라로 사용자의 얼굴을 찍어 서버로 보낸다.
- 서버에서 Mediapipe 모델을 통해 face landmark를 검출한다.
- landmark를 통해 눈과 동공의 좌표를 파악하고 후처리 알고리즘을 통해 제스처를 검출한다.
- 제스처 코드를 사용자의 휴대폰에 전송하여 제스처를 실행한다.

# 앱

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/126538540/95436c30-53c6-4728-a820-186f10704110)
앱 초기 화면

url : 접속할 페이지 주소를 입력한다.

ip address : 사용자의 얼굴 사진을 받고 후처리 진행할 서버의 주소를 입력한다.

 port number : 서버에 접속하기 위한 포트를 입력한다.

최초 실행 딜레이 : 서버와 최초 접속하기 위해 필요한 시간 (1000ms 추천)

동작 주기 : 사용자의 얼굴 사진을 서버에 전송하는 주기 (600 ~ 800ms 추천)

| 작동 조건 ​ | 실행 동작 |
| --- | --- |
| 왼쪽 응시​ | backward​ |
| 오른쪽 응시​ | forward​ |
| 위쪽 응시​ | scoll_up​ |
| 아래쪽 응시​ | scroll_down​ |
| 위쪽 3프레임 이상 응시​ | auto_scroll_up​ |
| 아래쪽 3프레임 이상 응시​ | auto_scroll_down​ |
| 눈 3 프레임 이상 감기​ | stop_auto_scroll​ |
| 눈 5프레임 이상 감기​ | shut_down​ |

# 결과

### **가치**

- 신체적 어려움, 불편한 자세 등의 원인으로 화면을 손으로 직접 조작하기 어려운 사람이 특별  한 기기 없이 스마트폰을 조작할 수 있다.
- 일반적으로 Eye tracking 할 때 보조기기를 착용해 비용이 많이 들지만 본 프로젝트에선 어플리케이션만으로 동작한다.

### 보완 및 확장

- 웹 페이지가 아닌 모든 앱의 백그라운드에서 작동하여 다양한 앱에서 활용할 수 있게 한다.
- 동공의 위치 기준을 다양화 하여 더 많은 제스처를 활용할 수 있다.
- 동공의 움직임을 스마트폰 화면에 대응하여 마우스 포인터처럼 움직이고 클릭할 수 있도록 개발한다.
- 서버와 통신하는 방식이 아닌 앱 내에서 모델이 동작하도록 앱 개발한다.
