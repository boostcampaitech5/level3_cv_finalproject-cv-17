### 💻 **EyePhone**

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/a3a149a2-3e26-4345-932d-bb1934776a37)

- 현대인의 스마트폰 사용 시간이 지속적으로 증가함에 따라, 다양한 환경에서도 스마트폰을 편리하게 조작할 수 있는 기능을 만들고자 한다.
- eye-phone은 동공의 위치를 탐지하고, 안구 내 움직임을 파악하여 스마트폰 화면에서 일정한 기능을 수행하도록 돕는 어플리케이션이다.
  - 누워 있거나, 자세가 고정되어 있거나, 거리가 멀어 기기를 손으로 직접 조작하기 힘든 상황에서도 쉽게 스마트폰을 동작시킬 수 있도록 한다.
  - 신체적 어려움으로 손을 사용하기 힘든 사람들도 스마트폰을 사용하는 것이 가능하도록 돕는다.
  - 장시간 스마트폰을 사용함으로써 생기는 손의 피로도를 줄일 수 있도록 한다.

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/f04271da-524a-43f3-9a6e-ebd2bb54a3e1)

## Team Members

| <img src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/71297ab2-cbea-4069-8e2b-52fc1329da3b" width="80"> | <img src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/51d33796-1194-4229-875a-790fc1625483" width="80"> | <img src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/48c2e739-6a82-44b3-978d-c07add64d98b" width="80"> | <img src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/199ea001-e808-4e7e-ae8a-dbe1a082411e" width="80"> | <img src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/e891d773-31b0-4163-be1a-40cc6f9bdc42" width="80"> | <img src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/cf42f192-4de7-4232-910f-e115a9a7fe8e" width="80"> |
| :------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                     [강대호](https://github.com/dh3211)                                                      |                                                   [강정우](https://github.com/kangjjjjjww)                                                   |                                                  [박혜나](https://github.com/hyenagatha02)                                                   |                                                    [원유석](https://github.com/bigaguero)                                                    |                                                     [서지훈](https://github.com/Mugamta)                                                     |                                                   [정대훈](https://github.com/daehun1102)                                                    |
|                                                         모델 테스트<br>데이터 후처리                                                         |                                               모델 테스트<br>mediapipe 구현<br> streamlit 구현                                               |                                                데이터 서치<br>데이터 후처리<br>streamlit 구현                                                |                                                            데이터 서치<br>앱 구축                                                            |                                                            모델 테스트<br>앱 구축                                                            |                                                        모델 테스트<br>mediapipe 구현                                                         |

## Project Timeline

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/3000350e-e8df-4620-8f98-0ae34b4d9631)

## Model Architecture

### [reference](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/d500613f-9b5d-4f2c-baaf-e7d623cc4882)

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/50596e58-a706-4370-9b43-4a6f38a1c3ca)

## Service Architecture

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/22c9a8a1-dc01-4125-963a-9f18a536e85c)

- 휴대폰의 전면 카메라로 사용자의 얼굴 이미지를 프레임 단위로 찍어 서버로 송신
- 서버에서 Mediapipe 모델을 통해 전송 받은 이미지에 대한 face landmark를 검출
- landmark를 통해 눈과 동공의 좌표를 파악한 뒤 후처리 알고리즘을 통해 제스처 명령어 생성
- 제스처 코드를 사용자의 휴대폰에 전송하여 제스처에 해당하는 화면 동작 기능 수행

## Function

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/0b5712a8-fb0d-4e66-8d37-00065d85af39)

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/680c295a-6602-41d4-8725-9f4e0aafb8b6)

## App

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-17/assets/70469008/e0ba471d-fde2-4857-ba07-2ba3ee07e20f)

## Result

### Effect

- 신체적 어려움, 불편한 자세 등의 원인으로 화면을 손으로 직접 조작하기 어려운 사람들도 스마트폰을 조작할 수 있다.
- 스마트폰만 있다면 그 외 다른 장치가 없어도 누구나 해당 어플리케이션을 사용할 수 있다.
- 일반적으로 Eye tracking 기능을 지원하는 타 서비스는 보조기기를 착용해 비용이 많이 들지만 본 프로젝트에선 어플리케이션만으로 동작한다.

### Future works

- 웹 페이지가 아닌 모든 앱의 백그라운드에서 작동하여 다양한 앱에서 활용할 수 있게 한다.
- 동공의 위치 기준을 다양화 하여 더 많은 제스처를 활용할 수 있다.
- 동공의 움직임을 스마트폰 화면에 대응하여 마우스 포인터처럼 움직이고 클릭할 수 있도록 개발한다.
- 서버와 통신하는 방식이 아닌 앱 내에서 모델이 동작하도록 앱 개발한다.
