import streamlit as st
from PIL import Image
from mediapipe_streamlit import visualize_eyes

# # SETTING PAGE CONFIG TO WIDE MODE
# st.set_page_config(layout="wide")

def main():
    st.title("Eye Visualization")

    # 이미지 업로드
    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        # 업로드한 이미지를 PIL Image로 변환
        image = Image.open(image_file)
        
        # Mediapipe를 사용하여 눈 시각화
        visualize_eyes(image)
        
if __name__ == "__main__":
    main()
