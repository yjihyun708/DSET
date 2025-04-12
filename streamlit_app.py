import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# 모델 로드
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("keras_model.h5")
    return model

model = load_model()

# 클래스 이름 설정 (Teachable Machine에서 사용한 클래스 이름으로 바꿔주세요!)
class_names = ['왼쪽으로!', '오른쪽으로!', '위로!', '아래로!']

# 페이지 설정
st.set_page_config(page_title="🎄 크리스마스 미로", layout="centered")
st.title("🎄 크리스마스 미로 - AI 길 찾기")
st.markdown("📸 웹캠이나 이미지 업로드로 AI가 미로에서 갈 방향을 알려줘요!")

# 이미지 입력 받기
img_input = st.camera_input("사진을 찍어주세요!") or st.file_uploader("또는 이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if img_input:
    # 이미지 열기
    image = Image.open(img_input)
    st.image(image, caption="입력된 이미지", use_column_width=True)

    # 전처리 (Teachable Machine 기준: 224x224, 정규화)
    image = image.resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    # 예측
    prediction = model.predict(img_array)[0]
    pred_index = np.argmax(prediction)
    pred_label = class_names[pred_index]
    pred_conf = prediction[pred_index]

    # 결과 출력
    st.success(f"🧠 AI 예측 결과: **{pred_label}** ({pred_conf:.2%} 확신도)")

    st.markdown("🧩 AI가 제시하는 방향으로 이동해보세요!")

    # 방향 안내 메시지
    if pred_label == '왼쪽으로!':
        st.info("🚪 왼쪽 길을 따라가세요!")
    elif pred_label == '오른쪽으로!':
        st.info("🚪 오른쪽 길을 따라가세요!")
    elif pred_label == '위로!':
        st.info("🚪 위쪽으로 전진하세요!")
    elif pred_label == '아래로!':
        st.info("🚪 아래쪽으로 이동하세요!")

# Footer
st.markdown("---")
st.caption("🛠 만든 사람: [TeachableVerse 프로젝트 참고]")
