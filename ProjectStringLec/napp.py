import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from PIL import Image
import time
import os
from io import BytesIO

st.set_page_config(
    page_title="🎥 تطبيق الفلاتر الحي",
    page_icon="🎥",
    layout="wide"
)

# مجلد الحفظ
SAVE_DIR = "saved_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# تعريف الفلاتر
def apply_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

def apply_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_sepia(frame):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    frame = cv2.transform(frame, kernel)
    return np.clip(frame, 0, 255).astype(np.uint8)

def apply_invert(frame):
    return cv2.bitwise_not(frame)

def apply_sketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray
    blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_emboss(frame):
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
    return cv2.filter2D(frame, -1, kernel)

# خيارات الفلاتر
filter_options = {
    "بدون فلتر": None,
    "تدرج الرمادي": apply_grayscale,
    "تمويه Gaussian": apply_gaussian_blur,
    "كشف الحواف": apply_edge_detection,
    "لون سيبيا": apply_sepia,
    "عكس الألوان": apply_invert,
    "رسم قلم رصاص": apply_sketch,
    "نقش بارز": apply_emboss
}

# Sidebar
with st.sidebar:
    st.title("إعدادات الفلاتر")
    st.session_state.selected_filter = st.radio("اختر الفلتر:", list(filter_options.keys()))
    if 'capture_next' not in st.session_state:
        st.session_state.capture_next = False
        st.session_state.captured_image = None
    capture_btn = st.button("التقاط صورة 📸")

# Video Processor
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # قراءة الفلتر الحالي من session_state
        selected_filter = st.session_state.selected_filter
        if selected_filter != "بدون فلتر":
            filtered = filter_options[selected_filter](img)
            if len(filtered.shape) == 2:  # grayscale
                filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        else:
            filtered = img

        # التقاط الصورة عند الطلب
        if st.session_state.capture_next:
            st.session_state.captured_image = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
            st.session_state.capture_next = False

        return filtered

# تشغيل WebRTC
ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# التقاط الصورة
if capture_btn and ctx.video_processor:
    st.session_state.capture_next = True

# عرض وحفظ الصورة الملتقطة
if st.session_state.captured_image is not None:
    st.image(st.session_state.captured_image, caption="الصورة الملتقطة", use_column_width=True)
    buf = BytesIO()
    Image.fromarray(st.session_state.captured_image).save(buf, format="JPEG")
    buf.seek(0)
    st.download_button(
        "💾 حفظ الصورة",
        buf,
        file_name=f"captured_{int(time.time())}.jpg",
        mime="image/jpeg"
    )
