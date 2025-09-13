import streamlit as st
import cv2
import numpy as np
from PIL import Image

def run():
    st.title("📘 المحاضرة 6: كشف الحواف (Edge Detection)")

    # ===============================
    # 📝 الشرح النظري في Expander
    # ===============================
    with st.expander("📖 الشرح النظري", expanded=False):
        st.markdown("""
        كشف الحواف يساعد على تحديد الانتقالات المفاجئة في الإضاءة داخل الصورة.  
        - **Gradient:** يُظهر التغيرات في قيم البكسل.  
        - **Sobel:** يحدد الحواف الأفقية والعمودية.  
        - **Laplacian:** يجمع بين اتجاهات مختلفة لإظهار الحواف.  
        - **Canny:** أكثر الطرق دقة، يعتمد على Threshold منخفض ومرتفع لتحديد الحواف.
        """)

    st.divider()

    # ===============================
    # 📤 رفع صورة / صورة افتراضية
    # ===============================
    uploaded_file = st.file_uploader("📤 قم برفع صورة", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.full((200, 300, 3), 120, dtype=np.uint8)  # صورة رمادية بسيطة

    original_image = img.copy()
    processed_image = img.copy()

    # ===============================
    # 🎛️ اختيار نوع كشف الحافة
    # ===============================
    st.subheader("🎛️ نوع كشف الحواف")
    edge_type = st.selectbox("اختر الطريقة", ["Sobel", "Laplacian", "Canny"])

    gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)

    if edge_type == "Sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        processed_image = cv2.magnitude(sobelx, sobely)
        processed_image = np.uint8(np.clip(processed_image, 0, 255))
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    elif edge_type == "Laplacian":
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        processed_image = np.uint8(np.clip(laplacian, 0, 255))
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    else:  # Canny
        low_thresh = st.slider("Low Threshold", 0, 255, 50)
        high_thresh = st.slider("High Threshold", 0, 255, 150)
        edges = cv2.Canny(gray, low_thresh, high_thresh)
        processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # ===============================
    # 🔀 شريط التمرير قبل/بعد
    # ===============================
    st.subheader("📊 عرض قبل/بعد")
    ratio = st.slider("نسبة ظهور الصورة بعد الفلتر", 0, 100, 50)
    width, height = original_image.shape[1], original_image.shape[0]
    overlay_width = int(width * ratio / 100)

    combined = np.zeros_like(original_image)
    combined[:, :overlay_width] = processed_image[:, :overlay_width]
    combined[:, overlay_width:] = original_image[:, overlay_width:]

    st.image(combined, use_column_width=True, caption="قبل/بعد - كشف الحواف")

    # ===============================
    # 💾 حفظ الصورة
    # ===============================
    if st.button("💾 حفظ الصورة"):
        output = Image.fromarray(processed_image)
        output.save("lecture6_output.png")
        st.success("✅ تم حفظ الصورة باسم lecture6_output.png")
