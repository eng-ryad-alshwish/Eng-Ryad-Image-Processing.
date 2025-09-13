import streamlit as st
import cv2
import numpy as np
from PIL import Image

def run():
    st.title("📘 المحاضرة 4: الفلاتر والالتفاف (Filtering & Convolution)")

    # ===============================
    # 📝 الشرح النظري في Expander
    # ===============================
    with st.expander("📖 الشرح النظري", expanded=False):
        st.markdown("""
        الفلاتر (Filters) والالتفاف (Convolution) هي عمليات تُطبق على الصورة لتغيير مظهرها أو استخراج تفاصيل منها.  
        - **Kernel / Mask:** مصفوفة صغيرة تُطبق على البكسلات المجاورة لإنتاج تأثير محدد.  
        - **الفلاتر الأساسية:**  
          - **Blur (تمويه):** لتخفيف التفاصيل وتقليل الضوضاء.  
          - **Sharpen (توضيح):** لإبراز الحواف والتفاصيل.  
          - **Emboss (تجسيم):** لإظهار تأثير ثلاثي الأبعاد على الحواف.
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
    # 🎛️ اختيار نوع الفلتر
    # ===============================
    st.subheader("🎛️ اختيار الفلتر")
    filter_type = st.selectbox("اختر نوع الفلتر:", ["None", "Sharpen", "Blur (Gaussian)", "Blur (Median)", "Edge Detection", "Emboss"])

    # التحكم بحجم Kernel للتمويه
    kernel_size = 3
    if filter_type in ["Blur (Gaussian)", "Blur (Median)"]:
        kernel_size = st.slider("حجم Kernel (Gaussian/Median)", 3, 15, 5, 2)

    # ===============================
    # تطبيق الفلتر
    # ===============================
    if filter_type == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        processed_image = cv2.filter2D(original_image, -1, kernel)
    elif filter_type == "Blur (Gaussian)":
        processed_image = cv2.GaussianBlur(original_image, (kernel_size, kernel_size), 0)
    elif filter_type == "Blur (Median)":
        processed_image = cv2.medianBlur(original_image, kernel_size)
    elif filter_type == "Edge Detection":
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    elif filter_type == "Emboss":
        kernel = np.array([[-2,-1,0],
                           [-1,1,1],
                           [0,1,2]])
        processed_image = cv2.filter2D(original_image, -1, kernel)

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

    st.image(combined, use_column_width=True, caption="قبل/بعد - الفلتر المختار")

    # ===============================
    # 💾 حفظ الصورة
    # ===============================
    if st.button("💾 حفظ الصورة"):
        output = Image.fromarray(processed_image)
        output.save("lecture4_output.png")
        st.success("✅ تم حفظ الصورة باسم lecture4_output.png")
