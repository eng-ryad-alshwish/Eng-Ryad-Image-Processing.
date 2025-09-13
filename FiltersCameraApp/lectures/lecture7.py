import streamlit as st
import cv2
import numpy as np
from PIL import Image

def run():
    st.title("📘 المحاضرة 7: العمليات المورفولوجية (Morphological Ops)")

    # ===============================
    # 📝 الشرح النظري في Expander
    # ===============================
    with st.expander("📖 الشرح النظري", expanded=False):
        st.markdown("""
        العمليات المورفولوجية تُستخدم لمعالجة الصور الثنائية (Binary).  
        - **Erosion (التآكل):** يقلل من حجم الكائنات البيضاء.  
        - **Dilation (التوسيع):** يوسع الكائنات البيضاء.  
        - **Opening (الفتح):** Erosion ثم Dilation لإزالة الضوضاء الصغيرة.  
        - **Closing (الإغلاق):** Dilation ثم Erosion لسد الفجوات الصغيرة.
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
    # 🔄 تحويل الصورة إلى Binary
    # ===============================
    gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    st.subheader("🎛️ اختيار العملية المورفولوجية")
    morph_type = st.selectbox("اختر العملية", ["Erosion", "Dilation", "Opening", "Closing"])
    ksize = st.slider("حجم العنصر البنائي (Kernel size)", 1, 15, 3)
    kernel = np.ones((ksize, ksize), np.uint8)

    if morph_type == "Erosion":
        morphed = cv2.erode(binary, kernel, iterations=1)
    elif morph_type == "Dilation":
        morphed = cv2.dilate(binary, kernel, iterations=1)
    elif morph_type == "Opening":
        morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    else:  # Closing
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    processed_image = cv2.cvtColor(morphed, cv2.COLOR_GRAY2RGB)

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

    st.image(combined, use_column_width=True, caption="قبل/بعد - العمليات المورفولوجية")

    # ===============================
    # 💾 حفظ الصورة
    # ===============================
    if st.button("💾 حفظ الصورة"):
        output = Image.fromarray(processed_image)
        output.save("lecture7_output.png")
        st.success("✅ تم حفظ الصورة باسم lecture7_output.png")
