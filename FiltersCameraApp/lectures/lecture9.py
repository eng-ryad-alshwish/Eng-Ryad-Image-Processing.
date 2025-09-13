import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def run():
    st.title("📘 المحاضرة 9: المشروع الختامي (Final Project)")

    # ===============================
    # 📝 الشرح النظري
    # ===============================
    with st.expander("📖 الشرح النظري", expanded=False):
        st.markdown("""
        في هذا المشروع النهائي، ستقوم بعمل سلسلة معالجة للصورة (Pipeline):  
        - رفع صورة أو استخدام صورة افتراضية.  
        - تطبيق سلسلة عمليات: تحويل إلى رمادي → تطبيق Blur → كشف الحواف.  
        - عرض النتيجة النهائية قبل وبعد.  
        - حفظ الصورة الناتجة.
        """)

    st.divider()

    # ===============================
    # 📤 رفع الصورة
    # ===============================
    uploaded_file = st.file_uploader("📤 قم برفع صورة", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.full((300, 400, 3), 180, dtype=np.uint8)  # صورة رمادية بسيطة

    original_image = img.copy()
    processed_image = img.copy()

    # ===============================
    # 🎛️ اختيار العمليات
    # ===============================
    st.subheader("🎛️ اختيار سلسلة العمليات (Pipeline)")
    apply_gray = st.checkbox("تحويل إلى رمادي (Grayscale)")
    apply_blur = st.checkbox("تطبيق Gaussian Blur")
    apply_edges = st.checkbox("كشف الحواف (Canny)")

    # إعداد نسخة للعمل عليها
    output = processed_image.copy()

    # ===============================
    # 🛠️ تطبيق العمليات
    # ===============================
    if apply_gray:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    if apply_blur:
        output = cv2.GaussianBlur(output, (7,7), 0)

    if apply_edges:
        gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        output = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    processed_image = output

    # ===============================
    # 🔀 عرض قبل/بعد مع شريط تمرير
    # ===============================
    st.subheader("📊 عرض قبل/بعد")
    ratio = st.slider("نسبة ظهور الصورة بعد العمليات", 0, 100, 50)

    width, height = original_image.shape[1], original_image.shape[0]
    overlay_width = int(width * ratio / 100)
    combined = np.zeros_like(original_image)
    combined[:, :overlay_width] = processed_image[:, :overlay_width]
    combined[:, overlay_width:] = original_image[:, overlay_width:]

    st.image(combined, use_column_width=True, caption="قبل/بعد - سلسلة العمليات")

    # ===============================
    # 💾 حفظ الصورة
    # ===============================
    if st.button("💾 حفظ الصورة"):
        output_pil = Image.fromarray(processed_image)
        output_pil.save("final_project_output.png")
        st.success("✅ تم حفظ الصورة باسم final_project_output.png")
