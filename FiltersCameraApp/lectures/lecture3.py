import streamlit as st
import cv2
import numpy as np
from PIL import Image

def run():
    st.title("📘 المحاضرة 3: العمليات على البكسل (Point Operations)")

    # ===============================
    # 📝 الشرح النظري في Expander قابل للطي
    # ===============================
    with st.expander("📖 الشرح النظري", expanded=False):
        st.markdown("""
        العمليات على البكسل **(Point Operations)** هي تغييرات يتم تطبيقها مباشرة على قيمة كل بكسل دون الاعتماد على الجيران.  
        من أهم هذه العمليات:
        
        1. **تعديل السطوع (Brightness):** بإضافة قيمة لكل البكسلات لزيادة/تقليل الإضاءة.  
        2. **تعديل التباين (Contrast):** بتوسيع أو تقليص مدى القيم لزيادة وضوح التفاصيل.  
        3. **الصورة السالبة (Negative):** عكس الألوان بحيث يصبح البكسل `255 - value`.  
        4. **العتبة (Thresholding):** تحويل الصورة إلى ثنائية (أبيض/أسود) حسب قيمة عتبة محددة أو باستخدام خوارزمية أوتسو (Otsu).
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
    # 🎚️ التحكم بالسطوع والتباين
    # ===============================
    st.subheader("🎛️ تعديل السطوع والتباين")
    brightness = st.slider("السطوع (Brightness)", -100, 100, 0)
    contrast = st.slider("التباين (Contrast)", 0.5, 3.0, 1.0, 0.1)

    processed_image = cv2.convertScaleAbs(original_image, alpha=contrast, beta=brightness)

    # ===============================
    # 🔄 الصورة السالبة
    # ===============================
    if st.button("🌓 تطبيق الصورة السالبة (Negative)"):
        processed_image = cv2.bitwise_not(processed_image)

    # ===============================
    # ⚪ Thresholding
    # ===============================
    st.subheader("⚪ العتبة (Thresholding)")
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    method = st.radio("اختر الطريقة:", ["بسيط", "Otsu"])

    if method == "بسيط":
        thresh_val = st.slider("قيمة العتبة", 0, 255, 127)
        _, thresh_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        _, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh_img_rgb = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB)

    # ===============================
    # 🔀 شريط التمرير (قبل/بعد)
    # ===============================
    st.subheader("📊 عرض قبل/بعد")
    ratio = st.slider("نسبة ظهور الصورة بعد المعالجة", 0, 100, 50)
    width, height = original_image.shape[1], original_image.shape[0]
    overlay_width = int(width * ratio / 100)

    combined = np.zeros_like(original_image)
    combined[:, :overlay_width] = processed_image[:, :overlay_width]
    combined[:, overlay_width:] = original_image[:, overlay_width:]

    st.image(combined, use_column_width=True, caption="قبل/بعد - تعديل السطوع/التباين/السالب")

    # ===============================
    # 🖤 عرض Thresholding
    # ===============================
    st.subheader("⚪ نتيجة Thresholding")
    st.image(thresh_img_rgb, use_column_width=True)

    # ===============================
    # 💾 حفظ الصورة
    # ===============================
    if st.button("💾 حفظ الصورة"):
        output = Image.fromarray(processed_image)
        output.save("lecture3_output.png")
        st.success("✅ تم حفظ الصورة باسم lecture3_output.png")
