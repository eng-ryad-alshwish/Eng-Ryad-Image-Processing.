import streamlit as st
import cv2
import numpy as np
from PIL import Image

def run():
    st.title("📘 المحاضرة 5: إزالة الضوضاء (Denoising)")

    # ===============================
    # 📝 الشرح النظري في Expander
    # ===============================
    with st.expander("📖 الشرح النظري", expanded=False):
        st.markdown("""
        الضوضاء في الصور هي إشارات غير مرغوبة تؤثر على جودة الصورة.  
        - **Salt & Pepper:** نقاط سوداء وبيضاء تظهر بشكل عشوائي.  
        - **Gaussian Noise:** تباين عشوائي ناعم حول قيم البكسل.  
        الفلاتر المختلفة تساعد على إزالة هذه الضوضاء دون فقدان التفاصيل المهمة.  
        - **Median Filter:** ممتاز لإزالة Salt & Pepper.  
        - **Bilateral Filter:** يقلل الضوضاء مع الحفاظ على الحواف.
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
    # 🎛️ إضافة ضوضاء (اختياري)
    # ===============================
    st.subheader("🎚️ إضافة ضوضاء (اختياري)")
    add_noise = st.checkbox("أضف ضوضاء للصورة")
    noise_type = st.selectbox("اختر نوع الضوضاء", ["Salt & Pepper", "Gaussian"])

    if add_noise:
        if noise_type == "Salt & Pepper":
            s_vs_p = 0.5
            amount = 0.04
            noisy = processed_image.copy()
            num_salt = np.ceil(amount * noisy.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in noisy.shape]
            noisy[coords[0], coords[1], :] = 255
            num_pepper = np.ceil(amount * noisy.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in noisy.shape]
            noisy[coords[0], coords[1], :] = 0
            processed_image = noisy
        else:  # Gaussian Noise
            row, col, ch = processed_image.shape
            mean = 0
            sigma = 15
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            processed_image = np.clip(processed_image + gauss, 0, 255).astype(np.uint8)

    # ===============================
    # 🎛️ اختيار الفلتر لإزالة الضوضاء
    # ===============================
    st.subheader("🎛️ إزالة الضوضاء")
    denoise_type = st.selectbox("اختر نوع الفلتر", ["None", "Median Filter", "Bilateral Filter"])

    if denoise_type == "Median Filter":
        ksize = st.slider("حجم Kernel للفلتر Median", 3, 11, 3, 2)
        processed_image = cv2.medianBlur(processed_image, ksize)
    elif denoise_type == "Bilateral Filter":
        d = st.slider("Diameter (Bilateral Filter)", 5, 15, 9, 2)
        sigmaColor = st.slider("Sigma Color", 10, 100, 75, 5)
        sigmaSpace = st.slider("Sigma Space", 10, 100, 75, 5)
        processed_image = cv2.bilateralFilter(processed_image, d, sigmaColor, sigmaSpace)

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

    st.image(combined, use_column_width=True, caption="قبل/بعد - إزالة الضوضاء")

    # ===============================
    # 💾 حفظ الصورة
    # ===============================
    if st.button("💾 حفظ الصورة"):
        output = Image.fromarray(processed_image)
        output.save("lecture5_output.png")
        st.success("✅ تم حفظ الصورة باسم lecture5_output.png")
