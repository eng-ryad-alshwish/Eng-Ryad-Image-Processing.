import streamlit as st
import cv2
import numpy as np
from PIL import Image

def run():
    st.title("📘 المحاضرة 8: التحويلات الهندسية (Geometric Transforms)")

    # ===============================
    # 📝 الشرح النظري في Expander
    # ===============================
    with st.expander("📖 الشرح النظري", expanded=False):
        st.markdown("""
        التحويلات الهندسية تسمح بتغيير شكل الصورة أو موضعها:  
        - **Translation (التحريك):** نقل الصورة أفقيًا أو عموديًا.  
        - **Rotation (الدوران):** تدوير الصورة بزاوية معينة.  
        - **Scaling (التكبير/التصغير):** تغيير أبعاد الصورة بنسبة محددة.  
        - **Flipping (الانعكاس):** انعكاس أفقي أو رأسي.  
        - **Cropping (القص):** قص جزء من الصورة.
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
        img = np.full((300, 400, 3), 180, dtype=np.uint8)  # صورة رمادية بسيطة

    original_image = img.copy()
    processed_image = img.copy()
    h, w = original_image.shape[:2]

    # ===============================
    # 🎛️ أدوات التحكم بالتحويلات
    # ===============================
    st.subheader("🎛️ التحكم بالتحويلات الهندسية")

    # Rotation
    angle = st.slider("زاوية الدوران (degrees)", -180, 180, 0)

    # Scaling
    scale = st.slider("نسبة التكبير/التصغير", 0.1, 3.0, 1.0)

    # Flipping
    flip_mode = st.selectbox("الانعكاس", ["None", "Horizontal", "Vertical"])

    # Cropping
    crop_x = st.slider("قص: بدء X", 0, w-1, 0)
    crop_y = st.slider("قص: بدء Y", 0, h-1, 0)
    crop_w = st.slider("قص: عرض المنطقة", 1, w-crop_x, w-crop_x)
    crop_h = st.slider("قص: ارتفاع المنطقة", 1, h-crop_y, h-crop_y)

    # ===============================
    # 🛠️ تطبيق التحويلات
    # ===============================
    # الدوران + التكبير/التصغير
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
    transformed = cv2.warpAffine(processed_image, M, (w, h))

    # الانعكاس
    if flip_mode == "Horizontal":
        transformed = cv2.flip(transformed, 1)
    elif flip_mode == "Vertical":
        transformed = cv2.flip(transformed, 0)

    # القص
    transformed = transformed[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    processed_image = transformed

    # ===============================
    # 🔀 عرض قبل/بعد مع شريط تمرير
    # ===============================
    st.subheader("📊 عرض قبل/بعد")
    ratio = st.slider("نسبة ظهور الصورة بعد التحويل", 0, 100, 50)
    
    # لضبط حجم الصورة المعروضة بحيث تساوي حجم الأصل
    display_h = processed_image.shape[0]
    display_w = processed_image.shape[1]

    # ملء الصورة الأصلية لتكون بنفس الحجم بعد القص
    combined = np.zeros_like(processed_image)
    overlay_width = int(display_w * ratio / 100)
    combined[:, :overlay_width] = processed_image[:, :overlay_width]
    combined[:, overlay_width:] = processed_image[:, overlay_width:]  # هنا الأصل هو processed_image بعد التحويل

    st.image(combined, use_column_width=True, caption="قبل/بعد - التحويلات الهندسية")

    # ===============================
    # 💾 حفظ الصورة
    # ===============================
    if st.button("💾 حفظ الصورة"):
        output = Image.fromarray(processed_image)
        output.save("lecture8_output.png")
        st.success("✅ تم حفظ الصورة باسم lecture8_output.png")
