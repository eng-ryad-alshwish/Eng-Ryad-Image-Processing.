import streamlit as st
import cv2
import numpy as np
from PIL import Image

def run():
    # ===============================
    # 📝 الشرح النظري
    # ===============================
    st.title("📘 المحاضرة 2: أنظمة الألوان (Color Spaces)")
    st.markdown("""
    الصورة الرقمية يمكن تمثيلها بعدة أنظمة ألوان (Color Spaces) حسب التطبيق:

    1. **RGB (Red, Green, Blue):** الأكثر شيوعًا في العرض والشاشات.  
    2. **BGR:** تنسيق OpenCV الافتراضي (مقلوب عن RGB).  
    3. **Grayscale:** قناة واحدة (درجة إضاءة فقط)، يستخدم في المعالجة البسيطة أو كشف الحواف.  
    4. **HSV (Hue, Saturation, Value):** يفصل اللون (Hue) عن الإضاءة (Value)، مناسب لمعالجة الألوان وكشف الأجسام.

    اختيار نظام الألوان المناسب يسهل العمليات مثل: **التقطيع (Segmentation)** أو **كشف الحواف** أو **تحسين الألوان**.
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
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[:, :150] = (255, 0, 0)   # نصف أحمر
        img[:, 150:] = (0, 255, 0)   # نصف أخضر

    original_image = img.copy()

    # ===============================
    # 🔘 اختيار العملية
    # ===============================
    st.subheader("🎛️ العمليات المتاحة")
    option = st.radio("اختر العملية:", ["تحويل إلى Gray", "تحويل إلى HSV", "تقسيم القنوات (R/G/B)"])

    if option == "تحويل إلى Gray":
        processed_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)  # لإظهارها بنفس الشكل

    elif option == "تحويل إلى HSV":
        processed_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_HSV2RGB)  # للعرض الصحيح

    elif option == "تقسيم القنوات (R/G/B)":
        R, G, B = cv2.split(original_image)
        zeros = np.zeros_like(R)
        red_img = cv2.merge([R, zeros, zeros])
        green_img = cv2.merge([zeros, G, zeros])
        blue_img = cv2.merge([zeros, zeros, B])
        st.image([red_img, green_img, blue_img], caption=["القناة الحمراء", "القناة الخضراء", "القناة الزرقاء"], use_column_width=True)
        processed_image = original_image  # للعرض في شريط التمرير لاحقًا

    else:
        processed_image = original_image

    # ===============================
    # 🔀 شريط التمرير (قبل/بعد)
    # ===============================
    st.subheader("📊 عرض قبل/بعد")
    ratio = st.slider("نسبة ظهور الصورة بعد التحويل", 0, 100, 50)

    width, height = original_image.shape[1], original_image.shape[0]
    overlay_width = int(width * ratio / 100)
    combined = np.zeros_like(original_image)
    combined[:, :overlay_width] = processed_image[:, :overlay_width]
    combined[:, overlay_width:] = original_image[:, overlay_width:]

    st.image(combined, use_column_width=True)

    # ===============================
    # 💾 حفظ الصورة
    # ===============================
    if st.button("💾 حفظ الصورة"):
        output = Image.fromarray(processed_image)
        output.save("lecture2_output.png")
        st.success("✅ تم حفظ الصورة باسم lecture2_output.png")
