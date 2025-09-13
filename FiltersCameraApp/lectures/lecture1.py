import streamlit as st
import cv2
import numpy as np
from PIL import Image

def run():
    # ===============================
    # 📝 الشرح النظري (مكثف 6–8 أسطر)
    # ===============================
    st.title("📘 المحاضرة 1: مدخل ومعمارية الصور الرقمية")
    st.markdown("""
    الصورة الرقمية **Digital Image** هي تمثيل عددي لمصفوفة ثنائية أو ثلاثية الأبعاد 
    تتكون من نقاط صغيرة تسمى **بكسلات (Pixels)**، وكل بكسل يمثل أصغر عنصر في الصورة.
    
    كل صورة يتم تعريفها بواسطة ثلاثة عناصر أساسية:
    1. **الأبعاد (Height × Width)**: تحدد عدد الصفوف والأعمدة من البكسلات.
    2. **القنوات (Channels)**: مثل قناة واحدة (رمادي) أو ثلاث قنوات (ملونة RGB).
    3. **العمق اللوني (Bit Depth)**: يحدد عدد القيم الممكنة للبكسل (8-bit → 256 مستوى، 16-bit → 65536 مستوى).

    على سبيل المثال: صورة ملونة بحجم (1080×1920×3) تحتوي على 1080 صفًا و1920 عمودًا 
    و3 قنوات (أحمر، أخضر، أزرق). كل بكسل يحتوي على قيم رقمية بين 0 و255 
    تحدد شدة اللون في كل قناة.
    """)

    st.divider()

    # ===============================
    # 📤 رفع صورة / صورة افتراضية
    # ===============================
    uploaded_file = st.file_uploader("📤 قم برفع صورة لاكتشاف معلوماتها ", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        # إذا رفع صورة
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # ===============================
        # ℹ️ عرض معلومات الصورة
        # ===============================
        height, width = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]
        bit_depth = img.dtype.itemsize * 8

        st.subheader("ℹ️ معلومات عن الصورة:")
        st.write(f"- **العرض (Width):** {width} بكسل")
        st.write(f"- **الارتفاع (Height):** {height} بكسل")
        st.write(f"- **عدد القنوات (Channels):** {channels}")
        st.write(f"- **العمق اللوني (Bit Depth):** {bit_depth}-bit")

        # ===============================
        # 🖼️ عرض الصورة
        # ===============================
        st.subheader("🖼️ الصورة الأصلية")
        st.image(img, use_column_width=True)
    else:
        pass    

