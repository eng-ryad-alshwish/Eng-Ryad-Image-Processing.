import streamlit as st

import lectures.lecture1 as lecture1
import lectures.lecture2 as lecture2
import lectures.lecture3 as lecture3
import lectures.lecture4 as lecture4
import lectures.lecture5 as lecture5
import lectures.lecture6 as lecture6
import lectures.lecture7 as lecture7
import lectures.lecture8 as lecture8
import lectures.lecture9 as lecture9

st.set_page_config(
    page_title="📚 سلسلة محاضرات معالجة الصور",
    layout="wide"
)

lectures_dict = {
    "📘 المحاضرة 1:\n مدخل ومعمارية الصور الرقمية": lecture1,
    "📘 المحاضرة 2:\n أنظمة الألوان (Color Spaces)": lecture2,
    "📘 المحاضرة 3:\n العمليات على البكسل (Point Operations)": lecture3,
    "📘 المحاضرة 4:\n الفلاتر والالتفاف (Filtering & Convolution)": lecture4,
    "📘 المحاضرة 5:\n إزالة الضوضاء (Denoising)": lecture5,
    "📘 المحاضرة 6:\n كشف الحواف (Edge Detection)": lecture6,
    "📘 المحاضرة 7:\n العمليات المورفولوجية (Morphological Ops)": lecture7,
    "📘 المحاضرة 8:\n التحويلات الهندسية (Geometric Transforms)": lecture8,
    "📘 المحاضرة 9:\n المشروع الختامي (Final Project)": lecture9,
}

def main():
    st.sidebar.title("📚 سلسلة محاضرات معالجة الصور")
    st.sidebar.markdown("---")

    # اختيار المحاضرة من القائمة
    choice = st.sidebar.radio("اختر المحاضرة:", list(lectures_dict.keys()))

    # تشغيل المحاضرة المختارة
    module = lectures_dict[choice]
    module.run()

if __name__ == "__main__":
    main()
