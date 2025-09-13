import streamlit as st
import os

# =====================
# 🌐 التهيئة الأساسية للتطبيق الموحد
# =====================
st.set_page_config(
    page_title="🖥️ لوحة التحكم الرئيسية",
    page_icon="🌟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================
# 🎨 تصميم متجاوب مع الهاتف
# =====================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

h1 {
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    font-size: 3em;
    margin: 0.5em 0 1.5em 0;
    color: #ffffff;
    text-shadow: 0 4px 12px rgba(0,0,0,0.3);
    letter-spacing: -0.5px;
}

/* حاوية الأزرار */
.button-grid {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 2rem;
    padding: 2rem 1rem;
    margin-top: 2rem;
}

/* تصميم الأزرار (Glassmorphism + تأثيرات) */
.stButton > button {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    padding: 20px 30px;
    font-size: 1.4rem;
    font-weight: 600;
    color: white !important;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    min-width: 220px;
    max-width: 300px;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    text-align: center;
    line-height: 1.4;
}

.stButton > button:hover {
    transform: translateY(-6px) scale(1.03);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.35);
    background: rgba(255, 255, 255, 0.25);
    border-color: rgba(255, 255, 255, 0.3);
}

/* تحسينات للجوال */
@media (max-width: 768px) {
    h1 {
        font-size: 2em;
        margin: 0.3em 0 1em 0;
    }
    .button-grid {
        gap: 1rem;
        padding: 1rem;
        flex-direction: column;
        align-items: stretch;
    }
    .stButton > button {
        padding: 16px 20px;
        font-size: 1.1rem;
        min-width: 100% !important;
        max-width: 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)

import streamlit as st
import os
import subprocess, sys

# =====================
# عنوان التطبيق
# =====================
st.markdown("<h1>🖥️ لوحة التحكم الرئيسية</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.2em; color: #e0e0ff; margin-bottom: 3rem;'>اختر التطبيق الذي تريد تشغيله</p>", unsafe_allow_html=True)

# =====================
# التطبيقات المتاحة (مع أيقونات)
# =====================
apps = [
    {
        "name": " تسجيل الدخول بالوجه",
        "path": "FaceLoginWeb/app.py",
        "icon": "🔐",
        "description": "تسجيل دخول آمن باستخدام التعرف على الوجه",
        "port": 8502
    },
    {
        "name": " الفلاتر الذكية",
        "path": "FiltersCameraApp/app.py",
        "icon": "📸",
        "description": "تطبيق تحرير الصور بفلاتر ذكية",
        "port": 8503
    },
    {
        "name": "سلسلة المحاصرات التفاعليه",
        "path": "ProjectStringLec/app.py",
        "icon": "📈",
        "description": "معالجة الصور بطريقة والتعرف على الأنماط",
        "port": 8504
    }
]

# =====================
# عرض الأزرار
# =====================
with st.container():
    cols = st.columns([1, 1, 1] if len(apps) >= 3 else [1])
    
    for idx, app in enumerate(apps):
        with cols[idx % len(cols)]:
            with st.container():
                st.markdown(f"""
                    <div style='text-align: center; margin: 10px 0;'>
                        <span style='font-size: 2rem; margin-bottom: 10px;'>{app['icon']}</span><br>
                        <strong>{app['name']}</strong><br>
                        <small style='color: #d0d0e0; font-size: 0.9em;'>{app['description']}</small>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 ابدأ", key=f"btn_{idx}", use_container_width=True):
                    if os.path.exists(app["path"]):
                        subprocess.Popen([
                            sys.executable, "-m", "streamlit", 
                            "run", app["path"], 
                            "--server.headless=true",
                            f"--server.port={app['port']}"
                        ])
                        url = f"http://localhost:{app['port']}"
                        st.markdown(f"✅ التطبيق يعمل الآن: [اضغط هنا لفتحه في تبويب جديد]({url})", unsafe_allow_html=True)
                    else:
                        st.error(f"❌ الملف `{app['path']}` غير موجود.")

# =====================
# ملاحظة أسفل الصفحة
# =====================
st.markdown("""
    <hr style='border: 1px solid rgba(255,255,255,0.2); margin: 4rem 0;'>
    <p style='font-size: 0.9em; color: #b0b0d0; text-align: center;'>
        © 2025 لوحة التحكم المركزية | جميع التطبيقات تعمل محليًا<br>
        ⚠️ لا تستخدم هذه اللوحة على السحابة إلا إذا كانت التطبيقات مُركبة مسبقًا.
    </p>
""", unsafe_allow_html=True)

