import streamlit as st
from pages.HomePage import home_page
from pages.LoginPage import login_page
from pages.RegisterPage import register_page
from pages.DashboardPage import dashboard_page
import time

# =====================
# 🌐 التهيئة الأساسية للتطبيق
# =====================
st.set_page_config(
    page_title="FaceLogin",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================
# 🧠 إدارة الحالة (State Management)
# =====================
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

if "loading" not in st.session_state:
    st.session_state.loading = False

if "screen_size" not in st.session_state:
    st.session_state.screen_size = "desktop"  # desktop, tablet, mobile

# =====================
# 🎨 CSS عام لتحسين التصميم المتجاوب
# =====================
# st.markdown("""
# <style>
# /* أساسيات التصميم المتجاوب */
# :root {
#     --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     --card-bg: rgba(255, 255, 255, 0.12);
#     --card-border: 1px solid rgba(255, 255, 255, 0.2);
#     --card-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 30px rgba(255, 255, 255, 0.1);
#     --text-primary: #ffffff;
#     --text-secondary: #e0e0ff;
#     --font-main: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#     --font-heading: 'Montserrat', sans-serif;
# }

# /* تصميم متجاوب للعناصر الأساسية */
# .stApp {
#     background: var(--primary-gradient);
#     background-attachment: fixed;
#     color: var(--text-primary);
#     font-family: var(--font-main);
# }

# /* تحسينات عامة للعناصر */
# h1, h2, h3, h4, h5, h6 {
#     font-family: var(--font-heading);
#     text-align: center;
#     margin-bottom: 0.5rem;
# }

# h1 {
#     font-size: 2.8em;
#     font-weight: 800;
# }

# h2 {
#     font-size: 1.6em;
#     color: var(--text-secondary);
#     font-weight: 500;
# }

# p {
#     font-size: 1.1em;
#     line-height: 1.7;
#     color: #d0d0e0;
# }

# /* تحسينات للأزرار بشكل متجاوب */
# .stButton > button {
#     border-radius: 16px;
#     padding: 14px 28px;
#     font-size: 16px;
#     font-weight: 600;
#     color: white !important;
#     border: none !important;
#     cursor: pointer;
#     transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
#     margin: 8px 0;
#     box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
#     font-family: var(--font-main);
#     min-width: 120px;
#     min-height: 40px;
#     width: 100%;
# }

# .stButton > button:hover {
#     transform: translateY(-3px);
#     box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
# }

# /* تحسينات للحقول النصية */
# .stTextInput > div > div > input {
#     border-radius: 12px;
#     padding: 12px 16px;
#     font-size: 16px;
#     background: rgba(255, 255, 255, 0.1);
#     border: 1px solid rgba(255, 255, 255, 0.2);
#     color: white;
# }

# .stTextInput > div > div > input::placeholder {
#     color: rgba(255, 255, 255, 0.6);
# }

# /* تحسينات للأعمدة والعناصر المتجاوبة */
# .block-container {
#     padding: 2rem 1rem;
# }

# /* تحسينات للهواتف */
# @media (max-width: 768px) {
#     h1 {
#         font-size: 2.2em;
#     }
    
#     h2 {
#         font-size: 1.4em;
#     }
    
#     .stButton > button {
#         padding: 12px 20px;
#         font-size: 14px;
#     }
    
#     .block-container {
#         padding: 1rem 0.5rem;
#     }
# }

# /* تحسينات للشاشات الكبيرة */
# @media (min-width: 1200px) {
#     .block-container {
#         max-width: 1000px;
#         margin: 0 auto;
#     }
# }

# /* كارد زجاجي أساسي */
# .glass-card {
#     background: var(--card-bg);
#     backdrop-filter: blur(18px);
#     -webkit-backdrop-filter: blur(18px);
#     border-radius: 24px;
#     border: var(--card-border);
#     padding: 40px;
#     box-shadow: var(--card-shadow);
#     max-width: 700px;
#     margin: 30px auto;
#     text-align: center;
#     position: relative;
#     overflow: hidden;
#     animation: fadeInUp 0.8s ease-out;
# }

# .glass-card::before {
#     content: '';
#     position: absolute;
#     top: -50%;
#     left: -50%;
#     width: 200%;
#     height: 200%;
#     background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
#     pointer-events: none;
#     animation: pulse 4s infinite;
# }

# @keyframes fadeInUp {
#     from { opacity: 0; transform: translateY(40px); }
#     to { opacity: 1; transform: translateY(0); }
# }

# @keyframes pulse {
#     0% { transform: translate(-50%, -50%) scale(1); }
#     50% { transform: translate(-50%, -50%) scale(1.05); }
#     100% { transform: translate(-50%, -50%) scale(1); }
# }

# /* تحسينات للعناصر الداخلية في الكارد */
# .glass-card-content {
#     position: relative;
#     z-index: 2;
# }

# /* تحسينات للحالة loading */
# .stSpinner > div {
#     margin: 0 auto;
# }

# /* تحسينات للفيديو والعناصر الوسائط */
# .video-container {
#     margin: 20px auto;
#     border-radius: 16px;
#     overflow: hidden;
#     box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
#     background: rgba(0, 0, 0, 0.1);
#     max-width: 600px;
#     height: 360px;
#     position: relative;
#     border: 1px solid rgba(255, 255, 255, 0.1);
# }

# /* تحسينات للجداول والعناصر الأخرى */
# .stDataFrame {
#     border-radius: 12px;
#     overflow: hidden;
# }

# /* تحسينات للرسائل والألارم */
# .stAlert {
#     border-radius: 12px;
# }

# /* تحسينات للأيقونات */
# .icon-large {
#     font-size: 2em;
#     margin-bottom: 15px;
# }

# </style>
# """, unsafe_allow_html=True)

# =====================
# 🚀 دالة التنقل الآمنة بين الصفحات
# =====================
def navigate_to(page):
    if st.session_state.current_page != page:
        st.session_state.current_page = page
        st.session_state.loading = True
        st.rerun()

# =====================
# 📱 كاشف حجم الشاشة (مبسط)
# =====================
def detect_screen_size():
    # هذه دالة مبسطة - في التطبيق الحقيقي قد تحتاج إلى JavaScript
    # للكشف عن حجم الشاشة الفعلي
    return "desktop"  # سيتم تحسين هذا لاحقًا

# =====================
# 🖥️ دالة مساعدة لعرض الصفحة الحالية بطريقة متجاوبة
# =====================
def render_current_page():
    """
    عرض الصفحة الحالية مع التكيف التلقائي مع حجم الشاشة
    """
    screen_size = detect_screen_size()
    
    if st.session_state.current_page == "home":
        home_page(screen_size=screen_size)
    elif st.session_state.current_page == "login":
        login_page(screen_size=screen_size)
    elif st.session_state.current_page == "register":
        register_page(screen_size=screen_size)
    elif st.session_state.current_page == "dashboard":
        dashboard_page(screen_size=screen_size)
    else:
        st.error("❌ صفحة غير موجودة")
        if st.button("🏠 العودة للصفحة الرئيسية", use_container_width=True):
            navigate_to("home")

# =====================
# 🖥️ عرض الصفحة مع تأثير التحميل
# =====================
def render_page():
    if st.session_state.loading:
        with st.spinner("🔄 جاري التحول إلى الصفحة المطلوبة..."):
            time.sleep(0.6)
        st.session_state.loading = False

    # استخدام الحاوية الأساسية
    with st.container():
        render_current_page()

# =====================
# 🌟 دالة مساعدة لإنشاء أزرار متجاوبة
# =====================
def responsive_buttons(buttons_list, col_ratio=None):
    """
    عرض قائمة أزرار بشكل متجاوب.
    buttons_list: قائمة من tuples [(label, callback_function), ...]
    col_ratio: قائمة نسب الأعمدة (مثلاً [2,1])
    """
    num_buttons = len(buttons_list)
    
    # تحديد عدد الأعمدة بناءً على عدد الأزرار وحجم الشاشة
    if num_buttons <= 2:
        cols = st.columns(num_buttons)
    else:
        # للأزرار الكثيرة، نعرضها في صفوف متعددة على الهواتف
        cols_per_row = 2 if st.session_state.screen_size == "mobile" else num_buttons
        cols = st.columns(cols_per_row)
    
    for i, (label, callback) in enumerate(buttons_list):
        col_idx = i % len(cols)
        with cols[col_idx]:
            st.button(label, on_click=callback, use_container_width=True)

# =====================
# 🏁 تشغيل التطبيق
# =====================
if __name__ == "__main__":
    render_page()