import streamlit as st

def home_page(screen_size="desktop"):
    """
    الصفحة الرئيسية للتطبيق - تصميم متجاوب مع جميع أحجام الشاشات
    """
    
    # =====================
    # 🎨 إعداد التوافق مع حجم الشاشة
    # =====================
    if screen_size == "mobile":
        card_padding = "20px"
        title_size = "2em"
        subtitle_size = "1.2em"
        columns_layout = 1
    else:
        card_padding = "40px"
        title_size = "2.8em"
        subtitle_size = "1.6em"
        columns_layout = 2

    # =====================
    # 🖼️ المحتوى الرئيسي
    # =====================
    st.markdown(f"""
        <div class='glass-card' style='padding: {card_padding}; backdrop-filter: blur(10px); border-radius: 15px; background: rgba(255,255,255,0.05);'>
            <div class='glass-card-content'>
    """, unsafe_allow_html=True)

    # العنوان الرئيسي
    st.markdown(f"<h1 style='font-size: {title_size}; text-align:center;'>FaceLogin</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-size: {subtitle_size}; text-align:center;'>تسجيل دخول آمن بالوجه — بدون كلمات مرور</h2>", unsafe_allow_html=True)

    # وصف مختصر
    st.markdown("""
        <p style='font-size: 1.1em; text-align:center; margin: 20px 0; line-height: 1.5;'>
            تجربة تسجيل دخول تعتمد على تقنية التعرف على الوجه.<br>
            لا حاجة لتذكر كلمات المرور — فقط انظر إلى الكاميرا وادخل.
        </p>
    """, unsafe_allow_html=True)

    # =====================
    # 🔘 أزرار التنقل
    # =====================
    if columns_layout == 2:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔐 تسجيل الدخول", key="btn_login_home", use_container_width=True):
                st.session_state.current_page = "login"
                st.rerun()
        with col2:
            if st.button("📝 إنشاء حساب", key="btn_register_home", use_container_width=True):
                st.session_state.current_page = "register"
                st.rerun()
    else:
        if st.button("🔐 تسجيل الدخول", key="btn_login_home", use_container_width=True):
            st.session_state.current_page = "login"
            st.rerun()
        if st.button("📝 إنشاء حساب", key="btn_register_home", use_container_width=True):
            st.session_state.current_page = "register"
            st.rerun()


    # ملاحظة توضيحية
    st.markdown("""
        <p style='font-size: 0.9em; text-align:center; margin-top: 20px;'>
            ⚠️ أول مرة؟ اضغط على "إنشاء حساب" لتسجيل وجهك أولاً.
        </p>
    """, unsafe_allow_html=True)

    # خط فاصل زخرفي
    st.markdown("""
        <div style='margin: 20px 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,255,255,0.3), transparent);'></div>
    """, unsafe_allow_html=True)

    # حقوق التطبيق
    st.markdown("""
        <p style='font-size: 0.8em; text-align:center; margin-top: 10px;'>
            © 2025 FaceLogin Pro | تقنية التعرف على الوجه متقدمة
        </p>
    """, unsafe_allow_html=True)

    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)

    # تأثير بسيط للظهور
    st.markdown("""
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const cards = document.querySelectorAll('.glass-card');
            cards.forEach(card => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                card.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
                setTimeout(() => {
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, 100);
            });
        });
        </script>
    """, unsafe_allow_html=True)
