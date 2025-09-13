import streamlit as st

def dashboard_page(screen_size="desktop"):
    """
    صفحة لوحة التحكم - تصميم متجاوب مع جميع أحجام الشاشات
    """
    
    # =====================
    # 🎨 محتوى الصفحة (تصميم متجاوب)
    # =====================
    
    # تحديد أنماط التكيف مع حجم الشاشة
    if screen_size == "mobile":
        card_padding = "30px 20px"
        title_size = "2.0em"
        text_size = "1.1em"
        button_padding = "12px 20px"
    else:
        card_padding = "40px"
        title_size = "2.5em"
        text_size = "1.2em"
        button_padding = "14px 20px"

    # =====================
    # 🖼️ محتوى الصفحة
    # =====================
    st.markdown(f"""
        <div class='glass-card' style='padding: {card_padding}'>
            <div class='glass-card-content'>
    """, unsafe_allow_html=True)

    # العنوان الرئيسي
    st.markdown(f"<h1 style='font-size: {title_size}'>لوحة التحكم - Dashboard</h1>", unsafe_allow_html=True)

    # التوصيف الفرعي
    st.markdown(f"<p style='font-size: {text_size}'>مرحبًا بك! هذه الصفحة فارغة حاليًا ويمكنك تطويرها لاحقًا.</p>", unsafe_allow_html=True)

    # مسافة قبل زر تسجيل الخروج
    st.markdown("<br><br>", unsafe_allow_html=True)

    # زر تسجيل الخروج
    if st.button("🚪 تسجيل الخروج", key="btn_logout_dashboard", use_container_width=True):
        st.session_state.current_page = "home"
        st.rerun()

    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)

    # إضافة بعض العناصر التوضيحية (يمكن تطويرها لاحقًا)
    if screen_size != "mobile":  # فقط للشاشات الكبيرة
        st.markdown("<br>", unsafe_allow_html=True)
        
        # إضافة بعض الإحصائيات الوهمية كمثال
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div style='background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 16px; text-align: center;'>
                    <h3 style='color: #4facfe; margin: 0;'>5</h3>
                    <p style='margin: 5px 0 0 0;'>جلسات نشطة</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style='background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 16px; text-align: center;'>
                    <h3 style='color: #4caf50; margin: 0;'>12</h3>
                    <p style='margin: 5px 0 0 0;'>عمليات تسجيل</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div style='background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 16px; text-align: center;'>
                    <h3 style='color: #ff9800; margin: 0;'>98%</h3>
                    <p style='margin: 5px 0 0 0;'>دقة التعرف</p>
                </div>
            """, unsafe_allow_html=True)

    # إضافة رسوم متحركة خفيفة للكارد
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