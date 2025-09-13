import streamlit as st
import cv2
import numpy as np
import pickle
import os
import face_recognition
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import time

# =====================
# 📁 افتراضات: وجود مجلد face_encodings/ مع ملفات .pkl
# =====================
ENCODINGS_DIR = "face_encodings"
os.makedirs(ENCODINGS_DIR, exist_ok=True)

def login_page(screen_size="desktop"):
    """
    صفحة تسجيل الدخول - تصميم متجاوب مع جميع أحجام الشاشات
    """
    
    # =====================
    # 🎨 محتوى الصفحة (تصميم متجاوب)
    # =====================
    
    # تحديد أنماط التكيف مع حجم الشاشة
    if screen_size == "mobile":
        card_padding = "30px 20px"
        title_size = "2.2em"
        subtitle_size = "1.4em"
        section_title_size = "1.2em"
        text_size = "1.0em"
        video_height = "280px"
        button_layout = "vertical"  # أزرار عمودية للهواتف
    else:
        card_padding = "40px"
        title_size = "2.8em"
        subtitle_size = "1.6em"
        section_title_size = "1.3em"
        text_size = "1.1em"
        video_height = "360px"
        button_layout = "horizontal"  # أزرار أفقية للشاشات الكبيرة

    # =====================
    # 🖼️ محتوى الصفحة
    # =====================
    st.markdown(f"""
        <div class='glass-card' style='padding: {card_padding}'>
            <div class='glass-card-content'>
    """, unsafe_allow_html=True)

    # العنوان الرئيسي
    st.markdown(f"<h1 style='font-size: {title_size}'>👋 مرحبًا بك</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-size: {subtitle_size}'>تسجيل الدخول بالوجه</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: {text_size}'>يرجى إدخال اسم المستخدم ثم توجيه وجهك نحو الكاميرا لتأكيد الهوية.</p>", unsafe_allow_html=True)

    # حقل اسم المستخدم
    username = st.text_input(
        "👤 اسم المستخدم",
        placeholder="أدخل اسم المستخدم الخاص بك",
        key="login_username",
        label_visibility="collapsed"
    )

    # تحقق من صحة الاسم قبل بدء الكاميرا
    if username.strip() == "":
        st.info("📌 يرجى إدخال اسم المستخدم أولاً.")
        st.stop()

    enc_file = os.path.join(ENCODINGS_DIR, f"{username.strip()}.pkl")
    if not os.path.exists(enc_file):
        st.error(f"❌ لا يوجد حساب مسجل باسم '{username}'، يرجى التسجيل أولًا.")
        st.stop()

    # =====================
    # 🧠 إدارة الحالة
    # =====================
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False
    if "known_encodings" not in st.session_state:
        st.session_state.known_encodings = None
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "captured_frame" not in st.session_state:
        st.session_state.captured_frame = None
    if "last_attempt" not in st.session_state:
        st.session_state.last_attempt = None
    if "attempts" not in st.session_state:
        st.session_state.attempts = 0

    # تحميل الترميزات عند أول استخدام
    if st.session_state.known_encodings is None:
        try:
            with open(enc_file, "rb") as f:
                data = pickle.load(f)
                st.session_state.known_encodings = data['embeddings']
            st.success(f"✅ تم تحميل بيانات المستخدم: {username}")
        except Exception as e:
            st.error(f"❌ خطأ في تحميل ملف الترميزات: {str(e)}")
            st.stop()

    # =====================
    # 📹 منطقة الكاميرا
    # =====================
    st.markdown(f"<h3 style='font-size: {section_title_size}'>📹 الكاميرا الحية</h3>", unsafe_allow_html=True)

    video_placeholder = st.empty()
    status_placeholder = st.empty()

    # أزرار التحكم في الكاميرا (تخطيط متجاوب)
    if button_layout == "horizontal":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ تشغيل الكاميرا", use_container_width=True, key="btn_start"):
                st.session_state.camera_active = True
                st.session_state.authenticated = False
                st.session_state.captured_frame = None
                st.session_state.attempts = 0
                st.rerun()
        with col2:
            if st.button("🔍 التحقق من الوجه", use_container_width=True, key="btn_recognize", disabled=not st.session_state.camera_active):
                if st.session_state.captured_frame is None:
                    st.warning("⏳ لا يوجد إطار ملتقط. تأكد من أن الكاميرا تعمل.")
                else:
                    rgb_frame = cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    if len(face_locations) == 0:
                        st.error("❌ لم يتم اكتشاف أي وجه.")
                    else:
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        found_match = False
                        for encoding in face_encodings:
                            matches = face_recognition.compare_faces(st.session_state.known_encodings, encoding)
                            if True in matches:
                                found_match = True
                                break

                        if found_match:
                            st.session_state.authenticated = True
                            st.session_state.attempts += 1
                            st.session_state.last_attempt = "success"
                            st.success(f"✅ تم التعرف على الوجه! أهلاً بك {username}...")
                            time.sleep(1.5)
                            st.session_state.current_page = "dashboard"
                            st.rerun()
                        else:
                            st.session_state.attempts += 1
                            st.session_state.last_attempt = "failed"
                            st.error("❌ لم يتم التعرف على الوجه. جرب مرة أخرى.")
    else:
        # للهواتف: أزرار عمودية
        if st.button("▶️ تشغيل الكاميرا", use_container_width=True, key="btn_start"):
            st.session_state.camera_active = True
            st.session_state.authenticated = False
            st.session_state.captured_frame = None
            st.session_state.attempts = 0
            st.rerun()
        
        if st.button("🔍 التحقق من الوجه", use_container_width=True, key="btn_recognize", disabled=not st.session_state.camera_active):
            if st.session_state.captured_frame is None:
                st.warning("⏳ لا يوجد إطار ملتقط. تأكد من أن الكاميرا تعمل.")
            else:
                rgb_frame = cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                if len(face_locations) == 0:
                    st.error("❌ لم يتم اكتشاف أي وجه.")
                else:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    found_match = False
                    for encoding in face_encodings:
                        matches = face_recognition.compare_faces(st.session_state.known_encodings, encoding)
                        if True in matches:
                            found_match = True
                            break

                    if found_match:
                        st.session_state.authenticated = True
                        st.session_state.attempts += 1
                        st.session_state.last_attempt = "success"
                        st.success(f"✅ تم التعرف على الوجه! أهلاً بك {username}...")
                        time.sleep(1.5)
                        st.session_state.current_page = "dashboard"
                        st.rerun()
                    else:
                        st.session_state.attempts += 1
                        st.session_state.last_attempt = "failed"
                        st.error("❌ لم يتم التعرف على الوجه. جرب مرة أخرى.")

    # زر إعادة المحاولة
    if st.button("🔄 إعادة المحاولة", use_container_width=True, key="btn_retry", disabled=not st.session_state.camera_active or st.session_state.authenticated):
        st.session_state.captured_frame = None
        st.session_state.authenticated = False
        st.rerun()

    # عرض الفيديو الحي فقط إذا كانت الكاميرا نشطة
    if st.session_state.camera_active and not st.session_state.authenticated:
        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            st.session_state.captured_frame = img
            return frame

        # حاوية الفيديو مع حدود وأيقونة توجيه
        with st.container():
            st.markdown(f'<div class="video-container" style="height: {video_height}">', unsafe_allow_html=True)
            webrtc_streamer(
                key="login_camera",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # عرض الصورة الملتقطة في الوقت الفعلي
        if st.session_state.captured_frame is not None:
            frame_rgb = cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, caption="📍 ضع وجهك داخل الإطار وحافظ على الاستقرار...", channels="RGB", use_container_width=True)

        # مؤشر حالة التحقق
        if st.session_state.last_attempt == "success":
            status_placeholder.markdown("<span class='status-indicator success'></span> <strong style='color: #4CAF50;'>تم التعرف بنجاح!</strong>", unsafe_allow_html=True)
        elif st.session_state.last_attempt == "failed":
            status_placeholder.markdown("<span class='status-indicator error'></span> <strong style='color: #F44336;'>فشل التعرف. جرب مرة أخرى.</strong>", unsafe_allow_html=True)
        elif st.session_state.camera_active:
            status_placeholder.markdown("<span class='status-indicator warning'></span> <strong style='color: #FFC107;'>جارٍ التحليل...</strong>", unsafe_allow_html=True)

    # زر تسجيل الدخول بكلمة المرور (مقترح)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔑 تسجيل الدخول بكلمة المرور", use_container_width=True, key="btn_password_login"):
        st.info("🔒 هذه الميزة قيد التطوير. سيتم إضافة دعم كلمات المرور المشفرة قريبًا!")

    # زر العودة للصفحة الرئيسية
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ العودة للصفحة الرئيسية", use_container_width=True, key="btn_back_login"):
        st.session_state.camera_active = False
        st.session_state.authenticated = False
        st.session_state.known_encodings = None
        st.session_state.captured_frame = None
        st.session_state.last_attempt = None
        st.session_state.attempts = 0
        st.session_state.current_page = "home"
        st.rerun()

    # =====================
    # 🛡️ ملاحظات أمنية (تظهر فقط بعد محاولتين فاشلتين)
    # =====================
    if st.session_state.attempts >= 2:
        st.warning("⚠️ تم تجاوز عدد المحاولات. يُنصح بالتأكد من الإضاءة ووضوح الوجه.")

    st.markdown("""
            </div>
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