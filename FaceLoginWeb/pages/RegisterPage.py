import streamlit as st
import cv2
import numpy as np
import pickle
import os
import face_recognition
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import time

# =====================
# 📁 إعداد المجلدات المطلوبة
# =====================
USERS_DIR = "users"
ENCODINGS_DIR = "face_encodings"
os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(ENCODINGS_DIR, exist_ok=True)

def register_page(screen_size="desktop"):
    """
    صفحة التسجيل - تصميم متجاوب مع جميع أحجام الشاشات
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
        thumb_size = "60px"
        button_layout = "vertical"  # أزرار عمودية للهواتف
    else:
        card_padding = "40px"
        title_size = "2.8em"
        subtitle_size = "1.6em"
        section_title_size = "1.3em"
        text_size = "1.1em"
        video_height = "360px"
        thumb_size = "80px"
        button_layout = "horizontal"  # أزرار أفقية للشاشات الكبيرة

    # =====================
    # 🖼️ محتوى الصفحة
    # =====================
    st.markdown(f"""
        <div class='glass-card' style='padding: {card_padding}'>
            <div class='glass-card-content'>
    """, unsafe_allow_html=True)

    # العنوان الرئيسي
    st.markdown(f"<h1 style='font-size: {title_size}'>👋 أهلاً بك!</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-size: {subtitle_size}'>أنشئ حسابك باستخدام وجهك</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: {text_size}'>سنساعدك على التقاط 5–20 صورة لوجهك. كل صورة تُحسن دقة التعرف.</p>", unsafe_allow_html=True)

    # حقول الإدخال
    username = st.text_input(
        "👤 اسم المستخدم",
        placeholder="أدخل اسمًا فريدًا (مثل: أحمد_محمد)",
        key="reg_username",
        label_visibility="collapsed"
    )

    password = st.text_input(
        "🔒 كلمة المرور",
        type="password",
        placeholder="أدخل كلمة مرور قوية (8+ أحرف)",
        key="reg_password",
        label_visibility="collapsed"
    )

    confirm_password = st.text_input(
        "🔒 تأكيد كلمة المرور",
        type="password",
        placeholder="أعد إدخال كلمة المرور",
        key="reg_confirm_password",
        label_visibility="collapsed"
    )

    num_photos = st.number_input(
        "📸 عدد الصور المطلوب التقاطها (بين 5 و20):",
        min_value=5,
        max_value=20,
        value=8,
        step=1,
        key="num_photos",
        help="كلما زاد عدد الصور، زادت دقة التعرف في ظروف مختلفة (إضاءة، زوايا...)"
    )

    # زر العودة (متوفر في أي وقت)
    if st.button("⬅️ رجوع للصفحة الرئيسية", use_container_width=True, key="btn_back_register"):
        # تنظيف حالة الجلسة
        if "camera_active" in st.session_state:
            st.session_state.camera_active = False
        if "captured_images" in st.session_state:
            st.session_state.captured_images = []
        if "embeddings" in st.session_state:
            st.session_state.embeddings = []
        if "captured_count" in st.session_state:
            st.session_state.captured_count = 0
        if "captured_frame" in st.session_state:
            st.session_state.captured_frame = None
        if "last_capture_status" in st.session_state:
            st.session_state.last_capture_status = None
        
        st.session_state.current_page = "home"
        st.rerun()

    # التحقق من صحة البيانات
    if not username.strip():
        st.info("📌 يرجى إدخال اسم مستخدم.")
        st.stop()

    if not password:
        st.info("📌 يرجى إدخال كلمة مرور.")
        st.stop()

    if password != confirm_password:
        st.error("❌ كلمتا المرور غير متطابقتين!")
        st.stop()

    if len(password) < 8:
        st.error("❌ كلمة المرور يجب أن تكون 8 أحرف على الأقل.")
        st.stop()

    # تحقق من وجود المستخدم مسبقًا
    enc_file = os.path.join(ENCODINGS_DIR, f"{username.strip()}.pkl")
    if os.path.exists(enc_file):
        st.error(f"❌ اسم المستخدم '{username}' موجود مسبقًا. يرجى اختيار اسم آخر.")
        st.stop()

    # =====================
    # 🧠 إدارة الحالة
    # =====================
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False
    if "captured_images" not in st.session_state:
        st.session_state.captured_images = []
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = []
    if "captured_count" not in st.session_state:
        st.session_state.captured_count = 0
    if "captured_frame" not in st.session_state:
        st.session_state.captured_frame = None
    if "last_capture_status" not in st.session_state:
        st.session_state.last_capture_status = None

    # أزرار التحكم في الكاميرا (تخطيط متجاوب)
    if button_layout == "horizontal":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ تشغيل الكاميرا", use_container_width=True, key="btn_start_cam"):
                if st.session_state.captured_count > 0:
                    st.session_state.captured_images = []
                    st.session_state.embeddings = []
                    st.session_state.captured_count = 0
                    st.session_state.captured_frame = None
                st.session_state.camera_active = True
                st.session_state.last_capture_status = None
                st.rerun()
        with col2:
            btn_capture_disabled = not st.session_state.camera_active or st.session_state.captured_count >= num_photos
            if st.button(
                f"📸 التقاط صورة ({st.session_state.captured_count}/{num_photos})",
                use_container_width=True,
                key="btn_capture",
                disabled=btn_capture_disabled
            ):
                if st.session_state.captured_frame is None:
                    st.warning("⏳ لا يوجد إطار ملتقط. حافظ على استقرار الكاميرا.")
                    return

                frame = st.session_state.captured_frame.copy()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)

                if len(face_locations) == 0:
                    st.session_state.last_capture_status = "error"
                    st.error("❌ لم يتم اكتشاف أي وجه. ضع وجهك داخل الإطار.")
                    return

                if len(face_locations) > 1:
                    st.session_state.last_capture_status = "error"
                    st.error("❌ تم اكتشاف أكثر من وجه. تأكد من أنك الوحيد في الإطار.")
                    return

                # تقييم جودة الوجه (إضاءة، وضوح)
                face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
                if not face_landmarks:
                    st.session_state.last_capture_status = "error"
                    st.error("❌ لا يمكن تحديد ملامح الوجه. حاول مرة أخرى في إضاءة أفضل.")
                    return

                # استخراج الترميز
                encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                st.session_state.embeddings.append(encoding)

                # حفظ الصورة
                user_dir = os.path.join(USERS_DIR, username.strip())
                os.makedirs(user_dir, exist_ok=True)
                img_path = os.path.join(user_dir, f"{st.session_state.captured_count + 1}.jpg")
                cv2.imwrite(img_path, frame)
                st.session_state.captured_images.append(img_path)
                st.session_state.captured_count += 1

                st.session_state.last_capture_status = "success"
                st.success(f"✅ تم التقاط الصورة {st.session_state.captured_count}/{num_photos} بنجاح!")
                time.sleep(1)
                st.rerun()
    else:
        # للهواتف: أزرار عمودية
        if st.button("▶️ تشغيل الكاميرا", use_container_width=True, key="btn_start_cam"):
            if st.session_state.captured_count > 0:
                st.session_state.captured_images = []
                st.session_state.embeddings = []
                st.session_state.captured_count = 0
                st.session_state.captured_frame = None
            st.session_state.camera_active = True
            st.session_state.last_capture_status = None
            st.rerun()
        
        btn_capture_disabled = not st.session_state.camera_active or st.session_state.captured_count >= num_photos
        if st.button(
            f"📸 التقاط صورة ({st.session_state.captured_count}/{num_photos})",
            use_container_width=True,
            key="btn_capture",
            disabled=btn_capture_disabled
        ):
            if st.session_state.captured_frame is None:
                st.warning("⏳ لا يوجد إطار ملتقط. حافظ على استقرار الكاميرا.")
                return

            frame = st.session_state.captured_frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if len(face_locations) == 0:
                st.session_state.last_capture_status = "error"
                st.error("❌ لم يتم اكتشاف أي وجه. ضع وجهك داخل الإطار.")
                return

            if len(face_locations) > 1:
                st.session_state.last_capture_status = "error"
                st.error("❌ تم اكتشاف أكثر من وجه. تأكد من أنك الوحيد في الإطار.")
                return

            # تقييم جودة الوجه (إضاءة، وضوح)
            face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
            if not face_landmarks:
                st.session_state.last_capture_status = "error"
                st.error("❌ لا يمكن تحديد ملامح الوجه. حاول مرة أخرى في إضاءة أفضل.")
                return

            # استخراج الترميز
            encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            st.session_state.embeddings.append(encoding)

            # حفظ الصورة
            user_dir = os.path.join(USERS_DIR, username.strip())
            os.makedirs(user_dir, exist_ok=True)
            img_path = os.path.join(user_dir, f"{st.session_state.captured_count + 1}.jpg")
            cv2.imwrite(img_path, frame)
            st.session_state.captured_images.append(img_path)
            st.session_state.captured_count += 1

            st.session_state.last_capture_status = "success"
            st.success(f"✅ تم التقاط الصورة {st.session_state.captured_count}/{num_photos} بنجاح!")
            time.sleep(1)
            st.rerun()

    # عرض الفيديو الحي
    if st.session_state.camera_active and st.session_state.captured_count < num_photos:
        st.markdown(f"<h3 style='font-size: {section_title_size}'>📹 الكاميرا الحية</h3>", unsafe_allow_html=True)

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            st.session_state.captured_frame = img
            return frame

        with st.container():
            st.markdown(f'<div class="video-container" style="height: {video_height}">', unsafe_allow_html=True)
            webrtc_streamer(
                key="register_camera",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # عرض الصورة الحالية
        if st.session_state.captured_frame is not None:
            frame_rgb = cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="📍 ضع وجهك داخل الدائرة، وحافظ على تعبير طبيعي.", channels="RGB", use_container_width=True)

        # مؤشر حالة التقاط الصورة
        if st.session_state.last_capture_status == "success":
            st.markdown("<span class='status-indicator success'></span> <strong style='color: #4CAF50;'>صورة ملتقطة بنجاح!</strong>", unsafe_allow_html=True)
        elif st.session_state.last_capture_status == "error":
            st.markdown("<span class='status-indicator error'></span> <strong style='color: #F44336;'>خطأ في التقاط الصورة. جرب مرة أخرى.</strong>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-indicator warning'></span> <strong style='color: #FFC107;'>استعد للتقاط الصورة...</strong>", unsafe_allow_html=True)

    # تقدم التسجيل
    if st.session_state.captured_count > 0:
        st.markdown(f"<h3 style='font-size: {section_title_size}'>📊 تقدم التسجيل</h3>", unsafe_allow_html=True)
        progress_percent = int((st.session_state.captured_count / num_photos) * 100)
        st.progress(progress_percent)
        st.markdown(f"<p class='progress-label'>تم التقاط {st.session_state.captured_count} من {num_photos} صور</p>", unsafe_allow_html=True)

        # عرض الصور المصغرة (حد أقصى 6 صور)
        if st.session_state.captured_images:
            st.markdown("<div class='thumb-container'>", unsafe_allow_html=True)
            # تحديد عدد الأعمدة بناءً على حجم الشاشة
            num_cols = min(len(st.session_state.captured_images), 4 if screen_size == "mobile" else 6)
            cols = st.columns(num_cols)
            
            for idx, img_path in enumerate(st.session_state.captured_images[:num_cols]):
                with cols[idx]:
                    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    st.image(img, caption=f"{idx+1}", use_container_width=True)
            
            if len(st.session_state.captured_images) > num_cols:
                st.caption(f"... و{len(st.session_state.captured_images)-num_cols} صور أخرى")
            st.markdown("</div>", unsafe_allow_html=True)

    # زر حفظ الحساب (يظهر فقط عند اكتمال الصور)
    if st.session_state.camera_active and st.session_state.captured_count >= num_photos:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 حفظ الحساب والخروج", use_container_width=True, key="btn_save"):
            # تأكيد قبل الحفظ
            if st.session_state.captured_count < 5:
                st.error("❌ يجب التقاط ما لا يقل عن 5 صور لحفظ الحساب.")
                return

            enc_file = os.path.join(ENCODINGS_DIR, f"{username.strip()}.pkl")
            data = {
                "password_hash": st.hashlib.sha256(password.encode()).hexdigest(),  # 🔐 تشفير كلمة المرور!
                "embeddings": st.session_state.embeddings
            }

            try:
                with open(enc_file, "wb") as f:
                    pickle.dump(data, f)
                st.success(f"✅ تم حفظ حساب '{username.strip()}' بنجاح!")
                st.balloons()
                time.sleep(2)
                st.session_state.camera_active = False
                st.session_state.captured_images = []
                st.session_state.embeddings = []
                st.session_state.captured_count = 0
                st.session_state.current_page = "home"
                st.rerun()
            except Exception as e:
                st.error(f"❌ خطأ أثناء الحفظ: {str(e)}")

    # ⚠️ ملاحظة أمنية
    st.markdown("""
        <p style='font-size: 0.9em; margin-top: 30px;'>
            💡 ملاحظة: كلمة المرور الخاصة بك <strong>مشفرة</strong> ولا تُخزن كنص واضح.
            جميع بيانات الوجه تُخزن كقيم رياضية (embeddings) فقط.
        </p>
    """, unsafe_allow_html=True)

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

    # CSS إضافي خاص بصفحة التسجيل
    st.markdown(f"""
        <style>
        /* شريط التقدم */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, #4facfe, #00f2fe);
            border-radius: 10px;
        }}

        .progress-label {{
            font-size: 1.1em;
            margin-top: 10px;
            font-weight: 500;
        }}

        /* عرض الصور المصغرة */
        .thumb-container {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
            padding: 10px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 16px;
        }}

        .thumb-container img {{
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s;
            width: {thumb_size};
            height: {thumb_size};
            object-fit: cover;
        }}

        .thumb-container img:hover {{
            transform: scale(1.1);
            z-index: 10;
            position: relative;
        }}

        /* مؤشر التحقق */
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-left: 8px;
            animation: blink 1.5s infinite;
        }}

        .status-indicator.success {{ background-color: #4CAF50; }}
        .status-indicator.warning {{ background-color: #FFC107; }}
        .status-indicator.error {{ background-color: #F44336; }}

        @keyframes blink {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
        }}
        </style>
    """, unsafe_allow_html=True)