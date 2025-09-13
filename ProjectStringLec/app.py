"""
تطبيق فلاتر معالجة الصور والتعرف على الانماط
الجزء الخاص للتشغيل المحلي 

"""
import cv2
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import time
import io
import base64
import os
import time

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="🎥 تطبيق الفلاتر المتقدم",
    page_icon="🎥",
    layout="wide"
)

# تحديد مجلد الحفظ
save_folder = "saved_images"
os.makedirs(save_folder, exist_ok=True)

def crop_center(frame, target_width, target_height):
    h, w = frame.shape[:2]

    # حساب النسبة المطلوبة للإطار
    target_ratio = target_width / target_height
    current_ratio = w / h

    if current_ratio > target_ratio:
        # الصورة أعرض من الإطار → نقتص من الجانبين
        new_w = int(h * target_ratio)
        offset_w = (w - new_w) // 2
        cropped = frame[:, offset_w:offset_w + new_w]
    else:
        # الصورة أطول من الإطار → نقتص من الأعلى والأسفل
        new_h = int(w / target_ratio)
        offset_h = (h - new_h) // 2
        cropped = frame[offset_h:offset_h + new_h, :]

    # إعادة تحجيم الصورة لتتناسب مع الإطار
    return cv2.resize(cropped, (target_width, target_height))

# تعريف الفلاتر
def apply_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(frame, kernel_size=15):
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def apply_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_sepia(frame):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    frame = cv2.transform(frame, kernel)
    return np.clip(frame, 0, 255).astype(np.uint8)

def apply_invert(frame):
    return cv2.bitwise_not(frame)

def apply_sketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray
    blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    inv_blur = 255 - blur
    sketch = cv2.divide(gray, inv_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_emboss(frame):
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
    return cv2.filter2D(frame, -1, kernel)

def apply_lomo(frame):
    frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
    return cv2.applyColorMap(frame, cv2.COLORMAP_OCEAN)

def apply_vignette(frame):
    rows, cols = frame.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols,200)
    kernel_y = cv2.getGaussianKernel(rows,200)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    output = np.empty_like(frame)
    for i in range(3):
        output[:,:,i] = frame[:,:,i] * mask
    return output

def apply_warm(frame):
    return cv2.convertScaleAbs(frame, alpha=1.1, beta=30)

def apply_cool(frame):
    return cv2.applyColorMap(frame, cv2.COLORMAP_BONE)

def apply_contrast(frame):
    return cv2.convertScaleAbs(frame, alpha=2.0, beta=0)

def apply_cartoon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 2)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)

def apply_sepia_modern(frame):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    frame = cv2.transform(frame, kernel)
    return np.clip(frame,0,255)

def apply_hdr(frame):
    return cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)

def apply_fun_warp(frame, amplitude=10, frequency=10):
    h, w = frame.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    map_x = x + amplitude * np.sin(y / frequency)
    map_y = y + amplitude * np.sin(x / frequency)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    warped_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return warped_frame

def apply_color_bubble(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = cv2.add(hsv[:,:,1], 50)  # زيادة التشبع
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_neon_glow(frame):
    blur = cv2.GaussianBlur(frame, (15,15), 0)
    return cv2.addWeighted(frame, 0.6, blur, 0.4, 0)

def apply_pixelate(frame, blocks=20):
    h, w = frame.shape[:2]
    temp = cv2.resize(frame, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w,h), interpolation=cv2.INTER_NEAREST)

def apply_rainbow_invert(frame):
    inverted = cv2.bitwise_not(frame)
    rainbow = cv2.applyColorMap(inverted, cv2.COLORMAP_RAINBOW)
    return cv2.addWeighted(frame, 0.5, rainbow, 0.5, 0)


# تحميل Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# تحميل صورة النظارة مع قناة ألفا
glasses_img = cv2.imread("C:\\Users\\USER\\Desktop\\projectt\\11.png", cv2.IMREAD_UNCHANGED)  # PNG مع alpha
if glasses_img is None:
    st.error("❌ لم يتم العثور على صورة النظارة 8.png أو لا يمكن تحميلها")

else:
    # إذا لم تحتوي الصورة على قناة ألفا، أضف قناة ألفا كاملة
    if glasses_img.shape[2] == 3:
        b, g, r = cv2.split(glasses_img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        glasses_img = cv2.merge([b, g, r, alpha])


# تحميل صورة القبعة مع قناة ألفا
hat_img = cv2.imread("C:\\Users\\USER\\Desktop\\projectt\\16.png", cv2.IMREAD_UNCHANGED)
if hat_img is None:
    st.error("❌ لم يتم العثور على صورة القبعة hat.png أو لا يمكن تحميلها")
else:
    if hat_img.shape[2] == 3:  # إذا الصورة بدون Alpha
        b, g, r = cv2.split(hat_img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        hat_img = cv2.merge([b, g, r, alpha])


# تحميل الخلفية الجديدة
BACKGROUND_PATH = "C:\\Users\\USER\\Desktop\\projectt\\bg.png"
background_img = cv2.imread(BACKGROUND_PATH)


# كائن لطرح الخلفية
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=True)

def apply_virtual_background(frame):
    global background_img, fgbg

    if background_img is None:
        return frame  # إذا ما في خلفية بديلة نرجع الصورة الأصلية

    h, w, _ = frame.shape
    bg_resized = cv2.resize(background_img, (w, h))

    # إنشاء الماسك (الحركة أمام الكاميرا)
    fgmask = fgbg.apply(frame)

    # تحسين الماسك
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    fgmask = cv2.medianBlur(fgmask, 5)

    # الماسك العكسي للخلفية
    inv_mask = cv2.bitwise_not(fgmask)

    # استخراج الشخص
    person = cv2.bitwise_and(frame, frame, mask=fgmask)

    # استخراج الخلفية البديلة
    new_bg = cv2.bitwise_and(bg_resized, bg_resized, mask=inv_mask)

    # دمج الاثنين
    output = cv2.add(person, new_bg)

    return output


# فلتر قبعة التخرج
def apply_hat(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # اجعل عرض القبعة أكبر من الوجه بنسبة 20%
        hat_width = int(w * 1.3)
        hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])

        # تمركز القبعة
        x_offset = x - int((hat_width - w) / 2)  # توسيط القبعة
        y_offset = y - hat_height + int(h / 8)   # فوق الرأس بقليل

        # قص الحدود لو خرجت
        if y_offset < 0:
            hat_height += y_offset
            y_offset = 0
        if x_offset < 0:
            hat_width += x_offset
            x_offset = 0

        # إعادة تحجيم القبعة
        hat_resized = cv2.resize(hat_img, (hat_width, hat_height), interpolation=cv2.INTER_AREA)

        # قناة ألفا
        if hat_resized.shape[2] == 4:
            alpha = hat_resized[:, :, 3] / 255.0
            rgb_hat = hat_resized[:, :, :3]

            y1, y2 = y_offset, y_offset + hat_height
            x1, x2 = x_offset, x_offset + hat_width

            # تأكد من البقاء داخل الصورة
            if y2 > frame.shape[0]:
                y2 = frame.shape[0]
                rgb_hat = rgb_hat[:y2 - y1]
                alpha = alpha[:y2 - y1]
            if x2 > frame.shape[1]:
                x2 = frame.shape[1]
                rgb_hat = rgb_hat[:, :x2 - x1]
                alpha = alpha[:, :x2 - x1]

            # دمج القبعة مع الصورة
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (alpha * rgb_hat[:, :, c] +
                                          (1 - alpha) * frame[y1:y2, x1:x2, c])
    return frame

# فلتر النظارة
def apply_glasses(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # ضبط حجم النظارة بالنسبة للوجه
        glasses_width = w
        glasses_height = int(glasses_width * glasses_img.shape[0] / glasses_img.shape[1])
        y_offset = y + int(h / 4)
        x_offset = x

        # إعادة الحجم
        glasses_resized = cv2.resize(glasses_img, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

        # التأكد من وجود قناة ألفا
        if glasses_resized.shape[2] == 4:
            alpha = glasses_resized[:, :, 3] / 255.0  # قناة الشفافية
            rgb_glasses = glasses_resized[:, :, :3]

            y1, y2 = y_offset, y_offset + glasses_height
            x1, x2 = x_offset, x_offset + glasses_width

            # التأكد من حدود الإطار
            if y2 > frame.shape[0]:
                y2 = frame.shape[0]
                rgb_glasses = rgb_glasses[:y2 - y1]
                alpha = alpha[:y2 - y1]
            if x2 > frame.shape[1]:
                x2 = frame.shape[1]
                rgb_glasses = rgb_glasses[:, :x2 - x1]
                alpha = alpha[:, :x2 - x1]

            # دمج النظارة مع الإطار باستخدام قناة ألفا
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (alpha * rgb_glasses[:, :, c] +
                                          (1 - alpha) * frame[y1:y2, x1:x2, c])
    return frame

# الشريط الجانبي
with st.sidebar:
    st.title("إعدادات الفلاتر والكاميرا")

    # Sliders لتغيير حجم عرض الفيديو
    display_width = st.slider("عرض الفيديو", 320, 1280, 380)
    display_height = st.slider("ارتفاع الفيديو", 240, 960, 640)

    filter_options = {
        "بدون فلتر": None,
        "قبعة 🎩": apply_hat,
        "تدرج الرمادي": apply_grayscale,
        "تمويه Gaussian": apply_gaussian_blur,
        "كشف الحواف": apply_edge_detection,
        "لون سيبيا": apply_sepia,
        "عكس الألوان": apply_invert,
        "رسم قلم رصاص": apply_sketch,
        "نقش بارز": apply_emboss,
        "لومو / ريترو": apply_lomo,
        "تظليل الأطراف (فينيت)": apply_vignette,
        "ألوان دافئة": apply_warm,
        "ألوان باردة": apply_cool,
        "زيادة التباين": apply_contrast,
        "تأثير كرتوني": apply_cartoon,
        "سيبيا حديث": apply_sepia_modern,
        "تأثير HDR": apply_hdr,
        "تشويه ممتع": apply_fun_warp,
        "فقاعات ملونة": apply_color_bubble,
        "نيون": apply_neon_glow,
        "بكسلة": apply_pixelate,
        "قوس قزح عكسي": apply_rainbow_invert,
        "نظارة": apply_glasses,
        "استبدال الخلفية": apply_virtual_background


    }

    selected_filter = st.radio("اختر الفلتر:", list(filter_options.keys()))

# إنشاء العمود الرئيسي + العمود الأيمن
col_main, col_right = st.columns([6, 1])  # 3 للعمود الرئيسي، 1 للعمود الأيمن
st.markdown("""
<style>
div[data-testid="column"]:nth-of-type(2) {
    position: sticky;
    top: 0;
    height: 100vh;
}
</style>
""", unsafe_allow_html=True)


# with col_right:
#     with st.expander("📌 شريط جانبي أيمن", expanded=True):
#         st.radio("اختر فلتر:", ["بدون فلتر", "Grayscale", "Sepia"])
#         st.button("التقاط صورة 📸")
with col_right:
    st.markdown("**📌 شريط جانبي أيمن**")
    st.button("التقاط صورة 📸")

        # زر التقاط الصورة
    if st.button("التقاط صورة 📸", use_container_width=True):
        st.session_state.capture_next = True

    # عرض الصورة الملتقطة
    if st.session_state.get('captured_image') is not None:
        st.image(st.session_state.captured_image, caption="الصورة الملتقطة", use_container_width=True)
        
        # زر حفظ الصورة
        if st.button("💾 حفظ الصورة"):
            # إنشاء اسم فريد لكل صورة لتجنب الاستبدال
            file_name = f"captured_{int(time.time())}.jpg"
            save_path = os.path.join(save_folder, file_name)

            # حفظ الصورة على القرص
            cv2.imwrite(save_path, cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_RGB2BGR))
            st.success(f"تم حفظ الصورة في المجلد: {save_path}")


with col_main:
    #st.markdown('<h3 style="text-align:center;color:#6366f1;">🎥 تطبيق الفلاتر الحي</h3>', unsafe_allow_html=True)
    frame_placeholder = st.empty()  # placeholder للفيديو الحي


    # Placeholder للفيديو
    frame_placeholder = st.empty()

    # إعداد الكاميرا
    if 'cap' not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 380)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 650)

    # حالة الالتقاط
    if 'capture_next' not in st.session_state:
        st.session_state.capture_next = False
        st.session_state.captured_image = None

    # حلقة الفيديو الحي
    while True:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("لا يمكن الوصول إلى الكاميرا.")
            break

        frame = cv2.flip(frame, 1)  # انعكاس أفقي مثل الكاميرا الأمامية

        # تطبيق الفلتر
        if selected_filter != "بدون فلتر":
            frame_filtered = filter_options[selected_filter](frame)
            if len(frame_filtered.shape) == 2:  # إذا كان grayscale
                frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_GRAY2BGR)
        else:
            frame_filtered = frame

        # إعادة تحجيم الفريم للتحكم بالعرض والارتفاع
        frame_resized = cv2.resize(frame_filtered, (display_width, display_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # داخل حلقة الفيديو الحي
        frame_cropped_resized = crop_center(frame_filtered, display_width, display_height)

        frame_rgb = cv2.cvtColor(frame_cropped_resized, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # تحويل للصورة Base64 وعرضها مع الإطار
        buffered = io.BytesIO()
        frame_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        frame_placeholder.markdown(
            f"""
            <div style="
                display:flex; 
                justify-content:center; 
                border: 5px solid #6366f1; 
                padding: 10px; 
                border-radius: 15px; 
                width: fit-content;
                margin: auto;
            ">
                <img src="data:image/jpeg;base64,{img_str}" width="{display_width}" height="{display_height}" />
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # التقاط الصورة عند الضغط على الزر
        if st.session_state.capture_next:
            st.session_state.captured_image = frame_rgb
            st.session_state.capture_next = False

        # تأخير لتحسين الأداء
        time.sleep(0.03)
