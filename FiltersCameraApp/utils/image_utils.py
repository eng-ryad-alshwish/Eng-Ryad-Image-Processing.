from PIL import Image
import numpy as np

def load_image(uploaded_file, default_path="sample.jpg"):
    """
    رفع صورة أو استخدام الصورة الافتراضية إذا لم يتم رفع أي صورة.
    """
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    else:
        image = Image.open(default_path)
    return image

def to_numpy(image):
    """
    تحويل الصورة PIL إلى مصفوفة NumPy.
    """
    return np.array(image)
