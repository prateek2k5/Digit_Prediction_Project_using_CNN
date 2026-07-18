# ==========================================================
# DIGIT RECOGNITION USING CNN (STREAMLIT)
# ==========================================================
#
# This Streamlit application predicts handwritten digits
# (0-9) using a trained CNN model.

# Workflow
# 1. Upload Image
# 2. Image Preprocessing
# 3. CNN Prediction
# 4. Display Result


# ===================== IMPORT LIBRARIES =====================

# Streamlit web application banane ke liye
import streamlit as st

# NumPy arrays aur numerical operations ke liye
import numpy as np

# TensorFlow trained CNN model load karne ke liye
import tensorflow as tf

# PIL image open karne ke liye
from PIL import Image

# OpenCV preprocessing ke liye
import cv2

import pickle

#Load Model
with open('cnn_model.pkl','rb') as f:
    model=pickle.load(f)

# ===================== PAGE CONFIGURATION =====================

# Browser tab ka title aur icon set kar rahe hain.
st.set_page_config(page_title="Digit Recognition using CNN",page_icon="🔢",layout="centered")

# ===================== TITLE =====================

st.title("🔢 Digit Recognition using CNN (MNIST)")
st.write(
"""
Upload a handwritten digit image and the CNN model
will predict the digit.
"""
)

# ===================== IMAGE UPLOADER =====================

# User se handwritten digit image upload karwa rahe hain.
# type sirf inhi formats ko allow karega.
uploaded_file = st.file_uploader(
    "Upload a Digit Image",
    type=["jpg", "jpeg", "png"]
)

# Agar user ne image upload ki hai tabhi niche wala code chalega.
if uploaded_file is not None:

    # Uploaded image ko memory se open kar rahe hain.
    image = Image.open(uploaded_file)

    # Original uploaded image user ko dikha rahe hain.
    st.subheader("Original Image")
    st.image(image, width=200)

    # ===================== IMAGE PREPROCESSING =====================

    # Image ko Grayscale me convert kar rahe hain.
    # CNN ko sirf intensity values chahiye hoti hain, colors nahi.
    image = image.convert("L")

    # PIL Image ko NumPy array me convert kar rahe hain.
    # OpenCV functions NumPy array par hi kaam karte hain.
    image = np.array(image)

    # Gaussian Blur apply kar rahe hain.
    # (5,5) kernel image ka unwanted noise remove karta hai.
    # Isse thresholding aur contour detection improve hoti hai.
    image = cv2.GaussianBlur(image, (5,5), 0)

    # OTSU Thresholding automatically best threshold value choose karti hai.
    # THRESH_BINARY_INV se:
    # Background Black ho jata hai.
    # Digit White ho jata hai.
    # Ye format MNIST dataset jaisa hota hai.
    _, image = cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Agar digit black ho to invert kar do
    image = cv2.bitwise_not(image)

    # ===================== DIGIT EXTRACTION =====================

    # Image ke outer contours detect kar rahe hain.
    # Contour image ke object ki boundary hoti hai.
    contours, _ = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Agar koi contour mila hai to digit ko crop karenge.
    if contours:

        # Sabse bada contour select kar rahe hain.
        # Normally wahi handwritten digit hota hai.
        # Bahut chhote contours hata do
        contours = [c for c in contours if cv2.contourArea(c) > 100]
        if contours:
            c = max(contours, key=cv2.contourArea)

        # Bounding Rectangle digit ki exact position aur size deta hai.
        x, y, w, h = cv2.boundingRect(c)

        # Sirf digit wale area ko crop kar rahe hain.
        # Extra white/black background remove ho jayega.
        image = image[y:y+h, x:x+w]

    # ===================== RESIZE =====================

    # Cropped image ki height aur width nikal rahe hain.
    h, w = image.shape

    # Aspect Ratio maintain karte hue resize karenge.
    # Isse digit stretch ya compress nahi hogi.
    if h > w:
        new_h = 20                  # Height ko 20 pixels set kar rahe hain.
        new_w = int(w * 20 / h)     # Width proportion ke hisaab se calculate hogi.
    else:
        new_w = 20                  # Width ko 20 pixels set kar rahe hain.
        new_h = int(h * 20 / w)     # Height proportion ke hisaab se calculate hogi.

    # Digit ko resize kar rahe hain.
    image = cv2.resize(image, (new_w, new_h))

    # ===================== CREATE MNIST IMAGE =====================

    # 28x28 ka black canvas bana rahe hain.
    # MNIST dataset bhi isi size ki images use karta hai.
    canvas = np.zeros((28,28), dtype=np.uint8)

    # Resize hui digit ko center me place karne ke liye
    # x aur y offset calculate kar rahe hain.
    x = (28 - new_w) // 2
    y = (28 - new_h) // 2

    # Digit ko canvas ke center me paste kar rahe hain.
    canvas[y:y+new_h, x:x+new_w] = image


    # ===================== NORMALIZATION =====================

    # Pixel values ko 0-255 se convert karke 0-1 range me la rahe hain.
    # Neural Network normalized data par better perform karta hai.
    img_array = canvas.astype("float32") / 255.0

    # CNN input shape:
    # 1 = Number of Images
    # 28 = Height
    # 28 = Width
    # 1 = Grayscale Channel
    img_array = img_array.reshape(1, 28, 28, 1)

    # Final processed image display kar rahe hain.
    # Ye wahi image hai jo CNN ko prediction ke liye di jayegi.
    st.subheader("Processed Image")

    # clamp=True image ko proper grayscale me display karta hai.
    st.image(canvas,caption="Processed Image",width=150,clamp=True)

    # ===================== PREDICTION =====================

    # Trained CNN model se uploaded image ka prediction kar rahe hain.
    # Output me 10 probabilities milti hain (Digits 0 se 9 tak).
    prediction = model.predict(img_array, verbose=0)

    # np.argmax() sabse highest probability wali class ka index return karta hai.
    # Wahi index hamara final predicted digit hoga.
    predicted_digit = np.argmax(prediction)

    # Highest probability ko percentage me convert kar rahe hain.
    # Isse pata chalta hai model prediction ko kitna confidently kar raha hai.
    confidence = np.max(prediction) * 100


    # ===================== DISPLAY RESULT =====================

    # Prediction section ko visually separate karne ke liye horizontal line.
    st.markdown("---")

    # Heading display kar rahe hain.
    st.subheader("Prediction Result")

    # Final predicted digit ko green success box me show kar rahe hain.
    st.success(f"Predicted Digit : {predicted_digit}")

    # Prediction ki confidence percentage show kar rahe hain.
    st.info(f"Confidence : {confidence:.2f}%")


    # ===================== TOP 3 PREDICTIONS =====================

    # Top 3 highest probable digits show karenge.
    # Isse user dekh sakta hai ki model kis digit ko second aur third option maan raha tha.
    st.subheader("Top 3 Predictions")

    # Sabhi probabilities ko descending order me sort karke
    # sabse pehli 3 values select kar rahe hain.
    top3 = np.argsort(prediction[0])[::-1][:3]

    # Top 3 digits aur unki confidence percentage display kar rahe hain.
    for i in top3:
        st.write(f"Digit {i} : {prediction[0][i] * 100:.2f}%")


    # ===================== PROBABILITY GRAPH =====================

    # Har digit (0–9) ki prediction probability ko bar chart me show kar rahe hain.
    # Is graph se easily pata chal jata hai ki model kis digit ko
    # sabse zyada aur kis digit ko sabse kam probability de raha hai.
    st.subheader("Prediction Probability")
    st.bar_chart(prediction[0])


    # ===================== FOOTER =====================

    # App ke end me horizontal line.
    st.markdown("---")

    # Developer information display kar rahe hain.
    st.caption("Developed using TensorFlow, OpenCV and Streamlit")
