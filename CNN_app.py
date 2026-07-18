# %%writefile app.py (It must be the first line at any cost)
# Ye command current notebook cell ka code 'app.py' file me save karti hai.
# Streamlit app banane aur deploy karne ke liye use hoti hai.

# Streamlit se interactive web app banane ke liye

# NumPy arrays aur numerical operations ke liye

# TensorFlow trained CNN model ko load aur use karne ke liye

# PIL (Python Imaging Library)
# Image  : Uploaded image ko open/process karne ke liye
# ImageOps : Image resize, grayscale aur other transformations ke liye
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# ===================== LOAD MODEL =====================

# Saved .keras model ko load kar rahe hain
# taaki bina dobara training ke prediction kar sake.
model = tf.keras.models.load_model("mnist_cnn.keras")

# ===================== STREAMLIT UI =====================

# Web app ka title display kar rahe hain.
st.title("Digit Recognition using CNN (MNIST)")

# User ko instructions show kar rahe hain.
st.write("Upload an image of a digit (28×28 grayscale)")

# Image upload karne ke liye file uploader.
# Sirf jpg, jpeg aur png images allow hongi.
uploaded_file = st.file_uploader(
    "Choose an image...",
    type="jpg"
)

# Agar user ne image upload ki hai tabhi prediction karo.
if uploaded_file is not None:

    # Uploaded image ko open karke grayscale (L) me convert kar rahe hain.
    image = Image.open(uploaded_file).convert('L')

    # Image ko invert kar rahe hain taaki MNIST format match ho.
    # (White digit on black background)
    image = ImageOps.invert(image)

    # Image ko model ke required size (28x28) me resize kar rahe hain.
    image = image.resize((28, 28))

    # Image ko NumPy array me convert karke reshape kar rahe hain.
    # -1 = Number of images automatically detect hoga.
    # /255.0 = Pixel values ko 0–1 range me normalize kar rahe hain.
    img_array = np.array(image).reshape(-1, 28, 28, 1) / 255.0

    # Uploaded image ko Streamlit app me display kar rahe hain.
    st.image(image, caption="Uploaded Image", width=150)

    # Trained CNN model se prediction kar rahe hain.
    prediction = model.predict(img_array)

    # Sabse highest probability wali class (digit) display kar rahe hain.
    st.success(f"Predicted Digit: {np.argmax(prediction)}")
