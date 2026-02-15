import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="ML App", layout="centered")

st.title("🚦 Indian Traffic E-Challan Detection")

# ----------------------------------
# SAFE MODEL LOADING
# ----------------------------------
@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.getcwd(), "model.h5")

    if not os.path.exists(model_path):
        st.error("❌ model.h5 file not found in repository.")
        st.stop()

    return load_model(model_path)

model = load_trained_model()

# ----------------------------------
# IMAGE UPLOAD
# ----------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

class_names = ["Helmet", "No Helmet"]  # change according to your model
IMG_SIZE = 224

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
