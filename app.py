import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import requests

st.set_page_config(page_title="Traffic E-Challan Detection")

st.title("🚦 Indian Traffic E-Challan Detection")

MODEL_PATH = "model.h5"

# 🔥 PUT YOUR MODEL DIRECT DOWNLOAD LINK HERE
MODEL_URL = "https://your-direct-download-link/model.h5"


@st.cache_resource
def load_trained_model():

    # If model not present → download it
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model... Please wait ⏳")

        try:
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully ✅")
        except:
            st.error("❌ Failed to download model. Check your URL.")
            st.stop()

    return load_model(MODEL_PATH)


model = load_trained_model()

# ----------------------------------
# IMAGE UPLOAD
# ----------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

class_names = ["Helmet", "No Helmet"]
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


    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
