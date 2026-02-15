import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Indian Traffic E-Challan Detection",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Indian Traffic E-Challan Detection System")
st.write("Upload an image to detect Helmet / No Helmet")

# ----------------------------------
# MODEL LOADING (SAFE)
# ----------------------------------
@st.cache_resource
def load_trained_model():
    model_path = "model.h5"

    if not os.path.exists(model_path):
        st.error("❌ model.h5 not found. Please upload it to the repository.")
        st.stop()

    return load_model(model_path)

model = load_trained_model()
model.save("model.h5")
print("Model saved successfully!")


# ----------------------------------
# IMAGE UPLOAD
# ----------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

class_names = ["Helmet", "No Helmet"]
IMG_SIZE = 224

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    # ----------------------------------
    # Probability DataFrame
    # ----------------------------------
    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": prediction
    })

    # Bar Chart
    st.subheader("📊 Prediction Probability")
    st.bar_chart(prob_df.set_index("Class"))

    # Pie Chart
    st.subheader("🥧 Probability Distribution")
    fig, ax = plt.subplots()
    ax.pie(prediction, labels=class_names, autopct='%1.1f%%')
    ax.axis("equal")
    st.pyplot(fig)

    # Download Report
    csv = prob_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="prediction_report.csv",
        mime="text/csv"
    )
