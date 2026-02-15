import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="RecycleVision - Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ RecycleVision - Waste Classification Dashboard")

# ----------------------------------
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_trained_model():
    model = load_model("model.h5")  # replace with your model
    return model

model = load_trained_model()

class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
IMG_SIZE = 256

# ----------------------------------
# FILE UPLOADER
# ----------------------------------
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}%")

    # ----------------------------------
    # BAR CHART - Class Probabilities
    # ----------------------------------
    st.subheader("📊 Class Probability Distribution")

    prob_df = pd.DataFrame({
        'Class': class_names,
        'Probability': prediction
    })

    st.bar_chart(prob_df.set_index('Class'))

    # ----------------------------------
    # PIE CHART
    # ----------------------------------
    st.subheader("🥧 Probability Pie Chart")

    fig1, ax1 = plt.subplots()
    ax1.pie(prediction, labels=class_names, autopct='%1.1f%%')
    ax1.axis('equal')
    st.pyplot(fig1)

    # ----------------------------------
    # LINE CHART - Confidence Curve
    # ----------------------------------
    st.subheader("📈 Confidence Line Chart")

    st.line_chart(prob_df.set_index('Class'))

# ----------------------------------
# SAMPLE DATA ANALYSIS SECTION
# ----------------------------------
st.markdown("---")
st.header("📊 Sample Model Performance Analysis")

# Example data (replace with your real training history)
data = pd.DataFrame({
    'Epoch': [1,2,3,4,5],
    'Accuracy': [0.65, 0.75, 0.82, 0.88, 0.92],
    'Loss': [1.2, 0.8, 0.6, 0.4, 0.3]
})

st.subheader("Training Accuracy")
st.line_chart(data.set_index('Epoch')['Accuracy'])

st.subheader("Training Loss")
st.line_chart(data.set_index('Epoch')['Loss'])

st.subheader("Accuracy vs Loss Comparison")
st.bar_chart(data.set_index('Epoch'))
