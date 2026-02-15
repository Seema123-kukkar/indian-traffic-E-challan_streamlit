import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="RecycleVision Advanced Dashboard",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ RecycleVision - Advanced ML Dashboard")

# ----------------------------------
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_trained_model():
    return load_model("model.h5")

model = load_trained_model()

class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
IMG_SIZE = 256

# ----------------------------------
# IMAGE UPLOAD
# ----------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": prediction
    })

    # ----------------------------------
    # BAR CHART
    # ----------------------------------
    st.subheader("📊 Vertical Bar Chart")
    st.bar_chart(prob_df.set_index("Class"))

    # ----------------------------------
    # HORIZONTAL BAR
    # ----------------------------------
    st.subheader("📊 Horizontal Bar Chart")
    fig = px.bar(prob_df, x="Probability", y="Class", orientation='h')
    st.plotly_chart(fig)

    # ----------------------------------
    # PIE CHART
    # ----------------------------------
    st.subheader("🥧 Pie Chart")
    fig2 = px.pie(prob_df, values="Probability", names="Class")
    st.plotly_chart(fig2)

    # ----------------------------------
    # AREA CHART
    # ----------------------------------
    st.subheader("📈 Area Chart")
    st.area_chart(prob_df.set_index("Class"))

    # ----------------------------------
    # RADAR CHART
    # ----------------------------------
    st.subheader("🕸 Radar Chart")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(
        r=prediction,
        theta=class_names,
        fill='toself'
    ))
    fig3.update_layout(polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig3)

    # ----------------------------------
    # HISTOGRAM
    # ----------------------------------
    st.subheader("📊 Probability Histogram")
    fig4 = px.histogram(prob_df, x="Probability")
    st.plotly_chart(fig4)

    # ----------------------------------
    # BOX PLOT
    # ----------------------------------
    st.subheader("📦 Box Plot")
    fig5 = px.box(prob_df, y="Probability")
    st.plotly_chart(fig5)

# ----------------------------------
# MODEL PERFORMANCE SECTION
# ----------------------------------
st.markdown("---")
st.header("📊 Model Performance Visualization")

# Dummy Data (replace with your real data)
y_true = [0,1,2,3,4,5,0,1,2,3]
y_pred = [0,1,2,3,4,4,0,2,2,3]

# ----------------------------------
# CONFUSION MATRIX
# ----------------------------------
st.subheader("🔥 Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)

fig6, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig6)

# ----------------------------------
# HEATMAP (Random Feature Correlation Demo)
# ----------------------------------
st.subheader("🌡 Feature Correlation Heatmap")

demo_data = pd.DataFrame(np.random.rand(10,5),
                         columns=["F1","F2","F3","F4","F5"])

fig7, ax = plt.subplots()
sns.heatmap(demo_data.corr(), annot=True)
st.pyplot(fig7)

# ----------------------------------
# DOWNLOAD REPORT
# ----------------------------------
st.subheader("📥 Download Prediction Report")

csv = prob_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='prediction_report.csv',
    mime='text/csv'
)
