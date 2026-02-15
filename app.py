import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Indian Traffic E-Challan Detection (Demo)",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Indian Traffic E-Challan Detection System (Demo Version)")
st.write("Upload an image to simulate Helmet / No Helmet detection")

# ----------------------------------
# IMAGE UPLOAD
# ----------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

class_names = ["Helmet", "No Helmet"]

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ----------------------------------
    # DEMO PREDICTION (Random Simulation)
    # ----------------------------------
    helmet_prob = random.uniform(0.3, 0.9)
    no_helmet_prob = 1 - helmet_prob

    prediction = [helmet_prob, no_helmet_prob]

    predicted_class = class_names[np.argmax(prediction)]
    confidence = max(prediction) * 100

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

    # Line Chart
    st.subheader("📈 Probability Trend")
    st.line_chart(prob_df.set_index("Class"))

    # Download Report
    csv = prob_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="prediction_report.csv",
        mime="text/csv"
    )

st.markdown("---")
st.info("⚠ This is a demo version. No real model is being used.")
