import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Indian Traffic E-Challan Detection",
    page_icon="🚦",
    layout="wide"
)

# ----------------------------------
# SIDEBAR
# ----------------------------------
st.sidebar.title("🚦 Navigation")
page = st.sidebar.radio("Go to", ["Home","Analytics"])

# ----------------------------------
# HOME PAGE
# ----------------------------------
if page == "Home":
    st.title("🚦 Indian Traffic E-Challan Detection System")
    st.markdown("""
    ### Advanced Demo Version
    
    This version simulates Helmet / No Helmet detection
    without using a real deep learning model.
    
    ✔ Multi-page dashboard  
    ✔ Interactive charts  
    ✔ Confusion matrix simulation  
    ✔ Downloadable reports  
    """)



# ----------------------------------
# ANALYTICS PAGE
# ----------------------------------
elif page == "Analytics":

    st.title("📊 Model Analytics Dashboard (Simulated)")

    # Simulated Accuracy & Loss
    epochs = list(range(1, 11))
    accuracy = np.linspace(0.6, 0.95, 10)
    loss = np.linspace(1.2, 0.2, 10)

    df = pd.DataFrame({
        "Epoch": epochs,
        "Accuracy": accuracy,
        "Loss": loss
    })

    # Accuracy Chart
    st.subheader("📈 Accuracy Over Epochs")
    st.line_chart(df.set_index("Epoch")["Accuracy"])

    # Loss Chart
    st.subheader("📉 Loss Over Epochs")
    st.line_chart(df.set_index("Epoch")["Loss"])

    # ----------------------------------
    # Confusion Matrix Simulation
    # ----------------------------------
    st.subheader("🔥 Confusion Matrix")

    y_true = np.random.randint(0, 2, 50)
    y_pred = np.random.randint(0, 2, 50)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    cax = ax.matshow(cm)
    plt.colorbar(cax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Helmet", "No Helmet"])
    ax.set_yticklabels(["Helmet", "No Helmet"])

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # ----------------------------------
    # Correlation Heatmap
    # ----------------------------------
    st.subheader("🌡 Feature Correlation Heatmap")

    demo_data = pd.DataFrame(np.random.rand(20, 5),
                             columns=["Speed", "Signal", "Helmet", "Lane", "Time"])

    fig2, ax2 = plt.subplots()
    cax2 = ax2.matshow(demo_data.corr())
    plt.colorbar(cax2)
    ax2.set_xticks(range(len(demo_data.columns)))
    ax2.set_yticks(range(len(demo_data.columns)))
    ax2.set_xticklabels(demo_data.columns, rotation=45)
    ax2.set_yticklabels(demo_data.columns)

    st.pyplot(fig2)
