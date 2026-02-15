# 🚦 Indian Traffic E-Challan Detection System

An AI-powered web application built using Deep Learning and Streamlit 
to detect traffic violations (Helmet / No Helmet).

---

## 📌 Project Overview

This project uses a Convolutional Neural Network (CNN) model to classify 
images into:

- ✅ Helmet
- ❌ No Helmet

Users can upload an image and receive real-time prediction results 
along with confidence score and visual charts.

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib

---

## 📂 Project Structure

indian-traffic-e-challan_streamlit/
│
├── app.py
├── model.h5
├── requirements.txt
├── README.md

---

## 🚀 Run Locally

1. Clone the repository

   git clone https://github.com/your-username/indian-traffic-e-challan_streamlit.git

2. Navigate to folder

   cd indian-traffic-e-challan_streamlit

3. Install dependencies

   pip install -r requirements.txt

4. Run app

   streamlit run app.py

---

## 🌐 Deploy on Streamlit Cloud

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New App"
4. Select repository
5. Deploy

---

## 📊 Features

- Image Upload
- Real-Time Prediction
- Confidence Score
- Bar Chart Visualization
- Pie Chart Visualization
- Downloadable Prediction Report

---

## 🧠 Model Details

- Model Type: CNN
- Input Size: 224x224
- Output Classes: Helmet / No Helmet
- Activation: Softmax

---

## ⚠️ Important

Ensure `model.h5` is inside the repository before deployment.

If model size > 100MB, host it externally (Google Drive / Hugging Face).

---

## 👨‍💻 Author

Seema  Balasaheb Kukkar
