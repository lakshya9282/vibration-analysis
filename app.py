import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Vibration Diagnostics Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("vibration_model.h5")
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le

model, scaler, le = load_assets()

st.title("⚙️ Predictive Maintenance: Vibration Analysis")
st.markdown("Upload a CSV file containing vibration data (`acx`, `acy`, `acz`) to diagnose bearing health.")

# Sidebar for Upload
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.columns = [c.lower().strip() for c in data.columns]
    
    if all(col in data.columns for col in ['acx', 'acy', 'acz']):
        # Display Raw Data
        with st.expander("View Uploaded Data"):
            st.write(data.head())

        # Preprocessing for prediction (Take the first window of 100 samples)
        if len(data) >= 100:
            input_data = data[['acx', 'acy', 'acz']].iloc[:100].values
            input_scaled = scaler.transform(input_data)
            input_reshaped = input_scaled.reshape(1, 100, 3)

            # Prediction
            prediction = model.predict(input_reshaped)
            pred_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            label = le.inverse_transform([pred_class])[0]

            # Dashboard Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detected Condition", label)
            with col2:
                st.metric("Confidence Score", f"{confidence:.2f}%")

            # Status Indicator
            if "Faulty" in label:
                st.error(f" Warning: Maintenance Required! Condition: {label}")
            else:
                st.success(f" System Normal: {label}")

            # Visualization
            st.subheader("Vibration Signature (First 100 Samples)")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(input_data[:, 0], label="AcX", color="blue", alpha=0.7)
            ax.plot(input_data[:, 1], label="AcY", color="green", alpha=0.7)
            ax.plot(input_data[:, 2], label="AcZ", color="red", alpha=0.7)
            ax.legend()
            st.pyplot(fig)
            
        else:
            st.warning("CSV must have at least 100 rows for analysis.")
    else:
        st.error("CSV must contain columns: acx, acy, acz")
else:
    st.info("Awaiting CSV upload...")
