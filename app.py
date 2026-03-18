import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# LOAD MODEL + OBJECTS
# ==========================
model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="Bearing Fault Detection", layout="wide")

st.title("🔧 Bearing Fault Detection Dashboard")

# ==========================
# FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]

    st.subheader(" Raw Data")
    st.write(df.head())

    # ==========================
    # VISUALIZATION
    # ==========================
    st.subheader(" Signal Visualization")

    fig, ax = plt.subplots(3, 1, figsize=(10, 6))
    for i, col in enumerate(['acx', 'acy', 'acz']):
        ax[i].plot(df[col])
        ax[i].set_title(col)
    st.pyplot(fig)

    # ==========================
    # PREPROCESS
    # ==========================
    features = scaler.transform(df[['acx', 'acy', 'acz']])

    def segment_signal(data, window_size=100, step_size=20):
        segments = []
        for i in range(0, len(data) - window_size + 1, step_size):
            segments.append(data[i:i + window_size])
        return np.array(segments)

    X = segment_signal(features)

    # ==========================
    # PREDICTION
    # ==========================
    preds = model.predict(X)
    pred_classes = np.argmax(preds, axis=1)

    decoded_preds = label_encoder.inverse_transform(pred_classes)

    st.subheader(" Predictions")

    pred_df = pd.DataFrame(decoded_preds, columns=["Predicted Label"])
    st.write(pred_df.value_counts())

    # ==========================
    # DISTRIBUTION PLOT
    # ==========================
    st.subheader(" Prediction Distribution")

    fig2, ax2 = plt.subplots()
    sns.countplot(x=decoded_preds, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # ==========================
    # CONFIDENCE
    # ==========================
    st.subheader(" Prediction Confidence")

    confidence = np.max(preds, axis=1)
    st.write(pd.DataFrame(confidence, columns=["Confidence"]).describe())

else:
    st.info("Please upload a CSV file to begin.")
