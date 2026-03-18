# 🔧 Bearing Fault Detection Dashboard

## 📌 Overview

This project is an **AI-powered bearing fault detection system** that uses deep learning and classical machine learning models to classify bearing conditions based on vibration sensor data.

It also includes an **interactive dashboard** built using Streamlit for real-time prediction and visualization.

---

## 🌐 Live Demo

👉 **Streamlit App:**
https://your-streamlit-app-link.streamlit.app

> ⚠️ Replace the above link with your deployed Streamlit app URL.

---

## 🚀 Features

* ✅ Deep Learning model (CNN + BiLSTM + Attention + SE Block)
* ✅ Classical ML models (AdaBoost, XGBoost)
* ✅ Signal segmentation for time-series data
* ✅ Interactive dashboard for predictions
* ✅ Data visualization (violin plots, signal graphs)
* ✅ Prediction confidence analysis

---

## 🧠 Models Used

### Deep Learning Model

* MultiHead Attention
* 1D CNN
* Squeeze-and-Excitation Block
* Bidirectional LSTM
* Dense layers with dropout & regularization

### Classical Models

* AdaBoost (Decision Tree base)
* XGBoost

---

## 📂 Project Structure

```
project/
│── app.py                  # Streamlit dashboard
│── model.h5               # Trained deep learning model
│── scaler.pkl            # MinMax scaler
│── label_encoder.pkl     # Label encoder
│── data/                 # CSV datasets
│── training_script.py    # Model training code
│── README.md
```

---

## ⚙️ Installation

Install all required dependencies:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow joblib
```

---

## ▶️ Running the Project

### 1. Train the Model

Run your training script:

```bash
python training_script.py
```

This will generate:

* `model.h5`
* `scaler.pkl`
* `label_encoder.pkl`

---

### 2. Launch Dashboard

```bash
streamlit run app.py
```

---

## 📊 How It Works

1. Upload CSV file containing:

   * `acx`, `acy`, `acz` (sensor signals)

2. System:

   * Scales data
   * Segments into windows
   * Runs prediction using trained model

3. Dashboard displays:

   * Signal plots
   * Predicted labels
   * Prediction distribution
   * Confidence scores

---

## 📈 Data Processing

* Normalization using MinMaxScaler
* Sliding window segmentation:

  * Window size: 100
  * Step size: 20

---

## 📉 Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score
* Macro & Weighted averages

---

## 🔬 Visualization

* Violin plots for feature distribution
* Time-series signal graphs
* Prediction distribution charts

---

## 🧪 Advanced Features (Optional)

* Grad-CAM for model interpretability
* Model comparison dashboard
* Real-time IoT sensor integration
* Confusion matrix visualization

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* XGBoost
* Streamlit
* Matplotlib & Seaborn

---

## 📌 Future Improvements

* Deploy on cloud (Streamlit Cloud / AWS)
* Add real-time streaming data
* Improve model explainability
* Optimize model for edge devices

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is for educational and research purposes.

---

## 👨‍💻 Author

Developed as a machine learning project for bearing fault classification and monitoring.

---
