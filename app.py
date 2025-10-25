import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import resize_sequence, fourier_fixed, norm, TARGET_LENGTH

# Ensure utils.py is found
sys.path.append(os.path.dirname(__file__))

# -----------------------
# Streamlit page setup
# -----------------------
st.set_page_config(page_title="Exoplanet Detection AI", page_icon="🪐", layout="wide")
st.title("🪐 HELIOS (Exoplanet Detection AI)")
st.markdown("""
This web app uses a **1D CNN** trained on flux data to detect exoplanets.

Pipeline: **Resize → Fourier → Normalization**  
Now shows **CNN predictions along the light curve**.
""")

# -----------------------
# Load model
# -----------------------
model_path = os.path.join(os.path.dirname(__file__), "my_new_exo_model.keras")
try:
    model = load_model(model_path)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"Model not found or failed to load: {e}")
    model = None

# -----------------------
# File uploader
# -----------------------
uploaded_file = st.file_uploader("📂 Upload CSV flux data", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = None

# -----------------------
# Run prediction
# -----------------------
if uploaded_file and df is not None and st.button("🔍 Run Predictions"):
    try:
        X_raw = np.array(df.values, dtype=float)
        if X_raw.ndim == 1:
            X_raw = np.expand_dims(X_raw, axis=0)

        # Clean input
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Preprocessing
        X_raw = resize_sequence(X_raw, target_length=TARGET_LENGTH)
        _, X_raw = fourier_fixed(X_raw, X_raw, target_length=TARGET_LENGTH)
        _, X_raw = norm(X_raw, X_raw)

        # Prepare for CNN
        X_model = np.expand_dims(X_raw, axis=-1)

        # Predict
        if model:
            preds = model.predict(X_model)
            preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

            # Plot predictions
            st.subheader("📈 CNN Prediction Across Light Curve")
            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(preds.flatten(), label="CNN Prediction")
            ax.set_xlabel("Flux Index / Time")
            ax.set_ylabel("Prediction Confidence")
            ax.set_title("CNN Output Across Light Curve")
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("Model not loaded. Place 'my_new_exo_model.keras' in this folder.")

    except Exception as e:
        st.error(f"Error while processing data: {e}")

# -----------------------
# About section
# -----------------------
st.markdown("---")
st.subheader("💡 About This Project")
st.markdown("""
**1D CNN trained on Kepler flux data**  
Developer: Maximilian Solomon  
Libraries: TensorFlow, NumPy, SciPy, scikit-learn, Streamlit
""")
