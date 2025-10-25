import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import resize_sequence, fourier_fixed, safe_savgol_fixed, norm, robust, TARGET_LENGTH

# Ensure utils.py is found
sys.path.append(os.path.dirname(__file__))

st.set_page_config(page_title="Exoplanet Detection AI", page_icon="🪐", layout="wide")
st.title("🪐 HELIOS (Exoplanet Detection AI)")
st.markdown("""
This web app uses a **1D CNN** trained on flux data to detect exoplanets.

Pipeline: **Resize → Fourier → Savitzky–Golay → Normalization → Robust Scaling**
""")

# -----------------------
# Load model
# -----------------------
model_path = os.path.join(os.path.dirname(__file__), "my_exo_model.keras")
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Model not found or failed to load: {e}")
    model = None

# -----------------------
# File uploader
# -----------------------
uploaded_file = st.file_uploader("📂 Upload CSV flux data", type=["csv"])

df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head())

        if len(df) > 20:
            st.warning("Large CSV detected. Processing may take a while...")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = None

# -----------------------
# Run detection
# -----------------------
if df is not None and st.button("🔍 Run Exoplanet Detection"):
    try:
        X = np.array(df.values, dtype=float)
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)

        # Show progress
        progress_text = st.info("Processing: Resizing sequences...")
        X = resize_sequence(X, target_length=TARGET_LENGTH)
        progress_text.info("✅ Resize complete.")

        progress_text.info("Processing: Fourier transform...")
        X_train, X_test = fourier_fixed(X, X, target_length=TARGET_LENGTH)
        progress_text.info("✅ Fourier transform complete.")

        progress_text.info("Processing: Savitzky–Golay smoothing...")
        X_train = safe_savgol_fixed(X_train, target_length=TARGET_LENGTH)
        X_test  = safe_savgol_fixed(X_test, target_length=TARGET_LENGTH)
        progress_text.info("✅ Savitzky–Golay complete.")

        progress_text.info("Processing: Normalization...")
        X_train, X_test = norm(X_train, X_test)
        progress_text.info("✅ Normalization complete.")

        progress_text.info("Processing: Robust scaling...")
        X_train, X_test = robust(X_train, X_test)
        progress_text.info("✅ Robust scaling complete.")

        # Prepare input for CNN
        X_model = np.expand_dims(X_test, axis=-1)

        # Predict
        if model:
            progress_text.info("Predicting with model...")
            preds = model.predict(X_model)
            avg_pred = float(np.mean(preds))
            if avg_pred > 0.5:
                st.success(f"🌍 Exoplanet Detected! Confidence: {avg_pred:.2f}")
            else:
                st.info(f"🚫 No Exoplanet Detected. Confidence: {avg_pred:.2f}")
            progress_text.info("✅ Prediction complete.")
        else:
            st.error("Model not loaded. Place your folder-model in 'my_exo_model.keras/'")

    except Exception as e:
        st.error(f"Error while processing data: {e}")

st.markdown("---")
st.subheader("💡 About This Project")
st.markdown("""
**1D CNN trained on Kepler flux data**  
Developer: Maximilian Solomon  
Libraries: TensorFlow, NumPy, SciPy, scikit-learn, Streamlit
""")
