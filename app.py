import os
import sys
import time
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils import resize_sequence, fourier_fixed, norm, TARGET_LENGTH  # removed robust and Savitzky-Golay

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
""")

# -----------------------
# Load model
# -----------------------
model_path = os.path.join(os.path.dirname(__file__), "my_new_exo_model_.keras")
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

        # Step 0: Clean input
        st.info("Cleaning input data...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        st.success("✅ Input cleaned (NaNs/Infs handled)")

        total_steps = 3
        step_counter = 0

        # Step 1: Resize
        with st.spinner("Step 1/3: Resizing sequences..."):
            X = resize_sequence(X, target_length=TARGET_LENGTH)
        step_counter += 1
        st.success(f"✅ Step {step_counter}/{total_steps}: Resize complete")

        # Step 2: Fourier transform
        with st.spinner("Step 2/3: Fourier transform..."):
            _, X = fourier_fixed(X, X, target_length=TARGET_LENGTH)
        step_counter += 1
        st.success(f"✅ Step {step_counter}/{total_steps}: Fourier transform complete")

        # Step 3: Normalization
        with st.spinner("Step 3/3: Normalization..."):
            _, X = norm(X, X)
        step_counter += 1
        st.success(f"✅ Step {step_counter}/{total_steps}: Normalization complete")

        # Prepare input for CNN
        X_model = np.expand_dims(X, axis=-1)

        # Predict
        if model:
            with st.spinner("Predicting with model..."):
                preds = model.predict(X_model)
                preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
                avg_pred = float(np.mean(preds))
            if avg_pred > 0.5:
                st.success(f"🌍 Exoplanet Detected! Confidence: {avg_pred:.2f}")
            else:
                st.info(f"🚫 No Exoplanet Detected. Confidence: {avg_pred:.2f}")
            st.success("✅ Prediction complete")
        else:
            st.error("Model not loaded. Place your model file in 'my_exo_model_fft_norm_smote.keras/'")

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
