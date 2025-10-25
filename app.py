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
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        st.success("✅ Input cleaned (NaNs/Infs handled)")

        # Step 1: Resize
        X = resize_sequence(X, target_length=TARGET_LENGTH)
        st.success(f"✅ Resize complete")

        # Step 2: Fourier transform
        _, X = fourier_fixed(X, X, target_length=TARGET_LENGTH)
        st.success(f"✅ Fourier transform complete")

        # Step 3: Normalization
        _, X = norm(X, X)
        st.success(f"✅ Normalization complete")

        # Prepare input for CNN
        X_model = np.expand_dims(X, axis=-1)

        # Predict
        if model:
            preds = model.predict(X_model)
            preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

            # Plot predictions along sequence
            st.subheader("📈 CNN Prediction Along Sequence")
            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(preds.flatten(), label="Prediction")
            threshold = 0.3  # You can adjust this
            ax.axhline(threshold, color='r', linestyle='--', label=f"Threshold={threshold}")
            ax.set_xlabel("Time / Flux Index")
            ax.set_ylabel("Prediction Confidence")
            ax.set_title("Exoplanet Detection Confidence Across Light Curve")
            ax.legend()
            st.pyplot(fig)

            # Highlight overall detection
            max_pred = float(np.max(preds))
            if max_pred >= threshold:
                st.success(f"🌍 Potential Exoplanet Detected! Max Confidence: {max_pred:.2f}")
            else:
                st.info(f"🚫 No Exoplanet Detected. Max Confidence: {max_pred:.2f}")
        else:
            st.error("Model not loaded. Place your model file in 'my_new_exo_model.keras/'")

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
