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
Now with **sliding windows** for better detection of sparse transits.
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
        if len(df) > 20:
            st.warning("Large CSV detected. Processing may take a while...")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = None

# -----------------------
# Run detection
# -----------------------
if uploaded_file and df is not None and st.button("🔍 Run Exoplanet Detection"):
    try:
        X_raw = np.array(df.values, dtype=float)
        if X_raw.ndim == 1:
            X_raw = np.expand_dims(X_raw, axis=0)

        # Clean input
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # -----------------------
        # Parameters for sliding windows
        # -----------------------
        window_size = 500   # can adjust
        step_size = 100     # overlap
        threshold = 0.3     # prediction threshold

        all_preds = []

        for i in range(0, X_raw.shape[1], step_size):
            window = X_raw[:, i:i+window_size]

            if window.shape[1] < 10:
                continue  # skip tiny windows

            # Preprocess window
            window = resize_sequence(window, target_length=TARGET_LENGTH)
            _, window = fourier_fixed(window, window, target_length=TARGET_LENGTH)
            _, window = norm(window, window)
            X_model = np.expand_dims(window, axis=-1)

            # Predict
            if model:
                preds = model.predict(X_model)
                preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
                all_preds.append((i, preds.flatten()))
        
        if not all_preds:
            st.warning("No valid windows found for prediction.")
        else:
            # Combine predictions into a single sequence
            combined_preds = np.zeros(X_raw.shape[1])
            counts = np.zeros(X_raw.shape[1])
            for start, pred in all_preds:
                end = start + len(pred)
                combined_preds[start:end] += pred[:X_raw.shape[1]-start]
                counts[start:end] += 1
            counts[counts==0] = 1
            combined_preds /= counts

            # Plot
            st.subheader("📈 CNN Prediction Along Light Curve")
            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(combined_preds, label="Prediction")
            ax.axhline(threshold, color='r', linestyle='--', label=f"Threshold={threshold}")
            ax.set_xlabel("Time / Flux Index")
            ax.set_ylabel("Prediction Confidence")
            ax.set_title("Exoplanet Detection Confidence Across Light Curve")
            ax.legend()
            st.pyplot(fig)

            # Highlight detection
            max_pred = float(np.max(combined_preds))
            if max_pred >= threshold:
                st.success(f"🌍 Potential Exoplanet Detected! Max Confidence: {max_pred:.2f}")
            else:
                st.info(f"🚫 No Exoplanet Detected. Max Confidence: {max_pred:.2f}")

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
