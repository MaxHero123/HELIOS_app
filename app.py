import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Ensure utils.py is found
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from utils import preprocess_flux, TARGET_LENGTH

# -----------------------
# Streamlit page setup
# -----------------------
st.set_page_config(page_title="HELIOS - Exoplanet Detection AI", page_icon="🪐", layout="wide")
st.title("🪐 HELIOS (Exoplanet Detection AI)")
st.markdown("""
This web app uses a **1D CNN** trained on flux data to detect exoplanets.

Pipeline: **FFT → Savitzky–Golay → Min-max → Robust Scaling → Sliding-window detection**
""")

# -----------------------
# Load model
# -----------------------
model_path = os.path.join(os.path.dirname(__file__), "my_exo_model (3).keras")
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
        # Check for flux columns
        flux_cols = [c for c in df.columns if "FLUX" in c.upper()]
        if len(flux_cols) == 0:
            st.error("No flux columns found. Ensure CSV is from exoTrain/exoTest dataset.")
        else:
            # Extract flux values
            X = df[flux_cols].values
            X = np.atleast_2d(X)

            st.info("Preprocessing flux data...")
            X_processed = preprocess_flux(X, TARGET_LENGTH)
            st.success("✅ Preprocessing complete")

            # Sliding-window detection (optional, use full sequence here)
            X_input = np.expand_dims(X_processed, axis=2)  # shape: (samples, length, 1)
            preds = model.predict(X_input, verbose=0)
            max_pred = float(np.max(preds))

            st.write(f"**Maximum model output (confidence):** {max_pred:.3f}")

            if max_pred > 0.5:
                st.success(f"🌍 Exoplanet Detected! Confidence: {max_pred:.2f}")
            else:
                st.info(f"🚫 No Exoplanet Detected. Confidence: {max_pred:.2f}")

    except Exception as e:
        st.error(f"Error during processing: {e}")

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
