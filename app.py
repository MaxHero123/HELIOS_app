import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
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

        # -----------------------
        # Step 1: Clean input
        # -----------------------
        st.info("Cleaning input data...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        st.success("✅ Input cleaned")

        # -----------------------
        # Step 2: Resize
        # -----------------------
        st.info("Resizing sequences...")
        X = resize_sequence(X, target_length=TARGET_LENGTH)
        st.success("✅ Resize complete")

        # -----------------------
        # Step 3: Fourier Transform
        # -----------------------
        st.info("Applying Fourier transform...")
        _, X_test = fourier_fixed(X, X, target_length=TARGET_LENGTH)
        st.success("✅ Fourier transform complete")

        # -----------------------
        # Step 4: Normalization
        # -----------------------
        st.info("Normalizing data...")
        _, X_test = norm(X_test, X_test)
        st.success("✅ Normalization complete")

        # -----------------------
        # Step 5: Prepare for model
        # -----------------------
        X_model = np.expand_dims(X_test, axis=-1)
        st.write("Input shape for model:", X_model.shape)

        # -----------------------
        # Step 6: Predict
        # -----------------------
        if model:
            preds = model.predict(X_model)
            avg_pred = float(np.mean(preds))
            st.write(f"**Model output:** {avg_pred:.3f}")

            if avg_pred > 0.5:
                st.success(f"🌍 Exoplanet Detected! Confidence: {avg_pred:.2f}")
            else:
                st.info(f"🚫 No Exoplanet Detected. Confidence: {avg_pred:.2f}")
        else:
            st.error("Model not loaded properly.")

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
