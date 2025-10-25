import os
import sys
import time
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils import resize_sequence, fourier_fixed, safe_savgol_fixed, norm, robust, TARGET_LENGTH

# Ensure utils.py is found
sys.path.append(os.path.dirname(__file__))

# -----------------------
# Streamlit page setup
# -----------------------
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

        # -----------------------
        # Step 0: Clean input
        # -----------------------
        st.info("Cleaning input data...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        st.success("✅ Input cleaned (NaNs/Infs handled)")

        total_steps = 5
        step_counter = 0

        # -----------------------
        # Step 1: Resize
        # -----------------------
        with st.spinner("Step 1/5: Resizing sequences..."):
            X = resize_sequence(X, target_length=TARGET_LENGTH)
        step_counter += 1
        st.success(f"✅ Step {step_counter}/{total_steps}: Resize complete")
        st.write("Resize check:", "NaNs =", np.isnan(X).sum(), "Min =", np.min(X), "Max =", np.max(X))

        # -----------------------
        # Step 2: Fourier transform
        # -----------------------
        with st.spinner("Step 2/5: Fourier transform..."):
            X_train, X_test = fourier_fixed(X, X, target_length=TARGET_LENGTH)
        step_counter += 1
        st.success(f"✅ Step {step_counter}/{total_steps}: Fourier transform complete")
        st.write("Fourier check:", "NaNs =", np.isnan(X_test).sum(), "Min =", np.min(X_test), "Max =", np.max(X_test))

        # -----------------------
        # Step 3: Savitzky-Golay smoothing
        # -----------------------
        with st.spinner("Step 3/5: Savitzky–Golay smoothing..."):
            X_train = safe_savgol_fixed(X_train, target_length=TARGET_LENGTH)
            X_test  = safe_savgol_fixed(X_test, target_length=TARGET_LENGTH)
        step_counter += 1
        st.success(f"✅ Step {step_counter}/{total_steps}: Savitzky–Golay complete")
        st.write("Savitzky-Golay check:", "NaNs =", np.isnan(X_test).sum(), "Min =", np.min(X_test), "Max =", np.max(X_test))

        # -----------------------
        # Step 4: Normalization
        # -----------------------
        with st.spinner("Step 4/5: Normalization..."):
            X_train, X_test = norm(X_train, X_test)
        step_counter += 1
        st.success(f"✅ Step {step_counter}/{total_steps}: Normalization complete")
        st.write("Normalization check:", "NaNs =", np.isnan(X_test).sum(), "Min =", np.min(X_test), "Max =", np.max(X_test))

        # -----------------------
        # Step 5: Robust scaling
        # -----------------------
        with st.spinner("Step 5/5: Robust scaling..."):
            X_train, X_test = robust(X_train, X_test)
        step_counter += 1
        st.success(f"✅ Step {step_counter}/{total_steps}: Robust scaling complete")
        st.write("Robust scaling check:", "NaNs =", np.isnan(X_test).sum(), "Min =", np.min(X_test), "Max =", np.max(X_test))

        # -----------------------
        # Prepare input for CNN
        # -----------------------
        X_model = np.expand_dims(X_test, axis=-1)
        st.write("X_model shape:", X_model.shape)

        # -----------------------
        # Predict
        # -----------------------
        if model:
            with st.spinner("Predicting with model..."):
                preds = model.predict(X_model)
                # Ensure no NaNs propagate
                preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
                avg_pred = float(np.mean(preds))
            if avg_pred > 0.5:
                st.success(f"🌍 Exoplanet Detected! Confidence: {avg_pred:.2f}")
            else:
                st.info(f"🚫 No Exoplanet Detected. Confidence: {avg_pred:.2f}")
            st.success("✅ Prediction complete")
        else:
            st.error("Model not loaded. Place your folder-model in 'my_exo_model.keras/'")

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
