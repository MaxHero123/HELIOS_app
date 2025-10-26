import os
import sys
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

Pipeline: **Resize → Fourier → Savitzky–Golay → Normalization → Robust Scaling → Sliding-window detection**
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
        X_train, X_test = fourier_fixed(X, X, target_length=TARGET_LENGTH)
        st.success("✅ Fourier transform complete")

        # -----------------------
        # Step 4: Savitzky–Golay smoothing
        # -----------------------
        st.info("Applying Savitzky–Golay smoothing...")
        X_train = safe_savgol_fixed(X_train, target_length=TARGET_LENGTH)
        X_test = safe_savgol_fixed(X_test, target_length=TARGET_LENGTH)
        st.success("✅ Savitzky–Golay complete")

        # -----------------------
        # Step 5: Normalization
        # -----------------------
        st.info("Normalizing data...")
        X_train, X_test = norm(X_train, X_test)
        st.success("✅ Normalization complete")

        # -----------------------
        # Step 6: Robust scaling
        # -----------------------
        st.info("Applying robust scaling...")
        X_train, X_test = robust(X_train, X_test)
        st.success("✅ Robust scaling complete")

        # -----------------------
        # Step 7: Sliding-window prediction
        # -----------------------
        st.info("Running sliding-window detection...")

        window_size = TARGET_LENGTH  # matches model input
        stride = TARGET_LENGTH // 2  # 50% overlap

        max_pred = 0.0
        for start in range(0, X_test.shape[1] - window_size + 1, stride):
            window = X_test[:, start:start + window_size]
            window_input = np.expand_dims(window, axis=-1)
            preds = model.predict(window_input, verbose=0)
            max_pred = max(max_pred, float(np.max(preds)))

        st.write(f"**Maximum model output from all windows:** {max_pred:.3f}")

        if max_pred > 0.5:
            st.success(f"🌍 Exoplanet Detected! Confidence: {max_pred:.2f}")
        else:
            st.info(f"🚫 No Exoplanet Detected. Confidence: {max_pred:.2f}")

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
