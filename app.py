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
# Load model (exactly like original code)
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

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    fig, ax = plt.subplots()
    ax.plot(df.iloc[:, 0], color="orange")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Flux")
    ax.set_title("Flux Light Curve")
    st.pyplot(fig)

    if st.button("🔍 Run Exoplanet Detection"):
        try:
            X = np.array(df.values, dtype=float)
            if X.ndim == 1:
                X = np.expand_dims(X, axis=0)

            # -----------------------
            # Resize to CNN input length
            # -----------------------
            X = resize_sequence(X, target_length=TARGET_LENGTH)

            # -----------------------
            # Preprocessing
            # -----------------------
            X_train, X_test = fourier_fixed(X, X, target_length=TARGET_LENGTH)
            X_train = safe_savgol_fixed(X_train, target_length=TARGET_LENGTH)
            X_test  = safe_savgol_fixed(X_test, target_length=TARGET_LENGTH)
            X_train, X_test = norm(X_train, X_test)
            X_train, X_test = robust(X_train, X_test)

            # -----------------------
            # Prepare input for CNN
            # -----------------------
            X_model = np.expand_dims(X_test, axis=-1)  # shape = (batch, 3198, 1)

            # -----------------------
            # Predict
            # -----------------------
            if model:
                preds = model.predict(X_model)
                avg_pred = float(np.mean(preds))
                if avg_pred > 0.5:
                    st.success(f"🌍 Exoplanet Detected! Confidence: {avg_pred:.2f}")
                else:
                    st.info(f"🚫 No Exoplanet Detected. Confidence: {avg_pred:.2f}")
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
