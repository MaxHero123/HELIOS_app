import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import fourier, norm, robust, smote  # safe_savgol replaces savgol below
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

st.set_page_config(page_title="Exoplanet Detection AI", page_icon="🪐", layout="wide")

st.title("🪐 HELIOS (Exoplanet Detection AI)")
st.markdown("""
This web app uses a **1D Convolutional Neural Network (CNN)** trained on flux data 
to detect the presence of **exoplanets** in light curves.

Upload a CSV file containing flux data, and the AI will analyze it using
a preprocessing pipeline:
**Fourier → Savitzky–Golay → Normalization → Robust Scaling → SMOTE augmentation**.
""")

# -----------------------
# Helper functions
# -----------------------
def safe_savgol(df1, df2, window_length=3198, polyorder=4):
    def _filter(arr):
        arr = np.atleast_2d(arr)  # (samples, time_steps)
        n_time = arr.shape[1]
        wl = min(window_length, n_time)
        if wl % 2 == 0:  # must be odd
            wl -= 1
        if wl < 3:  # too short
            return arr
        return savgol_filter(arr, window_length=wl, polyorder=polyorder, axis=1, mode='interp')
    return _filter(df1), _filter(df2)

def resize_sequence(arr, target_length=1200):
    arr = np.atleast_2d(arr)
    n_samples, n_time = arr.shape
    x_old = np.arange(n_time)
    x_new = np.linspace(0, n_time - 1, target_length)
    resized = np.zeros((n_samples, target_length))
    for i in range(n_samples):
        f = interp1d(x_old, arr[i], kind='linear', fill_value="extrapolate")
        resized[i] = f(x_new)
    return resized

# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_cnn_model():
    try:
        return load_model("model/my_exo_model.keras")
    except Exception as e:
        st.error(f"Model not found or failed to load: {e}")
        return None

import os
model_path = os.path.join(os.path.dirname(__file__), "my_exo_model.keras")
model = load_model(model_path)

# -----------------------
# File upload
# -----------------------
uploaded_file = st.file_uploader("📂 Upload your flux data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    fig, ax = plt.subplots()
    ax.plot(df.iloc[:, 0], color="orange")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Flux Value")
    ax.set_title("Flux Light Curve")
    st.pyplot(fig)

    if st.button("🔍 Run Exoplanet Detection"):
        try:
            # Convert CSV to array
            X = np.array(df.values, dtype=float)
            if X.ndim == 1:
                X = X[np.newaxis, :]
            elif X.ndim == 2 and X.shape[0] == 1:
                X = X

            # Resize/interpolate to match model input
            X = resize_sequence(X, target_length=1200)
            X_train, X_test = X, X

            # Preprocessing pipeline
            X_train, X_test = fourier(X_train, X_test)
            X_train, X_test = safe_savgol(X_train, X_test, window_length=3198)
            X_train, X_test = norm(X_train, X_test)
            X_train, X_test = robust(X_train, X_test)

            # SMOTE: skip if only one class
            y_fake = np.zeros(X_train.shape[0], dtype=int)
            if len(np.unique(y_fake)) > 1:
                try:
                    X_train, y_res = smote(X_train, y_fake)
                except Exception as e:
                    st.warning(f"SMOTE skipped: {e}")
            else:
                st.info("SMOTE skipped (single-class input)")

            # Predict
            if model:
                X_model = np.expand_dims(X_test, axis=-1)
                preds = model.predict(X_model)
                avg_pred = float(np.mean(preds))
                if avg_pred > 0.5:
                    st.success(f"🌍 Exoplanet Detected! Confidence: {avg_pred:.2f}")
                else:
                    st.info(f"🚫 No Exoplanet Detected. Confidence: {avg_pred:.2f}")
            else:
                st.error("Model not loaded. Please upload your trained model to the 'model' folder.")

        except Exception as e:
            st.error(f"Error while processing data: {e}")

st.markdown("---")
st.subheader("💡 About This Project")
st.markdown("""
**A Novel Machine Learning Pipeline for High-Accuracy Exoplanet Light-Curve Interpretation with Optimized Fourier Analysis and SMOTE Synthesis**  
- **Developer:** Maximilian Solomon  
- **Model:** 1D CNN trained on NASA Kepler flux data  
- **Libraries:** TensorFlow, NumPy, SciPy, scikit-learn, imbalanced-learn, Streamlit  
""")
