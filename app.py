import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils import safe_savgol_fixed
from io import BytesIO
import base64

# -----------------------
# Streamlit setup
# -----------------------
st.set_page_config(page_title="Exoplanet Detection AI", page_icon="🪐", layout="wide")
st.title("🪐 HELIOS (Exoplanet Detection AI)")
st.markdown("""
**1D CNN trained on FFT of raw Kepler flux data**  
Pipeline: **FFT → Optional Savitzky–Golay → Sliding-window detection**
""")

# -----------------------
# Load model
# -----------------------
model_path = os.path.join(os.path.dirname(__file__), "my_exo_model.keras")
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model = None

# -----------------------
# Upload CSV
# -----------------------
uploaded_file = st.file_uploader("Upload CSV flux data", type=["csv"])
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())
        numeric_cols = df.select_dtypes(include=np.number).columns
        flux_cols = [c for c in numeric_cols if c.lower() != 'label']
        if len(flux_cols) == 0:
            st.error("No numeric flux columns found in CSV.")
            df = None
        else:
            st.info(f"Using {len(flux_cols)} numeric columns as light curve sequence")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = None

# -----------------------
# Run detection
# -----------------------
if df is not None and st.button("Run Exoplanet Detection"):
    if model is None:
        st.error("Model is not loaded. Cannot run detection.")
    else:
        try:
            X = df[flux_cols].values.astype(float)
            if X.ndim == 1:
                X = np.expand_dims(X, axis=0)

            # -----------------------
            # FFT magnitude
            # -----------------------
            X_fft = np.abs(np.fft.fft(X, n=X.shape[1], axis=1))
            st.success("FFT complete")

            # -----------------------
            # Optional Savitzky–Golay smoothing
            # -----------------------
            X_fft = safe_savgol_fixed(X_fft, target_length=X_fft.shape[1])
            st.success("Savitzky–Golay complete")

            # -----------------------
            # Prepare input for CNN
            # -----------------------
            X_input = np.expand_dims(X_fft, axis=-1)  # shape: (batch, length, 1)

            # -----------------------
            # Sliding-window prediction
            # -----------------------
            max_pred = 0.0
            for row in X_input:
                preds = model.predict(np.expand_dims(row, axis=0), verbose=0)
                max_pred = max(max_pred, float(np.max(preds)))

            st.write(f"Maximum model output: {max_pred:.3f}")
            if max_pred > 0.5:
                st.success(f"🌍 Exoplanet Detected! Confidence: {max_pred:.2f}")
            else:
                st.info(f"🚫 No Exoplanet Detected. Confidence: {max_pred:.2f}")

            # -----------------------
            # Download preprocessed FFT CSV
            # -----------------------
            processed_df = pd.DataFrame(X_fft, columns=[f'FFT_{i+1}' for i in range(X_fft.shape[1])])
            csv_buffer = BytesIO()
            processed_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            b64 = base64.b64encode(csv_data).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="preprocessed_fft.csv">💾 Download Preprocessed FFT CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during processing: {e}")

# -----------------------
# About
# -----------------------
st.markdown("---")
st.subheader("About")
st.markdown("""
**Developer:** Maximilian Solomon  
**Libraries:** TensorFlow, NumPy, SciPy, Streamlit  
**Model:** 1D CNN trained on FFT of raw Kepler flux data
""")
