import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils import safe_savgol_fixed, norm, robust
from io import BytesIO
import base64

# Ensure utils.py is found
sys.path.append(os.path.dirname(__file__))

# -----------------------
# Streamlit page setup
# -----------------------
st.set_page_config(page_title="Exoplanet Detection AI", page_icon="🪐", layout="wide")
st.title("🪐 HELIOS (Exoplanet Detection AI)")
st.markdown("""
This web app uses a **1D CNN** trained on flux FFT data to detect exoplanets.

Pipeline: **FFT → Optional Savitzky–Golay → Normalization → Robust Scaling → Sliding-window detection**
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

        # Detect numeric columns (exclude index or LABEL)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        flux_cols = [c for c in numeric_cols if c.lower() not in ['index','label']]

        if len(flux_cols) == 0:
            st.error("No numeric flux columns found. Please check your CSV.")
            df = None
        else:
            st.info(f"Using {len(flux_cols)} numeric columns as the light curve sequence.")

    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = None

# -----------------------
# Run detection
# -----------------------
if df is not None and st.button("🔍 Run Exoplanet Detection"):

    if model is None:
        st.error("Model is not loaded. Cannot run detection.")
    else:
        try:
            # Use all numeric columns as sequence
            flux_cols = [c for c in df.select_dtypes(include=np.number).columns if c.lower() not in ['index','label']]
            X = df[flux_cols].values.astype(float)
            if X.ndim == 1:
                X = np.expand_dims(X, axis=0)

            st.info(f"Input shape: {X.shape}")

            # -----------------------
            # Step 1: FFT magnitude
            # -----------------------
            st.info("Computing FFT magnitude...")
            X_test = np.abs(np.fft.fft(X, n=X.shape[1], axis=1))
            st.success("✅ FFT complete")

            # -----------------------
            # Step 2: Optional Savitzky–Golay smoothing
            # -----------------------
            st.info("Applying Savitzky–Golay smoothing...")
            X_test = safe_savgol_fixed(X_test, target_length=X_test.shape[1])
            st.success("✅ Savitzky–Golay complete")

            # -----------------------
            # Step 3: Normalization
            # -----------------------
            st.info("Normalizing data...")
            _, X_test = norm(X_test, X_test)
            st.success("✅ Normalization complete")

            # -----------------------
            # Step 4: Robust scaling
            # -----------------------
            st.info("Applying robust scaling...")
            _, X_test = robust(X_test, X_test)
            st.success("✅ Robust scaling complete")

            # -----------------------
            # Step 5: Sliding-window prediction
            # -----------------------
            st.info("Running sliding-window detection...")
            window_size = X_test.shape[1]  # single window for full FFT
            stride = window_size // 2
            max_pred = 0.0

            for row in X_test:
                for start in range(0, row.shape[0] - window_size + 1, stride):
                    window = row[start:start + window_size]
                    window_input = np.expand_dims(window, axis=(0, -1))
                    preds = model.predict(window_input, verbose=0)
                    max_pred = max(max_pred, float(np.max(preds)))

            st.write(f"**Maximum model output from all windows:** {max_pred:.3f}")

            if max_pred > 0.5:
                st.success(f"🌍 Exoplanet Detected! Confidence: {max_pred:.2f}")
            else:
                st.info(f"🚫 No Exoplanet Detected. Confidence: {max_pred:.2f}")

            # -----------------------
            # Optional: Download preprocessed CSV
            # -----------------------
            st.info("Preparing preprocessed FFT data for download...")
            processed_df = pd.DataFrame(X_test, columns=[f'FFT_{i+1}' for i in range(X_test.shape[1])])
            csv_buffer = BytesIO()
            processed_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            b64 = base64.b64encode(csv_data).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="preprocessed_fft.csv">💾 Download Preprocessed FFT CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("✅ Preprocessed FFT CSV ready for download")

        except Exception as e:
            st.error(f"Error while processing data: {e}")

# -----------------------
# About section
# -----------------------
st.markdown("---")
st.subheader("💡 About This Project")
st.markdown("""
**1D CNN trained on FFT of Kepler flux data**  
Developer: Maximilian Solomon  
Libraries: TensorFlow, NumPy, SciPy, scikit-learn, Streamlit
""")
