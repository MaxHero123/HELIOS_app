import os
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils import extract_flux, fft_transform, savgol_smooth, minmax_norm, robust_scale, trim_to_target_length, TARGET_LENGTH
from io import BytesIO
import base64

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="Exoplanet Detection AI", page_icon="🪐", layout="wide")
st.title("🪐 HELIOS (Exoplanet Detection AI)")
st.markdown("""
**1D CNN trained on FFT of preprocessed Kepler flux data**  
Pipeline: **Flux extraction → FFT → Savitzky–Golay → Min-max → RobustScaler → CNN**
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
            # Step 1: Extract flux columns
            X = extract_flux(df)

            # Step 2: FFT
            X_fft = fft_transform(X)

            # Step 3: Savitzky–Golay smoothing
            X_sg = savgol_smooth(X_fft)

            # Step 4: Min-max normalization (optional: use training min/max if available)
            X_norm = minmax_norm(X_sg)

            # Step 5: Robust scaling
            X_scaled, _ = robust_scale(X_norm)

            # Step 6: Trim to CNN target length
            X_ready = trim_to_target_length(X_scaled, TARGET_LENGTH)

            # Step 7: Expand dims for CNN input
            X_input = np.expand_dims(X_ready, axis=2)  # shape: (batch, 3197, 1)

            # Step 8: Predict
            preds = model.predict(X_input, verbose=0)
            max_pred = float(np.max(preds))
            st.write(f"Maximum model confidence: {max_pred:.3f}")
            if max_pred > 0.5:
                st.success(f"🌍 Exoplanet Detected! Confidence: {max_pred:.2f}")
            else:
                st.info(f"🚫 No Exoplanet Detected. Confidence: {max_pred:.2f}")

            # -----------------------
            # Optional: download preprocessed FFT CSV
            processed_df = pd.DataFrame(X_ready, columns=[f'FFT_{i+1}' for i in range(X_ready.shape[1])])
            csv_buffer = BytesIO()
            processed_df.to_csv(csv_buffer, index=False)
            b64 = base64.b64encode(csv_buffer.getvalue()).decode()
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
**Model:** 1D CNN trained on FFT of preprocessed Kepler flux data
""")
