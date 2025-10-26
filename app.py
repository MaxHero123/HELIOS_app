import os
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_flux, TARGET_LENGTH

st.set_page_config(page_title="Exoplanet Detection AI", page_icon="🪐", layout="wide")
st.title("🪐 HELIOS (Exoplanet Detection AI)")

# Load model
model_path = os.path.join(os.path.dirname(__file__), "my_exo_model.keras")
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model = None

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV flux data", type=["csv"])
df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df = None

# Run detection
if df is not None and st.button("Run Exoplanet Detection") and model is not None:
    try:
        # Auto-detect flux columns
        flux_cols = [c for c in df.columns if "FLUX" in c.upper()]
        if len(flux_cols) == 0:
            st.error("No FLUX columns found in CSV")
        else:
            X = df[flux_cols].values.astype(float)

            # Preprocess for CNN
            X_ready = preprocess_flux(X, TARGET_LENGTH)
            X_input = np.expand_dims(X_ready, axis=2)  # shape: (batch, 3197, 1)

            # Predict
            preds = model.predict(X_input, verbose=0)
            max_pred = float(np.max(preds))
            st.write(f"Maximum model confidence: {max_pred:.3f}")
            if max_pred > 0.5:
                st.success(f"🌍 Exoplanet Detected! Confidence: {max_pred:.2f}")
            else:
                st.info(f"🚫 No Exoplanet Detected. Confidence: {max_pred:.2f}")
    except Exception as e:
        st.error(f"Error during processing: {e}")
