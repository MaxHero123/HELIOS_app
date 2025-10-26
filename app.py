import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_flux, augment_single_flux, TARGET_LENGTH

st.set_page_config(page_title="HELIOS - Exoplanet Detection AI", page_icon="🪐")
st.title("🪐 HELIOS (Exoplanet Detection AI)")

model_path = "my_exo_model (3).keras"
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Model not found or failed to load: {e}")
    model = None

uploaded_file = st.file_uploader("Upload CSV flux data", type=["csv"])
if uploaded_file and model:
    df = pd.read_csv(uploaded_file)
    flux_cols = [c for c in df.columns if "FLUX" in c.upper()]
    if len(flux_cols) == 0:
        st.error("No flux columns found. CSV must match exoTrain/exoTest format.")
    else:
        X = df[flux_cols].values
        if X.shape[1] < TARGET_LENGTH:
            st.error(f"Sequence too short: {X.shape[1]}. Must be {TARGET_LENGTH} flux points.")
        else:
            # preprocess
            X_proc = preprocess_flux(X)
            
            # if single row, augment like training
            if X_proc.shape[0] == 1:
                X_proc = augment_single_flux(X_proc)

            # reshape for CNN
            X_input = X_proc.reshape(X_proc.shape[0], TARGET_LENGTH, 1)

            # predict on batch, take max confidence
            preds = model.predict(X_input, verbose=0)
            conf = float(np.max(preds))

            st.write(f"**Confidence:** {conf:.3f}")
            if conf > 0.5:
                st.success(f"🌍 Exoplanet Detected! Confidence: {conf:.2f}")
            else:
                st.info(f"🚫 No Exoplanet Detected. Confidence: {conf:.2f}")
