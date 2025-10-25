import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import resize_sequence, safe_savgol, fourier_fixed, norm, robust
import os

st.set_page_config(page_title="Exoplanet Detection AI", page_icon="🪐", layout="wide")

st.title("🪐 HELIOS (Exoplanet Detection AI)")
st.markdown("""
This web app uses a **1D Convolutional Neural Network (CNN)** trained on flux data 
to detect the presence of **exoplanets** in light curves.

Upload a CSV file containing flux data, and the AI will analyze it using
the preprocessing pipeline:
**Resize → Fourier → Savitzky–Golay → Normalization → Robust Scaling**.
""")

# -----------------------
# Load CNN model
# -----------------------
@st.cache_resource
def load_cnn_model(path):
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"Model not found or failed to load: {e}")
        return None

model_path = os.path.join(os.path.dirname(__file__), "my_exo_model.keras")
model = load_cnn_model(model_path)

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

            # Resize/interpolate to model input length
            X = resize_sequence(X, target_length=1200)
            X_train, X_test = X, X  # same for inference

            # Preprocessing pipeline
            X_train, X_test = fourier_fixed(X_train, X_test)
            X_train, X_test = safe_savgol(X_train, X_test, window_length=3198)
            X_train, X_test = norm(X_train, X_test)
            X_train, X_test = robust(X_train, X_test)

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
                st.error("Model not loaded. Please place your trained model in the 'model' folder.")

        except Exception as e:
            st.error(f"Error while processing data: {e}")

st.markdown("---")
st.subheader("💡 About This Project")
st.markdown("""
**A Novel Machine Learning Pipeline for High-Accuracy Exoplanet Light-Curve Interpretation with Optimized Fourier Analysis**  
- **Developer:** Maximilian Solomon  
- **Model:** 1D CNN trained on NASA Kepler flux data  
- **Libraries:** TensorFlow, NumPy, SciPy, scikit-learn, Streamlit  
""")
