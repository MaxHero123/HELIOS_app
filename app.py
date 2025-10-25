import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import resize_sequence, fourier_fixed, safe_savgol, norm, robust
import streamlit as st
import os

st.set_page_config(page_title="Exoplanet Detection AI", page_icon="🪐", layout="wide")

st.title("🪐 HELIOS (Exoplanet Detection AI)")
st.markdown("""
This web app uses a **1D Convolutional Neural Network (CNN)** trained on flux data 
to detect the presence of **exoplanets** in light curves.

Upload a CSV file containing flux data, and the AI will analyze it using a preprocessing pipeline:
**Resize → Fourier → Savitzky–Golay → Normalization → Robust Scaling**.
""")

# -----------------------
# Load model the way you did
# -----------------------
@st.cache_resource
def load_cnn_model():
    try:
        return load_model("model/my_exo_model.keras")  # folder model
    except Exception as e:
        st.error(f"Model not found or failed to load: {e}")
        return None

model_path = os.path.join(os.path.dirname(__file__), "my_exo_model.keras")
model = load_model(model_path)

# -----------------------
# File uploader
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
            X = np.array(df.values, dtype=float)
            if X.ndim == 1:
                X = np.expand_dims(X, axis=0)

            # Resize to ensure 1200 features for CNN
            X = resize_sequence(X, target_length=1200)

            # Preprocessing pipeline
            X_train, X_test = fourier_fixed(X, X, target_length=1200)
            X_train, X_test = safe_savgol(X_train, X_test)
            X_train, X_test = norm(X_train, X_test)
            X_train, X_test = robust(X_train, X_test)

            # CNN expects (batch, 1200, 1)
            X_model = np.expand_dims(X_test, axis=-1)

            if model:
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
**A Novel Machine Learning Pipeline for High-Accuracy Exoplanet Light-Curve Interpretation**  
- **Developer:** Maximilian Solomon  
- **Model:** 1D CNN trained on NASA Kepler flux data  
- **Libraries:** TensorFlow, NumPy, SciPy, scikit-learn, Streamlit  
""")

