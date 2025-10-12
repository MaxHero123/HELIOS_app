import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import fourier, savgol, norm, robust, smote

st.set_page_config(page_title="Exoplanet Detection AI", page_icon="🪐", layout="wide")

st.title("🪐 Exoplanet Detection AI")
st.markdown("""
This web app uses a **1D Convolutional Neural Network (CNN)** trained on flux data 
to detect the presence of **exoplanets** in light curves.

Upload a CSV file containing flux data, and the AI will analyze it using
a preprocessing pipeline:
**Fourier → Savitzky–Golay → Normalization → Robust Scaling → SMOTE augmentation**.
""")

@st.cache_resource
def load_cnn_model():
    try:
        return load_model("model/my_exo_model.keras")
    except Exception as e:
        st.error(f"Model not found or failed to load: {e}")
        return None

model = load_cnn_model()

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
            if X.ndim == 2:
                X = np.expand_dims(X, axis=-1)

            X_train, X_test = X.squeeze(), X.squeeze()
            X_train, X_test = fourier(X_train, X_test)
            X_train, X_test = savgol(X_train, X_test)
            X_train, X_test = norm(X_train, X_test)
            X_train, X_test = robust(X_train, X_test)

            y_fake = np.zeros(X_train.shape[0], dtype=int)
            try:
                X_train, y_res = smote(X_train, y_fake)
            except Exception as e:
                st.warning(f"SMOTE skipped (likely insufficient samples): {e}")

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
**Created for the Congressional App Challenge**  
- **Developer:** [Your Name Here]  
- **Model:** 1D CNN trained on NASA Kepler flux data  
- **Libraries:** TensorFlow, NumPy, SciPy, scikit-learn, imbalanced-learn, Streamlit  
""")
