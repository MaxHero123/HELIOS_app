import os
import joblib
import numpy as np
from scipy.signal import savgol_filter

# CNN input length used in training
TARGET_LENGTH = 3197  

# Load saved scalers and min/max values
base_path = os.path.dirname(__file__)
robust_scaler = joblib.load(os.path.join(base_path, "robust_scaler.pkl"))
train_min = np.load(os.path.join(base_path, "train_min.npy"))
train_max = np.load(os.path.join(base_path, "train_max.npy"))

def preprocess_flux(X, target_length=TARGET_LENGTH):
    """
    Preprocess flux data for CNN model.
    
    Steps:
    1. Convert to 2D array and replace NaNs/Infs with 0
    2. Resize if not already the target length
    3. Apply FFT
    4. Savitzky–Golay smoothing
    5. Min-max normalization using training min/max
    6. Robust scaling using saved scaler
    """
    X = np.atleast_2d(X).astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Resize if necessary
    if X.shape[1] != target_length:
        X_resized = np.zeros((X.shape[0], target_length))
        for i in range(X.shape[0]):
            X_resized[i] = np.interp(
                np.linspace(0, X.shape[1]-1, target_length),
                np.arange(X.shape[1]),
                X[i]
            )
        X = X_resized

    # FFT
    X_fft = np.abs(np.fft.fft(X, axis=1))

    # Savitzky–Golay smoothing
    X_sg = np.zeros_like(X_fft)
    for i in range(X_fft.shape[0]):
        row = X_fft[i]
        wl = min(21, len(row) if len(row) % 2 == 1 else len(row)-1)
        wl = max(wl, 3)
        X_sg[i] = savgol_filter(row, wl, 4)

    # Min-max normalization
    X_norm = (X_sg - train_min) / (train_max - train_min + 1e-8)

    # Robust scaling
    X_scaled = robust_scaler.transform(X_norm)

    return X_scaled
