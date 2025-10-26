import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler, normalize

TARGET_LENGTH = 3197  # CNN input length

def preprocess_flux(X, target_length=TARGET_LENGTH):
    """
    Preprocess a flux array to match CNN input:
    - Fill NaNs with 0
    - Resize/interpolate to target length
    - FFT magnitude
    - Savitzky–Golay smoothing
    - Min-max normalization
    - Robust scaling
    """
    X = np.atleast_2d(X).astype(float)

    # Fill NaNs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Resize / interpolate
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
        wl = min(21, len(row) if len(row)%2==1 else len(row)-1)
        wl = max(wl, 3)
        X_sg[i] = savgol_filter(row, wl, 4)

    # Min-max normalization
    minval = np.min(X_sg)
    maxval = np.max(X_sg)
    X_norm = (X_sg - minval) / (maxval - minval + 1e-8)

    # Robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_norm)

    return X_scaled
