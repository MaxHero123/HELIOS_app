import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler

# Match your CNN training input length
TARGET_LENGTH = 3198  

# -----------------------
# Resize sequence
# -----------------------
def resize_sequence(X, target_length=TARGET_LENGTH):
    """Resize each row of X to exactly target_length points using linear interpolation."""
    X = np.atleast_2d(X)
    resized = np.zeros((X.shape[0], target_length))
    for i in range(X.shape[0]):
        resized[i] = np.interp(
            np.linspace(0, X.shape[1]-1, target_length),
            np.arange(X.shape[1]),
            X[i]
        )
    return resized

# -----------------------
# Fourier transform
# -----------------------
def fourier_fixed(X1, X2, target_length=TARGET_LENGTH):
    """Apply FFT safely and return magnitude, length = target_length"""
    def _fft(arr):
        arr = np.atleast_2d(arr)
        fft_vals = np.fft.fft(arr, n=target_length, axis=1)
        return np.abs(fft_vals)
    return _fft(X1), _fft(X2)

# -----------------------
# Savitzky–Golay smoothing
# -----------------------
def safe_savgol_fixed(X, window_length=21, polyorder=3, target_length=TARGET_LENGTH):
    """Apply Savitzky–Golay safely and resize output to target_length"""
    X = np.atleast_2d(X)
    filtered = np.zeros((X.shape[0], target_length))
    for i in range(X.shape[0]):
        row = X[i]
        wl = min(window_length, len(row) if len(row)%2==1 else len(row)-1)
        wl = max(wl, 3)
        try:
            sg = savgol_filter(row, wl, polyorder)
        except ValueError:
            sg = row
        filtered[i] = np.interp(
            np.linspace(0, len(sg)-1, target_length),
            np.arange(len(sg)),
            sg
        )
    return filtered

# -----------------------
# Normalization
# -----------------------
def norm(X_train, X_test):
    """Min-max normalization across both arrays"""
    minval = min(np.min(X_train), np.min(X_test))
    maxval = max(np.max(X_train), np.max(X_test))
    if maxval - minval == 0:
        # Avoid division by zero
        X_train_norm = np.zeros_like(X_train)
        X_test_norm = np.zeros_like(X_test)
    else:
        X_train_norm = (X_train - minval) / (maxval - minval)
        X_test_norm = (X_test - minval) / (maxval - minval)
    return X_train_norm, X_test_norm

# -----------------------
# Robust scaling
# -----------------------
def robust(X1, X2):
    """Apply RobustScaler to both arrays, fit on X1"""
    scaler = RobustScaler()
    X1_scaled = scaler.fit_transform(X1)
    X2_scaled = scaler.transform(X2)
    return X1_scaled, X2_scaled
