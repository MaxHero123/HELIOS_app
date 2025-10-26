import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler, normalize

# CNN expected input length
TARGET_LENGTH = 3197  

def extract_flux(df):
    """
    Extract all columns starting with 'FLUX' in order.
    Returns a 2D numpy array.
    """
    flux_cols = [c for c in df.columns if c.startswith('FLUX')]
    if len(flux_cols) == 0:
        raise ValueError("No FLUX columns found in CSV")
    return df[flux_cols].values.astype(float)

def fft_transform(X):
    """
    Compute FFT magnitude along axis=1
    """
    return np.abs(np.fft.fft(X, axis=1))

def savgol_smooth(X, window_length=21, polyorder=4):
    """
    Apply Savitzky–Golay smoothing safely
    """
    X = np.atleast_2d(X)
    smoothed = np.zeros_like(X)
    for i in range(X.shape[0]):
        row = X[i]
        wl = min(window_length, len(row) if len(row)%2==1 else len(row)-1)
        wl = max(wl, 3)
        smoothed[i] = savgol_filter(row, wl, polyorder)
    return smoothed

def minmax_norm(X, reference_min=None, reference_max=None):
    """
    Min-max normalize to 0-1
    If reference_min/max provided, use those (for train/test consistency)
    """
    X = np.atleast_2d(X)
    if reference_min is None:
        reference_min = np.min(X)
    if reference_max is None:
        reference_max = np.max(X)
    normed = (X - reference_min) / (reference_max - reference_min + 1e-8)
    return normed

def robust_scale(X, scaler=None):
    """
    Apply RobustScaler. If scaler not provided, fit a new one.
    Returns scaled X and scaler object.
    """
    if scaler is None:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler

def trim_to_target_length(X, target_length=TARGET_LENGTH):
    """
    Trim or interpolate X to target_length.
    Here we simply slice if longer; you can modify if interpolation needed.
    """
    X = np.atleast_2d(X)
    if X.shape[1] > target_length:
        X_trimmed = X[:, :target_length]
    elif X.shape[1] < target_length:
        # linear interpolation if shorter
        X_trimmed = np.zeros((X.shape[0], target_length))
        for i in range(X.shape[0]):
            X_trimmed[i] = np.interp(
                np.linspace(0, X.shape[1]-1, target_length),
                np.arange(X.shape[1]),
                X[i]
            )
    else:
        X_trimmed = X
    return X_trimmed
