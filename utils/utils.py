import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import normalize

TARGET_LENGTH = 3197

def fourier(X):
    return np.abs(np.fft.fft(X, axis=1))

def savgol_fixed(X):
    X_filtered = np.zeros_like(X)
    for i in range(X.shape[0]):
        wl = min(21, X.shape[1] if X.shape[1]%2==1 else X.shape[1]-1)
        wl = max(wl, 3)
        X_filtered[i] = savgol_filter(X[i], wl, 4)
    return X_filtered

def norm(X):
    return normalize(X)

def robust(X):
    scaler = RobustScaler()
    return scaler.fit_transform(X)

def preprocess_flux(X):
    X = X[:, :TARGET_LENGTH]  # truncate if longer
    X_fft = fourier(X)
    X_sg = savgol_fixed(X_fft)
    X_norm = norm(X_sg)
    X_scaled = robust(X_norm)
    return X_scaled
