import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import normalize

TARGET_LENGTH = 3197

def fourier(X):
    return np.abs(np.fft.fft(X, n=TARGET_LENGTH, axis=1))

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
    X = X[:, :TARGET_LENGTH]  # truncate
    X_fft = fourier(X)
    X_sg = savgol_fixed(X_fft)
    X_norm = norm(X_sg)
    X_scaled = robust(X_norm)
    return X_scaled

def augment_single_flux(X, repeats=5, noise_level=1e-4):
    """
    Duplicate the row multiple times and add tiny noise
    to mimic training augmentation.
    """
    X_batch = np.tile(X, (repeats,1))
    X_batch += np.random.normal(0, noise_level, X_batch.shape)
    return X_batch
