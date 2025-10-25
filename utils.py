import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler

TARGET_LENGTH = 3917  # must match your model's input

def resize_sequence(X, target_length=TARGET_LENGTH):
    """
    Resize each row of X to have exactly target_length points using linear interpolation.
    """
    X = np.atleast_2d(X)
    resized = np.zeros((X.shape[0], target_length))
    for i in range(X.shape[0]):
        resized[i] = np.interp(
            np.linspace(0, X.shape[1]-1, target_length),
            np.arange(X.shape[1]),
            X[i]
        )
    return resized

def fourier_fixed(df1, df2, target_length=TARGET_LENGTH):
    """
    Apply FFT and take absolute value. Output length is target_length.
    """
    def _fft(arr):
        arr = np.atleast_2d(arr)
        return np.abs(np.fft.fft(arr, n=target_length, axis=1))
    return _fft(df1), _fft(df2)

def safe_savgol_fixed(X, window_length=21, polyorder=3, target_length=TARGET_LENGTH):
    """
    Apply Savitzky–Golay safely row-wise and resize output to target_length.
    """
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
        # Resize to target length
        filtered[i] = np.interp(
            np.linspace(0, len(sg)-1, target_length),
            np.arange(len(sg)),
            sg
        )
    return filtered

def norm(X_train, X_test):
    minval = min(np.min(X_train), np.min(X_test))
    maxval = max(np.max(X_train), np.max(X_test))
    X_train = (X_train - minval) / (maxval - minval)
    X_test = (X_test - minval) / (maxval - minval)
    return X_train, X_test

def robust(df1, df2):
    scaler = RobustScaler()
    X_train = scaler.fit_transform(df1)
    X_test = scaler.transform(df2)
    return X_train, X_test
