import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler

def resize_sequence(X, target_length=1200):
    X = np.atleast_2d(X)
    resized = np.zeros((X.shape[0], target_length))
    for i in range(X.shape[0]):
        resized[i] = np.interp(np.linspace(0, X.shape[1]-1, target_length), np.arange(X.shape[1]), X[i])
    return resized

def fourier_fixed(df1, df2, target_length=1200):
    def _fft(arr):
        arr = np.atleast_2d(arr)
        return np.abs(np.fft.fft(arr, n=target_length, axis=1))
    return _fft(df1), _fft(df2)

def safe_savgol(df1, df2, window_length=21, polyorder=3):
    def _savgol(arr):
        arr = np.atleast_2d(arr)
        n = arr.shape[1]
        wl = min(window_length, n-1 if n % 2 == 0 else n)
        if wl < 3: return arr
        try:
            return savgol_filter(arr, wl, polyorder, axis=1)
        except ValueError:
            return arr
    return _savgol(df1), _savgol(df2)

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
