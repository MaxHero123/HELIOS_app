import numpy as np
from imblearn.over_sampling import SMOTE
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler, normalize

def smote(a, b):
    model = SMOTE()
    X, y = model.fit_resample(a, b)
    return X, y

from scipy.signal import savgol_filter

def savgol(df1, df2, window_length=None, polyorder=4):
    def _safe_savgol(data, window_length, polyorder):
        n = len(data)
        if n < 3:  # too short to smooth
            return data
        # window_length must be odd and <= n
        if window_length is None or window_length > n:
            window_length = n if n % 2 == 1 else n-1
        return savgol_filter(data, window_length, polyorder, deriv=0)
    
    x = _safe_savgol(df1, window_length, polyorder)
    y = _safe_savgol(df2, window_length, polyorder)
    return x, y
def fourier(df1, df2):
    X_train = np.abs(np.fft.fft(df1, axis=1))
    X_test = np.abs(np.fft.fft(df2, axis=1))
    return X_train, X_test

def norm(X_train, X_test):
    minval = np.minimum(np.min(X_train), np.min(X_test))
    maxval = np.maximum(np.max(X_train), np.max(X_test))
    norm_X_train = (X_train - minval) / (maxval - minval)
    norm_X_test = (X_test - minval) / (maxval - minval)
    return norm_X_train, norm_X_test

def robust(df1, df2):
    scaler = RobustScaler()
    X_train = scaler.fit_transform(df1)
    X_test = scaler.transform(df2)
    return X_train, X_test
