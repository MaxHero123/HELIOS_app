import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler

def resize_sequence(arr, target_length=1200):
    arr = np.atleast_2d(arr)
    n_samples, n_time = arr.shape
    x_old = np.arange(n_time)
    x_new = np.linspace(0, n_time - 1, target_length)
    resized = np.zeros((n_samples, target_length))
    for i in range(n_samples):
        f = interp1d(x_old, arr[i], kind='linear', fill_value="extrapolate")
        resized[i] = f(x_new)
    return resized

def safe_savgol(df1, df2, window_length=3198, polyorder=4):
    def _filter(arr):
        arr = np.atleast_2d(arr)
        n_time = arr.shape[1]
        wl = min(window_length, n_time)
        if wl % 2 == 0:
            wl -= 1
        if wl < 3:
            return arr
        return savgol_filter(arr, window_length=wl, polyorder=polyorder, axis=1, mode='interp')
    return _filter(df1), _filter(df2)

def fourier_fixed(df1, df2):
    def _fft(arr):
        arr = np.atleast_2d(arr)
        n_samples, n_time = arr.shape
        fft_arr = np.abs(np.fft.fft(arr, axis=1))
        fft_arr = fft_arr[:, :n_time]  # truncate to original length
        return fft_arr
    return _fft(df1), _fft(df2)

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
