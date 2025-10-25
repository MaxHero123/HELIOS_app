import numpy as np

# Match your CNN training input length
TARGET_LENGTH = 3197  # updated to match the new model

def resize_sequence(X, target_length=TARGET_LENGTH):
    """
    Resize each row of X to exactly target_length points using linear interpolation.
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
    Apply FFT safely and keep length = target_length.
    Returns transformed df1 and df2.
    """
    def _fft(arr):
        arr = np.atleast_2d(arr)
        return np.abs(np.fft.fft(arr, n=target_length, axis=1))
    return _fft(df1), _fft(df2)

def norm(X_train, X_test):
    """
    Normalize both X_train and X_test to range [0, 1] based on min/max of both.
    """
    minval = min(np.min(X_train), np.min(X_test))
    maxval = max(np.max(X_train), np.max(X_test))
    X_train = (X_train - minval) / (maxval - minval)
    X_test  = (X_test - minval) / (maxval - minval)
    return X_train, X_test

