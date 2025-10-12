import numpy as np
from imblearn.over_sampling import SMOTE
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler, normalize

def smote(a, b):
    model = SMOTE()
    X, y = model.fit_resample(a, b)
    return X, y

def savgol(df1, df2):
    x = savgol_filter(df1, 21, 4, deriv=0)
    y = savgol_filter(df2, 21, 4, deriv=0)
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
