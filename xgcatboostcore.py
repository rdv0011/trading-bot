import pandas as pd
import numpy as np

# =============================================
# Feature engineering
# =============================================
def make_features(df):
    df = df.copy()
    df['ret1'] = df['close'].pct_change(1)
    for l in [1,2,3,5,10]:
        df[f'ret_lag_{l}'] = df['ret1'].shift(l)
    for span in [5,15,60,240]:
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        df[f'ema_diff_{span}'] = df[f'ema_{span}'] - df['close']
    df['tr'] = np.maximum(df['high']-df['low'],
                          np.abs(df['high']-df['close'].shift(1)),
                          np.abs(df['low']-df['close'].shift(1)))
    df['atr14'] = df['tr'].rolling(14).mean()
    df['vol_12'] = df['ret1'].rolling(12).std()
    df['vol_48'] = df['ret1'].rolling(48).std()
    df['hour'] = df.index.hour
    df['dow'] = df.index.weekday
    
    # Create dummy variables with all possible values to ensure consistency
    # Hour: 0-23, DOW: 0-6 (Monday=0, Sunday=6)
    df = pd.get_dummies(df, columns=['hour'], drop_first=True)
    df = pd.get_dummies(df, columns=['dow'], drop_first=True)
    
    # Ensure all hour columns exist (hour_1 through hour_23)
    for h in range(1, 24):
        col = f'hour_{h}'
        if col not in df.columns:
            df[col] = 0
    
    # Ensure all dow columns exist (dow_1 through dow_6)
    for d in range(1, 7):
        col = f'dow_{d}'
        if col not in df.columns:
            df[col] = 0
    
    df = df.dropna()
    return df

# =============================================
# Label generation (future return)
# =============================================
def make_labels(df, H=20):
    df = df.copy()
    df['future_close'] = df['close'].shift(-H)
    df['future_ret'] = (df['future_close'] / df['close']) - 1.0
    df = df.dropna()
    return df

# =============================================
# Adaptive training loop
# =============================================
def adaptive_thresholding(pred_series, num_candles=600, label_window=200):
    if len(pred_series) < num_candles:
        return np.nan, np.nan
    sorted_vals = pred_series.tail(num_candles).sort_values(ascending=False)
    frequency = max(1, int(num_candles / label_window))
    maxima_sort_threshold = sorted_vals.iloc[:frequency].mean()
    minima_sort_threshold = sorted_vals.iloc[-frequency:].mean()
    return maxima_sort_threshold, minima_sort_threshold