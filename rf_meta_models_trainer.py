import pandas as pd
import numpy as np
import ast
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from historical_data import get_historical_data
from rf_btc_lumibot_strategy import RandomForestBTCStrategy

compute_features = RandomForestBTCStrategy._compute_features
TIMEZONE = "UTC"


# -----------------------------
# Load optimized meta-parameters
# -----------------------------
meta_df = pd.read_csv("optimized_params.csv")
meta_df["optimized_params"] = meta_df["optimized_params"].apply(ast.literal_eval)
params_df = pd.json_normalize(meta_df["optimized_params"])
meta_df = pd.concat([meta_df.drop(columns=["optimized_params"]), params_df], axis=1)

# Ensure datetime
meta_df["window_start"] = pd.to_datetime(meta_df["window_start"])
meta_df["window_end"] = pd.to_datetime(meta_df["window_end"])


# -----------------------------
# Random Forest Training (Refined)
# -----------------------------
def train_meta_models(meta_df: pd.DataFrame, historical_data_func, symbol: str, interval: str, compute_features=None) -> dict:
    if compute_features is None:
        raise RuntimeError("compute_features not provided")

    X_list = []
    y_list = {col: [] for col in params_df.columns}
    feature_names = None

    # Loop through each optimization window
    for _, row in meta_df.iterrows():
        cur_start = pd.to_datetime(row["window_start"])
        cur_end = pd.to_datetime(row["window_end"])
        window_len = cur_end - cur_start

        # Previous window (features come from the *past*, target from the *future*)
        prev_start = cur_start - window_len
        prev_end = cur_start

        df_prev = historical_data_func(symbol, interval, prev_start, prev_end)
        df_features_prev = compute_features(df_prev, current_datetime=prev_end)
        if df_features_prev.empty:
            continue

        # Build a rich summary feature vector
        feat_mean = df_features_prev.mean()
        feat_std = df_features_prev.std()
        feat_last = df_features_prev.iloc[-1]
        feat_diff = df_features_prev.diff().mean()

        combined = pd.concat([feat_mean, feat_std, feat_last, feat_diff], axis=0)
        combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        X_list.append(combined.values)
        feature_names = combined.index if feature_names is None else feature_names

        # Targets = optimized params found *for this window*
        for col in params_df.columns:
            y_list[col].append(row[col])

    # Convert to arrays
    X = np.array(X_list)
    y_dict = {col: np.array(vals) for col, vals in y_list.items()}

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf_models = {}

    for col in params_df.columns:
        y = y_dict[col]
        if len(np.unique(y)) < 2 or np.std(y) < 1e-6:
            print(f"Skipping {col} — target nearly constant.")
            continue

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )

        # Use cross-validation for more stable R² estimates
        scores = cross_val_score(rf, X_scaled, y, cv=min(5, len(y)), scoring="r2")
        rf.fit(X_scaled, y)
        print(f"Trained RF for {col}, R² mean={scores.mean():.4f} ± {scores.std():.4f}")

        # Show top feature importances
        importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
        print(f"Top 5 features for {col}:\n{importances.head(5)}\n")

        rf_models[col] = (rf, scaler, feature_names)

    return rf_models


# -----------------------------
# Predict meta-parameters for next window
# -----------------------------
def predict_next_meta_params(
    rf_models: dict,
    historical_data_func,
    symbol: str,
    interval: str,
    window_end: datetime,
    window_size_days: int = 7
) -> dict:
    window_start = window_end - timedelta(days=window_size_days)
    df_ohlcv = historical_data_func(symbol, interval, window_start, window_end)
    df_features = compute_features(df_ohlcv, current_datetime=window_end)
    if df_features.empty:
        return {}

    feat_mean = df_features.mean()
    feat_std = df_features.std()
    feat_last = df_features.iloc[-1]
    feat_diff = df_features.diff().mean()
    combined = pd.concat([feat_mean, feat_std, feat_last, feat_diff], axis=0)
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_new = combined.values.reshape(1, -1)

    predictions = {}
    for col, (model, scaler, feature_names) in rf_models.items():
        X_scaled = scaler.transform(X_new)
        predictions[col] = float(model.predict(X_scaled)[0])

    return predictions


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    rf_models = train_meta_models(
        meta_df,
        get_historical_data,
        symbol="BTCUSDT",
        interval="1h",
        compute_features=compute_features,
    )

    next_params = predict_next_meta_params(
        rf_models,
        get_historical_data,
        symbol="BTCUSDT",
        interval="1h",
        window_end=datetime.utcnow(),
        window_size_days=7,
    )

    print("\nPredicted meta-parameters for next window:")
    for k, v in next_params.items():
        print(f"  {k}: {v:.6f}")
