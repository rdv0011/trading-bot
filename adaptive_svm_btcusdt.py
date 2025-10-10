import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from historical_data import get_historical_data

class AdaptiveSVM:
    def __init__(self, mode="classification", kernel="rbf", C=1.0, epsilon=0.1, use_volume_features=False):
        """
        mode: "regression" or "classification"
        kernel: SVM kernel
        C: regularization parameter
        epsilon: epsilon for SVR (ignored in classification)
        use_volume_features: whether to include volume-based features
        """
        self.mode = mode
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.use_volume_features = use_volume_features
        
        self.final_model = None
        self.scaler = None
        self.a, self.b = 1, 1  # adaptive weight parameters
    
    def prepare_input(self, df):
        """
        Compute features for SVM: rdp3, rdp6, rdp9, transformed close, volatility, and optionally volume features.
        Ensures no NaNs in transformed_cp even for short data slices.
        """
        close = df["close"]
        feats = pd.DataFrame(index=df.index)
        
        # Relative differences
        feats["rdp3"] = close.pct_change(3).fillna(0)
        feats["rdp6"] = close.pct_change(6).fillna(0)
        feats["rdp9"] = close.pct_change(9).fillna(0)
        
        # Rolling mean for transformed close
        window = min(100, len(close))  # adjust window if data is short
        rolling_mean = close.rolling(window, min_periods=1).mean()
        # transformed_cp = close - rolling mean
        feats["transformed_cp"] = close - rolling_mean
        
        # Volatility (30-period std of returns)
        vol_window = min(30, len(close))
        feats["volatility_30"] = close.pct_change().rolling(vol_window, min_periods=1).std().fillna(0)

        # Optional: volume-based features
        if self.use_volume_features and "volume" in df.columns:
            volume = df["volume"]
            feats["vol_24h_avg"] = volume.rolling(24, min_periods=1).mean().bfill()
            feats["vol_72h_avg"] = volume.rolling(72, min_periods=1).mean().bfill()
            feats["vol_change"] = volume.pct_change().fillna(0)
        
        feats = feats.fillna(0)
        return feats

    def adaptive_weights(self, n_samples):
        """
        Compute adaptive weights for C and epsilon over training samples
        """
        i = np.arange(1, n_samples + 1)
        C_i = self.C * (2 / (1 + np.exp(self.a - 2 * self.a * i / n_samples)))
        eps_i = self.epsilon * ((1 + np.exp(self.b - 2 * self.b * i / n_samples)) / 2)
        return C_i, eps_i

    def train(self, df):
        feats = self.prepare_input(df)

        # Base features
        feat_cols = ["rdp3", "rdp6", "rdp9", "transformed_cp", "volatility_30"]
        # Append volume features if enabled
        if self.use_volume_features and "volume" in df.columns:
            feat_cols += ["vol_24h_avg", "vol_72h_avg", "vol_change"]

        X_in = feats[feat_cols]
        
        if self.mode == "regression":
            y = feats["rdp3"]  # continuous target (can be modified)
        else:
            returns = df["close"].pct_change()
            y = np.sign(returns.shift(-1)).fillna(0)
            y[y == 0] = 1  # treat zero-change as up movement

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_in)

        # Compute adaptive weights
        C_i, eps_i = self.adaptive_weights(len(X_scaled))
        C_mean, eps_mean = np.mean(C_i), np.mean(eps_i)

        # Initialize model
        if self.mode == "regression":
            self.final_model = SVR(kernel=self.kernel, C=C_mean, epsilon=eps_mean)
        else:
            self.final_model = SVC(kernel=self.kernel, C=C_mean)

        self.final_model.fit(X_scaled, y)
        return self.final_model

    def predict(self, df):
        feats = self.prepare_input(df)

        feature_cols = ["rdp3", "rdp6", "rdp9", "transformed_cp", "volatility_30"]
        if self.use_volume_features and "volume" in df.columns:
            feature_cols += ["vol_24h_avg", "vol_72h_avg", "vol_change"]

        X_in = feats[feature_cols].fillna(0)

        if self.scaler is None:
            raise ValueError("Scaler not initialized. Train the model first.")

        X_scaled = self.scaler.transform(X_in)
        preds = self.final_model.predict(X_scaled)

        # Regression returns raw prediction; classification returns class label
        return np.sign(preds) if self.mode == "regression" else preds

    def check_accuracy(self, Y_pred, y_true):
        return np.mean(np.abs(Y_pred - y_true))

def backtest_adaptive_svm(
        symbol="BTCUSDT",
        interval="1h",
        train_window=2000,
        predict_window=24,
        step=24,
        kernel="rbf",
        C=1.0,
        epsilon=0.1,
        high_vol_threshold=0.02,
        use_volume_features=True,
):
    df = get_historical_data(
        symbol=symbol,
        interval=interval,
        backtesting_start=datetime(2024, 1, 1),
        backtesting_end=datetime(2025, 1, 1)
    )

    # Ensure required columns exist
    if "close" not in df.columns:
        raise ValueError("Historical data must contain 'close' column.")
    if use_volume_features and "volume" not in df.columns:
        print("⚠️ Volume data not found — disabling volume features.")
        use_volume_features = False
    
    close = df["close"]

    # Create cached models for reuse
    asvm_models = {
        "regression": AdaptiveSVM(mode="regression", kernel=kernel, C=C, epsilon=epsilon, use_volume_features=use_volume_features),
        "classification": AdaptiveSVM(mode="classification", kernel=kernel, C=C, epsilon=epsilon, use_volume_features=use_volume_features)
    }
    
    all_results = []
    total_iters = range(0, len(close) - train_window - predict_window, step)
    total = len(total_iters)
    
    for i, start in enumerate(total_iters, 1):
        progress = f"Progress: {i}/{total} ({i / total * 100:.2f}%)"
        print(progress, end="\r", flush=True)
    
        train_data = df.iloc[start:start + train_window]
        test_data = df.iloc[start + train_window:start + train_window + predict_window]

        # Compute recent volatility
        recent_vol = train_data["close"].pct_change().rolling(72).std().iloc[-1]
        # Dynamically choose mode
        current_mode = "classification" if recent_vol > high_vol_threshold else "regression"
        asvm = asvm_models[current_mode]

        # Train model
        asvm.train(train_data)
        # Predict
        preds = asvm.predict(test_data)

        # Action for trading
        action = np.sign(preds) if current_mode == "regression" else preds

        # Segment DataFrame
        segment = pd.DataFrame({
            "close": test_data["close"],
            "prediction": preds,
            "action": action
            },
            index=test_data.index
        )
        segment["returns_bh"] = segment["close"].pct_change()
        segment["strategy"] = segment["returns_bh"] * segment["action"].shift(1)

        # Compute MAE for this segment
        feats = asvm.prepare_input(train_data)
        target = feats["rdp3"]
        mae = asvm.check_accuracy(np.sign(preds[:len(target)]), target[-len(preds):]) if len(preds) <= len(target) else np.nan

        # Store segment results
        all_results.append({"segment_df": segment, "mae": mae})


    # 4️⃣ Combine and compute returns globally
    combined = pd.concat([x["segment_df"] for x in all_results]).sort_index()
    combined["returns_bh"] = combined["close"].pct_change()
    combined["strategy"] = combined["returns_bh"] * combined["action"].shift(1)

    combined["cum_bh"] = (1 + combined["returns_bh"]).cumprod()
    combined["cum_strategy"] = (1 + combined["strategy"]).cumprod()

    # Statistics
    average_mae = np.mean([x["mae"] for x in all_results])
    hourly_ret = combined["strategy"].dropna()
    sharpe = np.sqrt(24 * 365) * hourly_ret.mean() / hourly_ret.std() if hourly_ret.std() != 0 else 0
    cum_returns = (1 + hourly_ret).cumprod()
    drawdown = (cum_returns - cum_returns.cummax()) / cum_returns.cummax()
    max_dd = drawdown.min() if not drawdown.empty else 0

    # Long/short trade stats
    num_long = (combined["action"] == 1).sum()
    num_short = (combined["action"] == -1).sum()
    short_long_ratio = num_short / num_long if num_long > 0 else np.nan
    avg_long_return = combined.loc[combined["action"] == 1, "strategy"].mean()
    avg_short_return = combined.loc[combined["action"] == -1, "strategy"].mean()
    pct_profitable = (combined["strategy"] > 0).sum() / combined["strategy"].count()


    print("\n\n📊 Backtest Summary")
    print(f"Asset: {symbol}")
    print(f"Period: {df.index.min().date()} → {df.index.max().date()}")
    print(f"Use Volume Features: {use_volume_features}")
    print(f"Average MAE: {average_mae:.6f}")
    print(f"Final Cumulative Strategy Return: {combined['cum_strategy'].iloc[-1]:.6f}")
    print(f"Final Cumulative Buy & Hold Return: {combined['cum_bh'].iloc[-1]:.6f}")
    print(f"Sharpe Ratio (hourly): {sharpe:.3f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Short/Long Trade Ratio: {short_long_ratio:.4f}")
    print(f"Number of Long Trades: {num_long}")
    print(f"Number of Short Trades: {num_short}")
    print(f"Average Long Return per Trade: {avg_long_return:.6f}")
    print(f"Average Short Return per Trade: {avg_short_return:.6f}")
    print(f"Percent of Profitable Trades: {pct_profitable:.2%}")

    return combined


if __name__ == "__main__":
    results = backtest_adaptive_svm(use_volume_features=True)
    print(results.tail(50))