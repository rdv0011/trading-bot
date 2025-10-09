import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import MinMaxScaler
from historical_data import get_historical_data

class AdaptiveSVM:
    def __init__(self, mode="classification", kernel="rbf", C=1.0, epsilon=0.1):
        """
        mode: "regression" or "classification"
        kernel: SVM kernel
        C: regularization parameter
        epsilon: epsilon for SVR (ignored in classification)
        """
        self.mode = mode
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.final_model = None
        self.a, self.b = 1, 1  # adaptive weight parameters
    
    def prepare_input(self, data):
        """
        Compute features for SVM: rdp3, rdp6, rdp9, transformed close, volatility.
        Ensures no NaNs in transformed_cp even for short data slices.
        """
        df = pd.DataFrame(index=data.index)
        
        # Relative differences
        df["rdp3"] = data.pct_change(3).fillna(0)
        df["rdp6"] = data.pct_change(6).fillna(0)
        df["rdp9"] = data.pct_change(9).fillna(0)
        
        # Rolling mean for transformed close
        window = min(100, len(data))  # adjust window if data is short
        rolling_mean = data.rolling(window, min_periods=1).mean()
        
        # transformed_cp = close - rolling mean
        df["transformed_cp"] = data - rolling_mean
        
        # Volatility (30-period std of returns)
        vol_window = min(30, len(data))
        df["volatility_30"] = data.pct_change().rolling(vol_window, min_periods=1).std().fillna(0)
        
        return df

    def adaptive_weights(self, n_samples):
        """
        Compute adaptive weights for C and epsilon over training samples
        """
        i = np.arange(1, n_samples + 1)
        C_i = self.C * (2 / (1 + np.exp(self.a - 2 * self.a * i / n_samples)))
        eps_i = self.epsilon * ((1 + np.exp(self.b - 2 * self.b * i / n_samples)) / 2)
        return C_i, eps_i

    def train(self, X):
        feats = self.prepare_input(X)
        X_in = feats[["rdp3", "rdp6", "rdp9", "transformed_cp", "volatility_30"]]

        if self.mode == "regression":
            y = feats["rdp3"]  # continuous target (can be modified)
        else:
            returns = X.pct_change()
            y = np.sign(returns.shift(-1)).fillna(0)
            y[y == 0] = 1  # treat zero-change as up movement

        # Align features and target
        X_in = X_in.loc[y.index]

        # Compute adaptive weights
        C_i, eps_i = self.adaptive_weights(len(X_in))
        C_mean, eps_mean = np.mean(C_i), np.mean(eps_i)

        # Initialize model
        if self.mode == "regression":
            self.final_model = SVR(kernel=self.kernel, C=C_mean, epsilon=eps_mean)
        else:
            self.final_model = SVC(kernel=self.kernel, C=C_mean)

        self.final_model.fit(X_in, y)
        return self.final_model

    def predict(self, X):
        feats = self.prepare_input(X)
        X_in = feats[["rdp3", "rdp6", "rdp9", "transformed_cp", "volatility_30"]]
        preds = self.final_model.predict(X_in)

        # Universal action: regression -> sign, classification -> direct output
        if self.mode == "regression":
            return np.sign(preds)
        return preds

    def check_accuracy(self, Y_pred, y_true):
        return np.mean(np.abs(Y_pred - y_true))

def backtest_adaptive_svm(symbol="BTCUSDT", interval="1h",
                          train_window=2000, predict_window=24, step=24,
                          mode="regression", kernel="rbf", C=1.0, epsilon=0.1,
                          save_path=None):
    df = get_historical_data(
        symbol=symbol,
        interval=interval,
        backtesting_start=datetime(2024, 1, 1),
        backtesting_end=datetime(2025, 1, 1)
    )
    close = df["close"]
    asvm = AdaptiveSVM(mode=mode, kernel=kernel, C=C, epsilon=epsilon)
    
    all_results = []
    total_iters = range(0, len(close) - train_window - predict_window, step)
    total = len(total_iters)
    
    for i, start in enumerate(total_iters, 1):
        progress = f"Progress: {i}/{total} ({i / total * 100:.2f}%)"
        print(progress, end="\r", flush=True)
    
        train_data = close.iloc[start:start + train_window]
        test_data = close.iloc[start + train_window:start + train_window + predict_window]

        # Train model
        asvm.train(train_data)

        # Predict
        preds = asvm.predict(test_data)

        # Action for trading
        action = np.sign(preds) if mode == "regression" else preds

        # Segment DataFrame
        segment = pd.DataFrame({"close": test_data, "prediction": preds}, index=test_data.index)
        segment["action"] = action
        segment["returns_bh"] = segment["close"].pct_change()
        segment["strategy"] = segment["returns_bh"] * segment["action"].shift(1)

        # Compute MAE for this segment
        feats = asvm.prepare_input(close)
        true_vals = feats["rdp3"].iloc[start + train_window : start + train_window + predict_window]  # or your target
        mae = asvm.check_accuracy(preds, true_vals)

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
    drawdown = (1 + hourly_ret).cumprod() - (1 + hourly_ret).cumprod().cummax()
    max_dd = drawdown.min()

    # Long/short trade stats
    num_long = (combined["action"] == 1).sum()
    num_short = (combined["action"] == -1).sum()
    short_long_ratio = num_short / num_long if num_long > 0 else np.nan
    avg_long_return = combined.loc[combined["action"] == 1, "strategy"].mean()
    avg_short_return = combined.loc[combined["action"] == -1, "strategy"].mean()
    pct_profitable = (combined["strategy"] > 0).sum() / combined["strategy"].count()


    print(f"\n\nAverage MAE: {average_mae:.6f}")
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
    results = backtest_adaptive_svm(mode="regression")
    print(results.tail(50))