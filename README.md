# AdaptiveSVM Backtesting Framework

This repository implements an adaptive SVM (Support Vector Machine) for trading strategies on assets like BTCUSDT or other financial instruments. It supports regression and classification modes, optional volume-based features, and dynamic mode switching based on market volatility.

---

## 📊 Backtest Summary Metrics

After running a backtest, the framework prints a summary with the following items:

| Metric                                 | Explanation                                                                                                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Average MAE**                        | Mean Absolute Error between predicted and actual returns (or target metric). Lower values indicate more accurate predictions.                          |
| **Final Cumulative Strategy Return**   | Total compounded return of the strategy over the backtest period. Example: `1.7465` → +74.65% growth.                                                  |
| **Final Cumulative Buy & Hold Return** | Total compounded return if you simply bought the asset at the start and held it. Useful as a baseline.                                                 |
| **Sharpe Ratio (hourly)**              | Measures risk-adjusted performance: mean strategy return divided by its standard deviation. Higher values indicate better return per unit of risk.     |
| **Max Drawdown**                       | Largest peak-to-trough decline in the strategy’s cumulative equity. Shows potential downside risk.                                                     |
| **Short/Long Trade Ratio**             | Number of short trades divided by the number of long trades. Indicates whether the strategy is biased toward shorting or buying.                       |
| **Number of Long Trades**              | Total trades with a positive action signal.                                                                                                            |
| **Number of Short Trades**             | Total trades with a negative action signal.                                                                                                            |
| **Average Long Return per Trade**      | Mean return for trades where the strategy went long.                                                                                                   |
| **Average Short Return per Trade**     | Mean return for trades where the strategy went short.                                                                                                  |
| **Percent of Profitable Trades**       | Fraction of trades that were profitable. Note: This may be ~50% even when cumulative returns are high if winning trades are larger than losing trades. |

---

## ⚙️ Parameter Explanations & Guidance

| Parameter             | Explanation                                                                 | Guidance                                                                                                                                                                                   |
| --------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `interval`            | Candle/bar interval for historical data (`"1h"`, `"15m"`, `"1d"`, etc.)     | Choose based on your desired trading frequency. `1h` works well for swing trading, while `1d` is suitable for long-term strategies. Shorter intervals increase data points but also noise. |
| `train_window`        | Number of past bars used for model training                                 | Larger windows provide more data for training but may include outdated patterns. Typical values: 500–3000 for hourly BTC data.                                                             |
| `predict_window`      | Number of bars the model predicts into the future                           | Defines the prediction horizon. Shorter horizons (1–24h) are safer for high-frequency predictions; longer horizons may reduce accuracy.                                                    |
| `step`                | Step size to move the training window forward                               | Smaller steps increase computational cost but allow more frequent model updates. `step = predict_window` is common.                                                                        |
| `kernel`              | SVM kernel type (`"rbf"`, `"linear"`, `"poly"`)                             | `"rbf"` is versatile and handles non-linear patterns well. `"linear"` may be sufficient for simpler datasets. Test empirically.                                                            |
| `C`                   | SVM regularization parameter                                                | Controls trade-off between training accuracy and overfitting. Larger `C` fits training data more closely; smaller `C` generalizes better. Start with `C=1.0`.                              |
| `epsilon`             | SVR epsilon parameter (ignored in classification)                           | Defines tolerance for error in regression. Smaller `epsilon` tries to predict closer to actual values. Typically 0.01–0.1 for financial data.                                              |
| `high_vol_threshold`  | Volatility threshold to dynamically switch mode                             | Determines when to switch from regression to classification based on recent market volatility. Tune based on asset behavior; e.g., `0.02` for BTC hourly returns.                          |
| `use_volume_features` | Include optional volume indicators: 24h/72h rolling averages, volume change | Adding volume often improves prediction in markets where volume correlates with price moves. Can be switched off if data is unreliable or noisy.                                           |

---

## 💡 Notes on Parameter Selection

1. **`interval` and `train_window`**: Higher-frequency data (like 15m) requires larger training windows to capture sufficient patterns. For BTC hourly, 2000–2500 bars (~3 months) is a reasonable starting point.
2. **`predict_window`**: Align it with your trading strategy horizon. Day traders may use 1–4h; swing traders may use 24–48h.
3. **`step`**: If computational cost is high, set `step = predict_window`. This ensures no overlapping predictions.
4. **`kernel` and `C`**: Test different SVM kernels and C values using cross-validation or walk-forward validation. RBF is usually best for crypto due to non-linearity.
5. **`high_vol_threshold`**: Analyze historical volatility to pick a threshold that separates calm vs. volatile periods. A too-low threshold may switch to classification too often.
6. **`use_volume_features`**: Enable if volume data is clean. Helps in assets with strong volume-price correlation. Disable if noisy or missing.

---

## ⚡ Backtesting Recommendations

* Always **compare strategy returns to buy & hold** to ensure the model adds value.
* Use **rolling evaluation** to monitor Sharpe ratio and drawdowns over time.
* Adjust **train_window, predict_window, and high_vol_threshold** iteratively to optimize performance.
* **Volume features** generally improve crypto predictions (BTC, ETH) more than stocks with low liquidity.

---

Do you want me to also **add a visual diagram** showing how the backtesting flow works with `train_window`, `predict_window`, and rolling updates? This often helps understand the timing of predictions vs. trades.

## Example results

adaptive_svm_btcusdt.py, when mode is dynamically chooses as "regression"

📊 Backtest Summary
asset: BTCUSDT
period: 2024-01-01 → 2025-01-31
use_volume_features: True
average_mae: 0.9996520700468582
final_cum_strategy_return: 1.746547817997818
final_cum_bh_return: 1.59690638219606
sharpe_ratio_hourly: 1.5188064326012396
max_drawdown: -23.95%
short_long_ratio: 0.8520710059171598
num_long_trades: 4056
num_short_trades: 3456
avg_long_return_per_trade: 0.00015901741581557402
avg_short_return_per_trade: 7.703535819657453e-06
percent_profitable_trades: 50.21%