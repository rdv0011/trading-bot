import ccxt
import pandas as pd
import numpy as np
import time, psutil, warnings
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm
from mlio import MODEL_DIR, save_model, save_feature_columns
from xgcatboostcore import make_features, make_labels, adaptive_thresholding

warnings.filterwarnings("ignore")

SYMBOL = 'BTC/USDT'
DAYS = 60
SAVE_FINAL_MODEL = True # Models are saved to models/ directory

# Metaparameters as constants
STAKE_PCT = 0.5
STOP_LOSS_PCT = 0.01
TAKE_PROFIT_PCT = 0.02
PREDICT_WITH_SIGNAL_NUM_CANDLES = 600
PREDICT_WITH_SIGNAL_LABEL_WINDOW = 200
MAX_HISTORY_SIZE = 600
HISTORICAL_PRICES_LENGTH = 500
HISTORICAL_PRICES_LIMIT = 1000
MAX_HOLD_HOURS = 24  # Hours
HISTORICAL_PRICES_TIMEFRAME = "5m"
# It represents the number of future periods (candles) ahead to predict returns.
# Given the 5-minute timeframe (HISTORICAL_PRICES_TIMEFRAME = "5m"), H=20 corresponds to 100 minutes (1.67 hours) into the future.
# This value is likely chosen empirically or based on backtesting to balance prediction accuracy with market dynamics 
# (e.g., capturing short-term trends without excessive noise). 
# It aligns with MAX_HOLD_HOURS = 24 (longer holding) but focuses on a specific forward-looking window for labeling.
# In the code, H=20 is hardcoded as a constant, suggesting it's a tuned hyperparameter. 
# Adjusting it could impact model performance (e.g., higher H for longer-term predictions, lower for more responsive signals). 
# If needed, it could be made configurable for experimentation.
NUMBER_OF_CANDLES_AHEAD = 20 
##

# =============================================
# Update rolling_train_predict to save models
# =============================================
def rolling_train_predict(df, model_type='xgb', retrain_every=12, window=4032, save_final_model=True):
    """
    Rolling adaptive model trainer & predictor for BTC/USDT price.
    
    Auto-adjusts the training window based on dataset size.
    Works with XGBoost or CatBoost regressors.
    
    Args:
        df: DataFrame with features and labels
        model_type: 'xgb' or 'cat'
        retrain_every: How often to retrain (in candles)
        window: Training window size
        save_final_model: Whether to save the final trained model
    
    Returns:
        tuple: (predictions_df, resource_log_df, final_model, features)
    """

    n = len(df)

    # ✅ Auto-adjust window if not provided or too large
    if window is None or window >= n:
        window = max(50, n // 3)
        print(f"[Auto-adjust] Using window={window} for dataset length={n}")
    elif window < 50:
        window = 50
        print(f"[Adjusted] Minimum window enforced (50).")

    features = [c for c in df.columns if c not in ['future_close','future_ret']]
    preds = []
    resource_log = []
    mdl = None
    
    for i in tqdm(range(window, len(df)), desc=f"Training {model_type.upper()}", unit="candle"):
        if i % retrain_every == 0 or mdl is None:
            train_df = df.iloc[i-window:i]
            X_train, y_train = train_df[features], train_df['future_ret']
            t0 = time.time()
            if model_type=='xgb':
                mdl = XGBRegressor(n_estimators=500, n_jobs=1)
                mdl.fit(X_train, y_train)
            else:
                mdl = CatBoostRegressor(iterations=500, verbose=False)
                mdl.fit(X_train, y_train)
            train_time = time.time() - t0
            mem = psutil.Process().memory_info().rss / (1024**2)
            resource_log.append({'i':i, 'train_time_s':train_time, 'mem_MB':mem})
        X_pred = df.iloc[[i]][features]
        t1 = time.time()
        p = mdl.predict(X_pred)[0]
        infer_time = time.time() - t1
        preds.append(p)
        resource_log[-1]['infer_time_s'] = infer_time
    
    df_res = df.iloc[window:].copy()
    df_res['pred'] = preds
    res_df = pd.DataFrame(resource_log)
    
    # Save final model and features
    if save_final_model and mdl is not None:
        save_model(mdl, model_type)
        save_feature_columns(features, model_type)
        print(f"💾 Saved final {model_type.upper()} model and features")
    
    return df_res, res_df, mdl, features

# =============================================
# Generate trading signals
# =============================================
def simulate_trades(df, pred_col='pred'):
    wallet = 1.0
    stake = STAKE_PCT
    stop_loss = -STOP_LOSS_PCT
    take_profit = TAKE_PROFIT_PCT
    max_hold = MAX_HOLD_HOURS * 60  # Convert hours to minutes
    position = 0
    entry_price = 0
    pnl = []
    pred_hist = []
    entry_index = 0
    num_candles = PREDICT_WITH_SIGNAL_NUM_CANDLES
    label_window = PREDICT_WITH_SIGNAL_LABEL_WINDOW

    
    # Track individual trades with timestamps
    trades = []
    trade_markers = []  # For plotting: (timestamp, price, type, position_type)

    for i in range(len(df)):
        pred_hist.append(df[pred_col].iloc[i])
        if len(pred_hist) < num_candles:
            pnl.append(wallet)
            continue
        max_th, min_th = adaptive_thresholding(pd.Series(pred_hist), num_candles=num_candles, label_window=label_window)
        price = df['close'].iloc[i]
        timestamp = df.index[i]

        # Entry
        if position == 0 and not np.isnan(max_th):
            if df[pred_col].iloc[i] > max_th:
                position = 1
                entry_price = price
                entry_index = i
                trade_markers.append({'timestamp': timestamp, 'price': price, 'type': 'entry', 'position': 'long'})
            elif df[pred_col].iloc[i] < min_th:
                position = -1
                entry_price = price
                entry_index = i
                trade_markers.append({'timestamp': timestamp, 'price': price, 'type': 'entry', 'position': 'short'})

        # Exit
        if position != 0:
            perf = (price / entry_price - 1) * position
            if perf <= stop_loss or perf >= take_profit or (i - entry_index)*5 >= max_hold:
                wallet *= (1 + stake * perf)
                trades.append({'position': position, 'return': perf})
                trade_markers.append({'timestamp': timestamp, 'price': price, 'type': 'exit', 'position': 'long' if position == 1 else 'short', 'profit': perf > 0})
                position = 0
                entry_price = 0
        pnl.append(wallet)
    
    df = df.copy()
    df['wallet'] = pnl
    # Store trades and markers as attributes
    df.attrs['trades'] = trades
    df.attrs['trade_markers'] = trade_markers
    return df

def calculate_metrics(df):
    """
    Calculate comprehensive trading metrics.
    """
    metrics = {}
    
    # Extract wallet series
    wallet = df['wallet']
    
    # 1. Sharpe Ratio (hourly) - assumes 5-min candles, so 12 candles = 1 hour
    returns = wallet.pct_change().dropna()
    if len(returns) > 0:
        hourly_returns = returns.rolling(12).sum().dropna()
        if len(hourly_returns) > 0 and hourly_returns.std() != 0:
            metrics['sharpe_ratio_hourly'] = (hourly_returns.mean() / hourly_returns.std()) * np.sqrt(24)
        else:
            metrics['sharpe_ratio_hourly'] = 0.0
    else:
        metrics['sharpe_ratio_hourly'] = 0.0
    
    # 2. Max Drawdown
    cummax = wallet.cummax()
    drawdown = (wallet - cummax) / cummax
    metrics['max_drawdown'] = drawdown.min()
    
    # 3. Extract trades from dataframe attributes
    trades = df.attrs.get('trades', [])
    
    long_trades = [t for t in trades if t['position'] == 1]
    short_trades = [t for t in trades if t['position'] == -1]
    
    metrics['num_long_trades'] = len(long_trades)
    metrics['num_short_trades'] = len(short_trades)
    
    # 4. Short/Long Ratio
    if metrics['num_long_trades'] > 0:
        metrics['short_long_ratio'] = metrics['num_short_trades'] / metrics['num_long_trades']
    else:
        metrics['short_long_ratio'] = float('inf') if metrics['num_short_trades'] > 0 else 0.0
    
    # 5. Average returns per trade type
    if len(long_trades) > 0:
        metrics['avg_long_return_per_trade'] = np.mean([t['return'] for t in long_trades])
    else:
        metrics['avg_long_return_per_trade'] = 0.0
        
    if len(short_trades) > 0:
        metrics['avg_short_return_per_trade'] = np.mean([t['return'] for t in short_trades])
    else:
        metrics['avg_short_return_per_trade'] = 0.0
    
    # 6. Percent profitable trades
    if len(trades) > 0:
        profitable = sum(1 for t in trades if t['return'] > 0)
        metrics['percent_profitable_trades'] = (profitable / len(trades)) * 100
    else:
        metrics['percent_profitable_trades'] = 0.0
    
    # Final wallet value
    metrics['final_wallet'] = wallet.iloc[-1]
    
    return metrics

def _print_single_model_metrics(model_name, metrics, avg_train_time):
    """
    Print metrics for a single model with explanations and improvement suggestions.
    """
    print("\n" + "="*70)
    print(f"📊 {model_name} METRICS")
    print("="*70)
    
    # Final Wallet
    final_wallet = metrics['final_wallet']
    print(f"\n💰 Final Wallet: {final_wallet:.4f}")
    print(f"   └─ Explanation: Started with 1.0, ended with {final_wallet:.4f}")
    if final_wallet > 1.0:
        roi = (final_wallet - 1.0) * 100
        print(f"   └─ ✅ Profit: +{roi:.2f}% ROI")
        print(f"   └─ Suggestion: Consider increasing stake size or holding duration")
    elif final_wallet < 1.0:
        loss = (1.0 - final_wallet) * 100
        print(f"   └─ ❌ Loss: -{loss:.2f}%")
        print(f"   └─ Suggestion: Tighten stop-loss, adjust entry thresholds, or retrain more frequently")
    else:
        print(f"   └─ ⚠️  Break-even")
        print(f"   └─ Suggestion: Strategy needs significant adjustments")
    
    # Sharpe Ratio
    sharpe = metrics['sharpe_ratio_hourly']
    print(f"\n📈 Sharpe Ratio (Hourly): {sharpe:.4f}")
    print(f"   └─ Explanation: Risk-adjusted returns (higher is better)")
    if sharpe > 2.0:
        print(f"   └─ ✅ Excellent: Very strong risk-adjusted performance")
        print(f"   └─ Suggestion: Consider scaling up capital allocation")
    elif sharpe > 1.0:
        print(f"   └─ ✅ Good: Acceptable risk-adjusted returns")
        print(f"   └─ Suggestion: Optimize position sizing for better consistency")
    elif sharpe > 0:
        print(f"   └─ ⚠️  Fair: Barely beating risk-free rate")
        print(f"   └─ Suggestion: Improve entry/exit signals, reduce losing trades")
    else:
        print(f"   └─ ❌ Poor: Negative risk-adjusted returns")
        print(f"   └─ Suggestion: Strategy is taking too much risk for returns - major revision needed")
    
    # Max Drawdown
    max_dd = metrics['max_drawdown']
    print(f"\n📉 Max Drawdown: {max_dd:.2%}")
    print(f"   └─ Explanation: Largest peak-to-trough decline in wallet value")
    if max_dd > -0.05:
        print(f"   └─ ✅ Excellent: Very controlled risk (< 5%)")
        print(f"   └─ Suggestion: Strategy is conservative - could increase position size")
    elif max_dd > -0.15:
        print(f"   └─ ✅ Good: Acceptable drawdown (5-15%)")
        print(f"   └─ Suggestion: Monitor closely during volatile periods")
    elif max_dd > -0.30:
        print(f"   └─ ⚠️  Warning: High drawdown (15-30%)")
        print(f"   └─ Suggestion: Implement stricter stop-losses or reduce leverage")
    else:
        print(f"   └─ ❌ Critical: Severe drawdown (>30%)")
        print(f"   └─ Suggestion: Emergency - reduce position size immediately, review risk management")
    
    # Trade Counts
    num_long = metrics['num_long_trades']
    num_short = metrics['num_short_trades']
    total_trades = num_long + num_short
    print(f"\n🔢 Trade Statistics:")
    print(f"   ├─ Long Trades: {num_long}")
    print(f"   ├─ Short Trades: {num_short}")
    print(f"   └─ Total Trades: {total_trades}")
    if total_trades < 10:
        print(f"   └─ ⚠️  Very few trades - not enough data for statistical significance")
        print(f"   └─ Suggestion: Lower entry thresholds or increase data period")
    elif total_trades < 30:
        print(f"   └─ ⚠️  Low sample size - results may not be reliable")
        print(f"   └─ Suggestion: Collect more trades before drawing conclusions")
    else:
        print(f"   └─ ✅ Good sample size for analysis")
    
    # Short/Long Ratio
    sl_ratio = metrics['short_long_ratio']
    print(f"\n⚖️  Short/Long Ratio: {sl_ratio:.2f}")
    print(f"   └─ Explanation: Balance between short and long positions")
    if sl_ratio > 2.0:
        print(f"   └─ ⚠️  Heavily biased towards shorts")
        print(f"   └─ Suggestion: Check if model is overfitting to bearish trends")
    elif sl_ratio < 0.5:
        print(f"   └─ ⚠️  Heavily biased towards longs")
        print(f"   └─ Suggestion: Ensure model captures both market directions")
    else:
        print(f"   └─ ✅ Well-balanced strategy")
        print(f"   └─ Suggestion: Strategy adapts to market conditions effectively")
    
    # Average Returns per Trade Type
    avg_long_ret = metrics['avg_long_return_per_trade']
    avg_short_ret = metrics['avg_short_return_per_trade']
    print(f"\n💹 Average Returns per Trade:")
    print(f"   ├─ Long Trades: {avg_long_ret:.2%}")
    if avg_long_ret > 0.02:
        print(f"   │  └─ ✅ Strong long performance")
    elif avg_long_ret > 0:
        print(f"   │  └─ ⚠️  Marginal long performance - consider tighter entry criteria")
    else:
        print(f"   │  └─ ❌ Losing on longs - avoid long positions or fix entry logic")
    
    print(f"   └─ Short Trades: {avg_short_ret:.2%}")
    if avg_short_ret > 0.02:
        print(f"      └─ ✅ Strong short performance")
    elif avg_short_ret > 0:
        print(f"      └─ ⚠️  Marginal short performance - consider tighter entry criteria")
    else:
        print(f"      └─ ❌ Losing on shorts - avoid short positions or fix entry logic")
    
    # Percent Profitable Trades
    pct_profitable = metrics['percent_profitable_trades']
    print(f"\n🎯 Percent Profitable Trades: {pct_profitable:.2f}%")
    print(f"   └─ Explanation: Win rate of closed trades")
    if pct_profitable > 60:
        print(f"   └─ ✅ Excellent: High win rate")
        print(f"   └─ Suggestion: Consider increasing position size per trade")
    elif pct_profitable > 50:
        print(f"   └─ ✅ Good: Above 50% win rate")
        print(f"   └─ Suggestion: Focus on letting winners run longer")
    elif pct_profitable > 40:
        print(f"   └─ ⚠️  Below average: Strategy needs improvement")
        print(f"   └─ Suggestion: Improve entry signals or adjust stop-loss levels")
    else:
        print(f"   └─ ❌ Poor: Most trades are losing")
        print(f"   └─ Suggestion: Complete strategy overhaul needed - check feature engineering")
    
    # Training Time
    print(f"\n⏱️  Avg Train Time: {avg_train_time:.3f}s")
    print(f"   └─ Explanation: Average time to retrain model")
    if avg_train_time < 1.0:
        print(f"   └─ ✅ Fast: Good for high-frequency retraining")
    elif avg_train_time < 5.0:
        print(f"   └─ ✅ Acceptable: Reasonable for adaptive strategy")
    else:
        print(f"   └─ ⚠️  Slow: May impact real-time trading")
        print(f"   └─ Suggestion: Reduce n_estimators or training window size")

def print_metrics(metrics_xgb, metrics_cat, res_xgb, res_cat):
    """
    Display comprehensive metrics comparison for both models with insights.
    """
    _print_single_model_metrics("XGBOOST", metrics_xgb, res_xgb['train_time_s'].mean())
    _print_single_model_metrics("CATBOOST", metrics_cat, res_cat['train_time_s'].mean())
    
    # Comparative Summary
    print("\n" + "="*70)
    print("🏆 COMPARATIVE SUMMARY")
    print("="*70)
    
    # Determine winner
    xgb_wallet = metrics_xgb['final_wallet']
    cat_wallet = metrics_cat['final_wallet']
    
    if xgb_wallet > cat_wallet:
        diff = ((xgb_wallet - cat_wallet) / cat_wallet) * 100
        print(f"✅ Winner: XGBoost (outperformed by {diff:.2f}%)")
    elif cat_wallet > xgb_wallet:
        diff = ((cat_wallet - xgb_wallet) / xgb_wallet) * 100
        print(f"✅ Winner: CatBoost (outperformed by {diff:.2f}%)")
    else:
        print(f"⚖️  Tie: Both models performed equally")
    
    # Best metrics comparison
    print(f"\n📊 Best Metrics:")
    print(f"   ├─ Higher Sharpe: {'XGBoost' if metrics_xgb['sharpe_ratio_hourly'] > metrics_cat['sharpe_ratio_hourly'] else 'CatBoost'}")
    print(f"   ├─ Lower Drawdown: {'XGBoost' if metrics_xgb['max_drawdown'] > metrics_cat['max_drawdown'] else 'CatBoost'}")
    print(f"   ├─ Higher Win Rate: {'XGBoost' if metrics_xgb['percent_profitable_trades'] > metrics_cat['percent_profitable_trades'] else 'CatBoost'}")
    print(f"   └─ Faster Training: {'XGBoost' if res_xgb['train_time_s'].mean() < res_cat['train_time_s'].mean() else 'CatBoost'}")
    
    print("\n💡 General Recommendations:")
    if xgb_wallet > 1.0 or cat_wallet > 1.0:
        print("   ✅ Strategy shows promise - consider paper trading")
        print("   ✅ Optimize hyperparameters (stake size, stop-loss, max_hold)")
        print("   ✅ Test on different market conditions (bull/bear/sideways)")
    else:
        print("   ⚠️  Strategy needs major improvements before live trading")
        print("   ⚠️  Review feature engineering and label generation")
        print("   ⚠️  Consider different prediction horizons (H parameter)")
    
    print("="*70 + "\n")

def plot_results(df_xgb, df_cat):
    """
    Plot 2-panel comparison between XGBoost and CatBoost models.
    Top panel: Candlestick chart with trade entry/exit markers
    Bottom panel: Wallet performance comparison
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), sharex=True)
    
    # ========================================
    # TOP PANEL: Candlestick chart with trade markers
    # ========================================
    df_plot = df_xgb[['open', 'high', 'low', 'close']].copy()
    
    # Draw candlesticks (sampled for performance)
    sample_rate = max(1, len(df_plot)//500)
    for idx in range(0, len(df_plot), sample_rate):
        date = df_plot.index[idx]
        open_price = df_plot['open'].iloc[idx]
        high_price = df_plot['high'].iloc[idx]
        low_price = df_plot['low'].iloc[idx]
        close_price = df_plot['close'].iloc[idx]
        
        color = 'green' if close_price >= open_price else 'red'
        
        # High-Low line (wick)
        ax1.plot([date, date], [low_price, high_price], color='black', linewidth=0.5, alpha=0.5)
        # Open-Close body
        ax1.plot([date, date], [open_price, close_price], color=color, linewidth=2, alpha=0.8)
    
    # Add trade markers for XGBoost
    markers_xgb = df_xgb.attrs.get('trade_markers', [])
    for marker in markers_xgb:
        if marker['type'] == 'entry':
            if marker['position'] == 'long':
                ax1.scatter(marker['timestamp'], marker['price'], marker='^', s=100, 
                           color='lime', edgecolors='darkgreen', linewidth=1.5, 
                           zorder=5, label='Long Entry' if marker == markers_xgb[0] else '')
            else:
                ax1.scatter(marker['timestamp'], marker['price'], marker='v', s=100, 
                           color='red', edgecolors='darkred', linewidth=1.5, 
                           zorder=5, label='Short Entry' if marker == markers_xgb[0] else '')
        elif marker['type'] == 'exit':
            color = 'cyan' if marker['profit'] else 'orange'
            ax1.scatter(marker['timestamp'], marker['price'], marker='x', s=100, 
                       color=color, linewidth=2, zorder=5, 
                       label='Exit (Profit)' if marker['profit'] and marker == markers_xgb[0] else 
                             ('Exit (Loss)' if not marker['profit'] and marker == markers_xgb[0] else ''))
    
    ax1.set_title('BTC/USDT Price with XGBoost Trade Signals', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    
    # ========================================
    # BOTTOM PANEL: Wallet performance
    # ========================================
    ax2.plot(df_xgb.index, df_xgb['wallet'], label='XGBoost Wallet', 
             linewidth=2, color='blue', alpha=0.8)
    ax2.plot(df_cat.index, df_cat['wallet'], label='CatBoost Wallet', 
             linewidth=2, color='orange', alpha=0.8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial Wallet')
    
    # Fill area between wallet and initial value to show profit/loss
    ax2.fill_between(df_xgb.index, df_xgb['wallet'], 1.0, 
                      where=(df_xgb['wallet'] >= 1.0), alpha=0.2, color='green', 
                      interpolate=True)
    ax2.fill_between(df_xgb.index, df_xgb['wallet'], 1.0, 
                      where=(df_xgb['wallet'] < 1.0), alpha=0.2, color='red', 
                      interpolate=True)

    ax2.set_title('Wallet Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Wallet Value', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # =============================================
    # Download BTC/USDT data from Binance (5m)
    # =============================================

    exchange = ccxt.binance()
    timeframe = HISTORICAL_PRICES_TIMEFRAME
    limit = HISTORICAL_PRICES_LIMIT # adjust as needed
    since = exchange.milliseconds() - DAYS * 24 * 60 * 60 * 1000

    print("Fetching data from Binance...")

    all_ohlcv = []

    while True:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        since = ohlcv[-1][0] + (5 * 60 * 1000)  # move forward 5 minutes
        print(f"Fetched {len(all_ohlcv)} candles so far...")
        time.sleep(exchange.rateLimit / 1000)  # avoid rate limit

    df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('date').set_index('date').sort_index()
    print(f"✅ Loaded {len(df)} candles from Binance")

    df = make_features(df)

    df = make_labels(df, H=NUMBER_OF_CANDLES_AHEAD)
    print("Feature matrix shape:", df.shape)

    print("Running XGBoost adaptive model...")
    df_xgb, res_xgb, model_xgb, features_xgb = rolling_train_predict(
        df,
        'xgb',
        save_final_model=SAVE_FINAL_MODEL
    )

    print("Running CatBoost adaptive model...")
    df_cat, res_cat, model_cat, features_cat = rolling_train_predict(
        df,
        'cat',
        save_final_model=SAVE_FINAL_MODEL
    )

    df_xgb = simulate_trades(df_xgb)
    df_cat = simulate_trades(df_cat)

    # Calculate metrics
    metrics_xgb = calculate_metrics(df_xgb)
    metrics_cat = calculate_metrics(df_cat)

    # =============================================
    # 7. Plot & Display Results
    # =============================================
    plot_results(df_xgb, df_cat)
    print_metrics(metrics_xgb, metrics_cat, res_xgb, res_cat)