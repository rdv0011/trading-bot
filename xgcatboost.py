import ccxt
import pandas as pd
import numpy as np
import time, psutil, warnings
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm
from mlio import save_model, save_feature_columns
from xgcatboostcore import make_features, make_labels, adaptive_thresholding
from displayresults import plot_results, print_metrics

warnings.filterwarnings("ignore")

SYMBOL = 'BTC/USDT'
TRAINING_WINDOW_DAYS = 60 # How far to the past to fetch data and train ML model on
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

if __name__ == "__main__":
    # =============================================
    # Download BTC/USDT data from Binance (5m)
    # =============================================

    exchange = ccxt.binance()
    timeframe = HISTORICAL_PRICES_TIMEFRAME
    limit = HISTORICAL_PRICES_LIMIT # adjust as needed
    since = exchange.milliseconds() - TRAINING_WINDOW_DAYS * 24 * 60 * 60 * 1000

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