from typing import Optional
import numpy as np
import pandas as pd
import inspect

TARGET_COLUMN = 'future_ret'
SIGNAL_COLUMN = 'pred'
SEED_BASE = 42
NUMBER_OF_CANDLES_AHEAD = 20
PREDICT_WITH_SIGNAL_NUM_CANDLES = 600
MAX_HISTORY_SIZE = 600
PREDICT_WITH_SIGNAL_LABEL_WINDOW = 200
MIN_CANDLES_FOR_FEATURES = 250
OBJECTIVE_METRIC = "objective_score" # Composite metric better for ML training

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
    
    # --- Cyclical time encoding ---
    hours = df.index.hour
    dows = df.index.dayofweek

    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dows / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dows / 7)
    
    df = df.dropna().round(5)
    return df

# =============================================
# Label generation (future return)
# =============================================
def make_labels(df, H=20):
    """
    Adds future returns to dataframe for supervised learning.

    Args:
        df: pd.DataFrame with at least 'close' column
        H: prediction horizon (number of candles ahead)

    Returns:
        df with columns 'future_close' and 'future_ret'
    """
    df = df.copy()
    df['future_close'] = df['close'].shift(-H)
    df['future_ret'] = (df['future_close'] / df['close']) - 1.0
    
    df = df.dropna().round(5)

    return df

def get_features(df):
    """
    Returns list of columns to use as features for ML models.
    Excludes target columns 'future_close' and 'future_ret'.

    Args:
        df: pd.DataFrame

    Returns:
        list of feature column names
    """
    return [c for c in df.columns if c not in ['future_close','future_ret']]

# =============================================
# Adaptive training loop
# =============================================
def adaptive_thresholding(df, num_candles=PREDICT_WITH_SIGNAL_NUM_CANDLES, label_window=PREDICT_WITH_SIGNAL_LABEL_WINDOW):
    if len(df) < num_candles:
        print(f"Insufficient data for adaptive thresholding: {len(df)} < {num_candles} candles")
        return np.nan, np.nan
    sorted_vals = df.tail(num_candles).sort_values(ascending=False)
    frequency = max(1, int(num_candles / label_window))
    maxima_sort_threshold = sorted_vals.iloc[:frequency].mean()
    minima_sort_threshold = sorted_vals.iloc[-frequency:].mean()
    return maxima_sort_threshold, minima_sort_threshold

def _get_params_from_set(param_set, idx=None):
    """
    Get parameters from param_set (dict or list/tuple of dicts) by index.

    If param_set is a list/tuple, idx selects the element. If idx is out of bounds or None, 
    the first element is used. If param_set is a single dict, it is returned directly.

    Returns a dict with all expected keys filled with defaults if missing.
    """
    # Default values
    default_values = {
        "stake_pct": 0.01,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "max_hold_hours": 4,
    }

    # Select the dict
    if isinstance(param_set, (list, tuple)):
        if idx is None or idx < 0 or idx >= len(param_set):
            selected = param_set[0]
        else:
            selected = param_set[idx]
    elif isinstance(param_set, dict):
        selected = param_set
    else:
        raise ValueError("param_set must be a dict or a list/tuple of dicts")

    # Apply defaults for missing keys
    return {k: selected.get(k, v) for k, v in default_values.items()}


def simulate_trades_core(
    df,
    df_hist,
    signal_col,
    close_col='close',
    timeframe_minutes=1,
    params=None,
):
    """
    Simulate trades on the DataFrame with adaptive thresholds.

    If no trades occur, returns default metrics to avoid NaNs for ML pipelines.

    params can be a dict, list of dicts, or pandas Series of dicts.
    """
    df_iter = df.copy()
    wallet = 1.0
    wallet_history = []
    trades = []
    trade_markers = []

    pred_hist = []
    position = 0
    entry_price = 0.0
    entry_index = None
    entry_time = None

    # Use historical data for adaptive thresholds if provided
    adaptive_source = df_hist[signal_col].tolist()

    for i, (timestamp, row) in enumerate(df_iter.iterrows()):
        price = row[close_col]
        signal_value = row[signal_col]

        # Adaptive thresholds
        hist_for_thresholds = adaptive_source + pred_hist[-PREDICT_WITH_SIGNAL_NUM_CANDLES:]
        hist_for_thresholds.append(signal_value)

        if len(hist_for_thresholds) < PREDICT_WITH_SIGNAL_NUM_CANDLES:
            wallet_history.append(wallet)
            pred_hist.append(signal_value)
            continue

        adaptive_max, adaptive_min = adaptive_thresholding(pd.Series(hist_for_thresholds))

        # Get per-row parameters safely
        param_row = _get_params_from_set(params, i)
        stake = param_row["stake_pct"]
        stop_loss = param_row["stop_loss_pct"]
        take_profit = param_row["take_profit_pct"]
        max_hold = param_row["max_hold_hours"]

        # Entry logic
        if position == 0:
            enter_long = signal_value > adaptive_max
            enter_short = signal_value < adaptive_min
            if enter_long or enter_short:
                position = 1 if enter_long else -1
                entry_price = price
                entry_index = i
                entry_time = timestamp
                trade_markers.append({
                    'timestamp': timestamp,
                    'price': price,
                    'type': 'entry',
                    'position': 'long' if position == 1 else 'short',
                })

        # Exit logic
        if position != 0 and entry_index is not None:
            perf = (price / entry_price - 1.0) * position
            elapsed_minutes = (i - entry_index) * max(float(timeframe_minutes), 1.0)
            exit_on_stop = perf <= -stop_loss
            exit_on_take = perf >= take_profit
            exit_on_time = elapsed_minutes >= max_hold * 60

            if exit_on_stop or exit_on_take or exit_on_time:
                wallet *= (1.0 + stake * perf)
                exit_reason = 'stop_loss' if exit_on_stop else ('take_profit' if exit_on_take else 'max_hold')
                trades.append({
                    'position': position,
                    'return': perf,
                    'entry_timestamp': entry_time,
                    'exit_timestamp': timestamp,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'stake_pct': stake,
                    'exit_reason': exit_reason,
                })
                trade_markers.append({
                    'timestamp': timestamp,
                    'price': price,
                    'type': 'exit',
                    'position': 'long' if position == 1 else 'short',
                    'profit': perf > 0,
                    'reason': exit_reason,
                })

                position = 0
                entry_price = 0.0
                entry_index = None
                entry_time = None

        wallet_history.append(wallet)
        pred_hist.append(signal_value)

    # Close final open position
    if position != 0 and entry_index is not None:
        final_price = df_iter[close_col].iloc[-1]
        final_timestamp = df_iter.index[-1]
        perf = (final_price / entry_price - 1.0) * position
        wallet *= (1.0 + stake * perf)
        trades.append({
            'position': position,
            'return': perf,
            'entry_timestamp': entry_time,
            'exit_timestamp': final_timestamp,
            'entry_price': entry_price,
            'exit_price': final_price,
            'stake_pct': stake,
            'exit_reason': 'final_close',
        })
        trade_markers.append({
            'timestamp': final_timestamp,
            'price': final_price,
            'type': 'exit',
            'position': 'long' if position == 1 else 'short',
            'profit': perf > 0,
            'reason': 'final_close',
        })
        wallet_history[-1] = wallet

    # Build result DataFrame
    df_result = df_iter.copy()
    df_result['wallet'] = np.round(wallet_history, 5)
    df_result.attrs['trades'] = trades
    df_result.attrs['trade_markers'] = trade_markers

    _, metrics = caculate_metrics(trades, wallet)

    return df_result, metrics

def caculate_metrics(trades, wallet, expected_trades=10):
    """
    Compute a stable, always-defined objective metric for short walk-forward windows.

    Parameters
    ----------
    trades : list of dict
        Each dict must contain a 'return' field.
    wallet : float
        Final wallet value from simulation.
    expected_trades : int
        Normalization factor for the activity term.

    Returns
    -------
    objective_metric : float
        Composite score for walk-forward optimization.
    metrics : dict
        Component metrics for diagnostics.
    """
    if not trades:
        # No trades → very small score
        metrics = {
            'mean_return': 0.0,
            'win_rate': 0.0,
            'downside': 0.0,
            'activity': 0.0,
            'final_wallet': round(wallet, 5),
            'trades_count': 0,
            OBJECTIVE_METRIC: -0.1,
        }
        return -0.1, metrics

    returns = [t['return'] for t in trades]

    # --- Basic components ---
    mean_ret = float(np.mean(returns))
    win_rate = float(sum(r > 0 for r in returns) / len(returns))

    downside = (
        float(np.mean([-r for r in returns if r < 0]))
        if any(r < 0 for r in returns)
        else 0.0
    )

    # --- Activity factor (caps at 1.0) ---
    activity = min(len(trades) / float(expected_trades), 1.0)

    # --- Composite scoring function ---
    objective_metric = (
        3.0 * mean_ret +
        2.0 * win_rate -
        1.0 * downside +
        0.5 * activity
    )

    metrics = {
        'mean_return': mean_ret,
        'win_rate': win_rate,
        'downside': downside,
        'activity': activity,
        'final_wallet': wallet,
        'trades_count': len(trades),
        OBJECTIVE_METRIC: objective_metric,
    }

    return objective_metric, metrics

def create_model(model_cls, random_seed, model_params):
    """Safely instantiate model, passing the appropriate seed parameter if supported."""
    # Get constructor signature
    try:
        sig = inspect.signature(model_cls)
        params = sig.parameters.keys()
    except (TypeError, ValueError):
        params = []

    # Choose appropriate seed parameter name
    seed_args = {}
    if 'random_state' in params:
        seed_args['random_state'] = random_seed
    elif 'random_seed' in params:
        seed_args['random_seed'] = random_seed
    elif 'seed' in params:
        seed_args['seed'] = random_seed

    # Merge parameters safely
    return model_cls(**seed_args, **model_params)

def resolve_model_class(model_type):
    """
    Resolve the appropriate model class based on self.model_type.
    Supports: 'cat', 'xgb'
    """

    model_map = {
        "cat": ("catboost", "CatBoostRegressor"),
        "xgb": ("xgboost", "XGBRegressor"),
    }

    key = model_type.lower()

    if key not in model_map:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Supported values: {list(model_map.keys())}"
        )

    module_name, class_name = model_map[key]

    # Dynamically import module & class
    module = __import__(module_name, fromlist=[class_name])
    model_class = getattr(module, class_name)

    return model_class

def predict_param_dicts_from_model(model, metadata: Optional[dict], X: pd.DataFrame):
    """
    Returns a SINGLE dict mapping target_keys to predicted values.
    Handles callable or sklearn-like model.predict.
    Safely extracts a single prediction even if the model returns multiple.
    """

    # -------------------------
    # 1. Run model prediction
    # -------------------------
    try:
        raw_preds = model(X) if callable(model) else model.predict(X)
    except Exception:
        raw_preds = model.predict(X.values)

    # -------------------------
    # 2. Normalize to one prediction
    # -------------------------

    # Case A: model returned list of dicts → take first dict
    if isinstance(raw_preds, (list, np.ndarray)) and len(raw_preds) > 0:
        if isinstance(raw_preds[0], dict):
            return raw_preds[0]

    # Case B: model returned list of arrays → take first prediction
    # e.g., [array([0.1, 0.3, 0.8])]
    if isinstance(raw_preds, (list, np.ndarray)) and len(raw_preds) > 0:
        try:
            single_pred = np.asarray(raw_preds[0])
        except Exception:
            single_pred = np.asarray(raw_preds)
    else:
        # Fallback: assume the output itself is an array
        single_pred = np.asarray(raw_preds)

    # Ensure the result is 1D
    single_pred = np.atleast_1d(single_pred)

    # -------------------------
    # 3. Determine correct keys
    # -------------------------
    meta_tkeys   = metadata.get("target_keys") if metadata else None
    meta_valid   = metadata.get("valid_targets") if metadata else None
    meta_removed = metadata.get("removed_targets") if metadata else {}

    n_cols = single_pred.shape[0]

    target_keys = meta_tkeys or meta_valid or [f"param_{i}" for i in range(n_cols)]
    valid_keys  = meta_valid or target_keys

    # Only assign keys for which predictions exist
    col_to_key = valid_keys[:n_cols]

    # -------------------------
    # 4. Build single result dict
    # -------------------------
    result = {k: None for k in target_keys}

    # Fill predictions
    for j, key in enumerate(col_to_key):
        try:
            result[key] = float(single_pred[j])
        except Exception:
            result[key] = None

    # Add any constant removed targets
    for removed_key, constant_value in meta_removed.items():
        result[removed_key] = constant_value

    return result
