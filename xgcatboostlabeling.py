# xgcatboostlabeling.py
# Produces trained XGBoost/CatBoost models and a labeled historical price dataset

import ccxt
from flask import json
import pandas as pd
import numpy as np
import time, warnings
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
from xgcatboost import TRAINING_WINDOW_DAYS
from xgcatboostcore import make_features, make_labels
from xgcatboostcore import get_features, simulate_trades_core
from xgcatboostcore import TARGET_COLUMN, SEED_BASE, SIGNAL_COLUMN
from xgcatboostcore import PREDICT_WITH_SIGNAL_NUM_CANDLES, OBJECTIVE_METRIC
from mlio import LABEL_DIR, load_model, save_model
from itertools import product
import inspect
import ast
import os
from typing import Tuple, Optional

warnings.filterwarnings("ignore")

SYMBOL = 'BTC/USDT'
TRAINING_WINDOW_DAYS = 20 # How far to the past to fetch data and train ML model on
SAVE_FINAL_MODEL = True # Models are saved to models/ directory

# Metaparameters as constants
MAX_HISTORY_SIZE = 600
TRAINING_WINDOW_MILLISECONDS = TRAINING_WINDOW_DAYS * 24 * 60 * 60 * 1000
HISTORICAL_PRICES_LENGTH = 500
HISTORICAL_PRICES_LIMIT = 1000
HISTORICAL_PRICES_TIMEFRAME = "5m"
NUMBER_OF_CANDLES_AHEAD = 20
WALKFORWARD_EVAL_HORIZON = 288  # how many candles ahead to simulate strategy performance
#OBJECTIVE_METRIC = 'sharpe_ratio_hourly'  # Smooth risk-adjusted equity curve
#OBJECTIVE_METRIC = 'profit_factor'  # Balance of risk and reward
#OBJECTIVE_METRIC = 'sortino_ratio_hourly'  # Focus on avoiding downside volatility

DEFAULT_LABEL_PARAMS = {
    'stake_pct': 0.5,
    'stop_loss_pct': 0.01,
    'take_profit_pct': 0.02,
    'max_hold_hours': 24,
}

def walkforward_label_forward_windows(
    df,
    param_grid,
    signal_col,
    window_size,
    timeframe_minutes,
    step=120,
    default_param_set=None
):
    """
    Walk-forward parameter selection using trading simulation.

    If no trades occur (metric cannot be computed), the function
    uses best parameters from previous iteration or `default_param_set`.
    """
    labels = []

    # If no default provided, use first param-grid entry
    if default_param_set is None:
        default_param_set = param_grid[0]

    prev_best_param = default_param_set
    prev_best_idx = 0
    prev_best_metric = -np.inf

    df_hist = df[:PREDICT_WITH_SIGNAL_NUM_CANDLES].copy()
    df_live = df[PREDICT_WITH_SIGNAL_NUM_CANDLES:].copy()

    for start_idx in tqdm(
        range(0, len(df_live) - window_size, step),
        desc="Walkforward optimization",
        total=((len(df_live) - window_size) // step) + 1,
    ):
        end_idx = start_idx + window_size
        df_test = df_live.iloc[start_idx:end_idx]

        best_metric = -np.inf
        best_param = None
        best_param_idx = None

        # Evaluate parameters
        for idx, param_set in enumerate(param_grid):

            _, metrics = simulate_trades_core(
                df=df_test,
                df_hist=df_hist,
                signal_col=signal_col,
                close_col='close',
                timeframe_minutes=timeframe_minutes,
                params=param_set,
            )

            metric_value = metrics.get(OBJECTIVE_METRIC, np.nan)

            # Accept as best if higher
            if metric_value > best_metric:
                best_metric = metric_value
                best_param = param_set
                best_param_idx = idx

            # Basic early-stopping logic
            if metric_value < best_metric:
                continue

        # If no valid best_param found → fallback to previous or default
        if best_param is None:
            best_param = prev_best_param
            best_param_idx = prev_best_idx
            best_metric = prev_best_metric

        # Store current best as "previous best" for next iteration
        prev_best_param = best_param
        prev_best_idx = best_param_idx
        prev_best_metric = best_metric

        labels.append({
            "date": df.index[end_idx - 1],
            "best_param_idx": best_param_idx,
            "best_param": best_param,
            "best_metric": best_metric,
        })

        df_hist = (
            pd.concat([df_hist, df_test], ignore_index=False)
              .iloc[-PREDICT_WITH_SIGNAL_NUM_CANDLES:]
        )

        # Convert to DataFrame that is already date-indexed
    labels_df = pd.DataFrame(labels)

    # Ensure date is datetime
    labels_df["date"] = pd.to_datetime(labels_df["date"])

    # Use date as index from now on
    labels_df = labels_df.set_index("date").sort_index()

    return labels_df

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

def rolling_train_predict_multi(
        df,
        model_cls,
        model_type,
        model_params,
        features,
        target_col,
        signal_col,
        window=None
        ):
    """
    Rolling per-candle prediction for multiple models
    with automatic caching for repeated runs.

    Args:
        df (pd.DataFrame): Input dataframe with features and target.
        model_cls (type): Model class (e.g., XGBRegressor, CatBoostRegressor, etc.).
        model_type (str): String identifier for the model type.
        model_params (dict): Parameters for model initialization.
        features (callable): features columns from df.
        target_col (str): Name of target column.
        window (int): Lookback window size (default: max(50, n//3)).

    Returns:
       df_res with predictions in 'pred' column
    """

    n = len(df)

    if window is None or window >= n:
        w = max(50, n // 3)
        print(f"[{model_type}] Auto-adjusted window={w} for dataset length={n}")
    elif window < 50:
        w = 50
        print(f"[{model_type}] Minimum window enforced (50)")
    else:
        w = window

    preds = []
    mdl = None
    retrain_every = 1  # Retrain every N candles for speed

    # Train model on rolling window
    for i in tqdm(range(w, n), desc=f"Training {model_type.upper()}", unit="candle"):
        if i % retrain_every == 0 or mdl is None:
            train_df = df.iloc[i-w:i]
            X_train, y_train = train_df[features], train_df[target_col]

            anchor_idx = i - w
            random_seed = SEED_BASE + anchor_idx

            mdl = create_model(model_cls, random_seed, model_params)

            mdl.fit(X_train, y_train)

        # Predict current candle
        X_pred = df.iloc[[i]][features]
        p = mdl.predict(X_pred)[0]
        preds.append(p)

    # Attach prediction column (NaNs left as is)
    # The result size will be n - window
    df_out = df.iloc[window:].copy()
    df_out[signal_col] = preds

    return df_out

def label_and_evaluate_intervals(
    df,
    model_type,
    param_grid,
    intervals_hours=(1, 2, 4),
    timeframe_minutes=5,
    signal_col=SIGNAL_COLUMN,
):
    """
    Label the dataset for a single model and evaluate trading performance
    for multiple forecast intervals using walk-forward per-candle parameters.

    Returns:
        summary_df: pd.DataFrame with one row per interval
        all_labels: dict of labels DataFrames per interval (date-indexed)
        all_results: dict of tuples (df_labeled, df_sim, metrics) per interval
        best_hours: interval with best metric
        best_labels: labels df for best interval (date-indexed)
    """

    candles_per_hour = int(round(60.0 / float(timeframe_minutes)))
    summary_rows = []
    all_labels = {}
    all_results = {}

    for hours in sorted(set(intervals_hours)):

        window_size = int(hours * candles_per_hour)
        step = max(1, window_size)
        
        print(f"\n--- Labeling for {hours}h horizon ({window_size} candles), step={step} ---")

        # --- Create labels df (date-index from the beginning) ---
        labels_df = walkforward_label_forward_windows(
            df=df,
            param_grid=param_grid,
            signal_col=signal_col,
            window_size=window_size,
            timeframe_minutes=timeframe_minutes,
        )

        all_labels[hours] = labels_df

        # --- Prepare df slice ---
        first_label_date = labels_df.index[0]
        df_slice = df.loc[df.index >= first_label_date].reset_index().sort_values('date')

        # Need date column again for merge_asof:
        labels_df_for_merge = labels_df.reset_index()

        # --- Merge parameters onto sliced df ---
        df_merged = pd.merge_asof(
            df_slice,
            labels_df_for_merge[['date', 'best_param']],
            on='date',
            direction='backward'
        ).sort_values('date')

        param_series = df_merged['best_param'].tolist()

        df_merged = df_merged.set_index("date").sort_index()

        df_hist = df_merged.iloc[:PREDICT_WITH_SIGNAL_NUM_CANDLES].copy()

        # --- Simulate ---
        df_sim, metrics_dyn = simulate_trades_core(
            df=df_merged,
            df_hist=df_hist,
            signal_col=signal_col,
            close_col='close',
            timeframe_minutes=timeframe_minutes,
            params=param_series,
        )

        # Store results using already-indexed labels_df
        all_results[hours] = (labels_df, df_sim, metrics_dyn)

        # Build summary row
        row = {
            'interval_hours': hours,
            'horizon_candles': window_size,
            f"{model_type}_objective": metrics_dyn.get(OBJECTIVE_METRIC, np.nan),
            f"{model_type}_final_wallet": metrics_dyn.get('final_wallet', np.nan),
            f"{model_type}_num_trades": metrics_dyn.get('trades_count', 0),
        }
        summary_rows.append(row)

    # --- Summary DataFrame ---
    summary_df = pd.DataFrame(summary_rows).set_index('interval_hours').sort_index()

    print("\nInterval comparison summary:")
    print(summary_df)

    # --- Best interval selection ---
    best_hours = None
    best_labels = None

    if not summary_df.empty:
        best_hours = summary_df[f"{model_type}_objective"].idxmax()
        best_labels = all_labels[best_hours]  # already indexed correctly

        best_row = summary_df.loc[best_hours]
        print(
            f"\n✅ Best interval: {best_hours}h "
            f"with objective={best_row[f'{model_type}_objective']:.4f} "
            f"and final_wallet={best_row[f'{model_type}_final_wallet']:.4f}"
        )

    return summary_df, all_labels, all_results, best_hours, best_labels

def train_best_param_multi_model(
    predicted_dfs,
    feature_cols=None,
    test_size=0.2,
    random_state=SEED_BASE,
):
    """
    Train a multi-output CatBoost model that predicts best_param.

    Constant-valued parameters are detected BEFORE creating target_ columns.
    This prevents CatBoost crashes and keeps metadata clean.
    """

    df = predicted_dfs.copy()

    # Convert dict-like strings into dicts
    df["best_param"] = df["best_param"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # -------- Extract raw target keys --------
    target_keys = set()
    df["best_param"].dropna().apply(
        lambda d: target_keys.update(d.keys()) if isinstance(d, dict) else None
    )
    target_keys = list(target_keys)

    # -------- Build raw value lists for each key BEFORE creating any columns --------
    values_per_key = {k: [] for k in target_keys}
    for d in df["best_param"].dropna():
        for k in target_keys:
            values_per_key[k].append(d.get(k))

    # -------- Detect constant vs variable (NO PREFIXES) --------
    valid_keys = []
    removed_keys = {}

    print("\n🔎 Checking target variability:")
    for key, vals in values_per_key.items():
        arr = pd.Series(vals).astype(float)
        unique_vals = arr.nunique()

        if unique_vals > 1:
            valid_keys.append(key)
            print(f"   {key}: {unique_vals} unique → included")
        else:
            constant_value = float(arr.iloc[0])
            removed_keys[key] = constant_value
            print(f"   {key}: constant {constant_value} → skipped")

    # Safety check
    if len(valid_keys) == 0:
        raise RuntimeError(
            "❌ All target parameters are constant — expand param search space."
        )

    # -------- Create target_ columns only for variable keys --------
    for key in valid_keys:
        df[f"target_{key}"] = df["best_param"].apply(
            lambda d: d.get(key) if isinstance(d, dict) else np.nan
        )

    # -------- Auto detect features --------
    if feature_cols is None:
        exclude = ["best_param"] + [f"target_{k}" for k in target_keys]
        feature_cols = [
            c for c in df.columns
            if c not in exclude and np.issubdtype(df[c].dtype, np.number)
        ]

    print(f"📊 Using {len(feature_cols)} input features")

    X = df[feature_cols].fillna(0)
    Y = df[[f"target_{k}" for k in valid_keys]].fillna(0)

    # -------- Train/test split --------
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # Base Cat model
    base = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
    )

    multi_model = MultiOutputRegressor(base)

    print("\n🚀 Training multi-output CatBoost model…")
    multi_model.fit(X_train, Y_train)

    # -------- Evaluate --------
    predictions = multi_model.predict(X_val)
    rmse_per_target = {}

    print("\n📈 Validation RMSE per target:")
    for i, key in enumerate(valid_keys):
        rmse = np.sqrt(np.mean((Y_val.iloc[:, i] - predictions[:, i]) ** 2))
        rmse_per_target[key] = rmse
        print(f"   {key}: {rmse:.6f}")

    # -------- Predictor wrapper reconstructs full best_param dict --------
    def predictor(X_input):
        preds = multi_model.predict(X_input)
        if preds.ndim == 1:
            preds = preds.reshape(1, -1)

        full_rows = []
        for r in preds:
            d = {}
            pi = 0

            for key in target_keys:
                if key in valid_keys:
                    d[key] = float(r[pi])
                    pi += 1
                else:
                    d[key] = removed_keys[key]  # constant parameter

            full_rows.append(d)

        return full_rows if len(full_rows) > 1 else full_rows[0]

    # -------- Metadata --------
    metadata_out = {
        "feature_cols": feature_cols,
        "target_keys": target_keys,   # raw keys
        "valid_targets": valid_keys,  # raw keys
        "removed_targets": removed_keys,  # raw keys and constant values
    }

    print("\n✅ Training complete.")
    return multi_model, predictor, rmse_per_target, metadata_out

def _predict_param_dicts_from_model(model, metadata: Optional[dict], X: pd.DataFrame, target_keys: Optional[list] = None):
    """
    Returns a list of dicts (one per X row) mapping target_keys to predicted values.
    Handles callable or sklearn-like model.predict. Missing keys are filled with None.
    """
    # Get raw predictions
    try:
        raw_preds = model(X) if callable(model) else model.predict(X)
    except Exception:
        raw_preds = model.predict(X.values)

    # If model already returned list of dicts
    if isinstance(raw_preds, (list, np.ndarray)) and len(raw_preds) > 0 and isinstance(raw_preds[0], dict):
        return list(raw_preds)

    arr = np.atleast_2d(np.asarray(raw_preds))
    n_rows, n_cols = arr.shape

    # Determine target keys
    if target_keys is None:
        if metadata and "valid_targets" in metadata:
            target_keys = [k[len("target_"):] if k.startswith("target_") else k for k in metadata["valid_targets"]]
        else:
            target_keys = [f"param_{i}" for i in range(n_cols)]

    # Determine which output column maps to which key
    col_to_key = target_keys[:n_cols] if n_cols <= len(target_keys) else [f"param_{i}" for i in range(n_cols)]

    # Build list of dicts
    dicts = []
    for r in range(n_rows):
        d = {k: None for k in target_keys}
        for j, key in enumerate(col_to_key):
            try:
                d[key] = float(arr[r, j])
            except Exception:
                d[key] = None
        dicts.append(d)

    return dicts

def run_simulation_from_predicted_dfs(
    predicted_dfs: pd.DataFrame,
    model,
    metadata,
    feature_cols: Optional[list],
    target_keys: Optional[list],
    removed_targets: Optional[dict],
    model_type: str = "cat",
    signal_col: str = SIGNAL_COLUMN,
    close_col: str = "close",
) -> Tuple[pd.DataFrame, dict]:

    df_hist = predicted_dfs[:PREDICT_WITH_SIGNAL_NUM_CANDLES].copy()
    df_live = predicted_dfs[PREDICT_WITH_SIGNAL_NUM_CANDLES:].copy()

    X_live = df_live[feature_cols].fillna(0)

    # 1) Predict variable targets
    param_dicts = _predict_param_dicts_from_model(
        model, metadata, X_live, target_keys=target_keys
    )

    # 2) Ensure list length matches df_live
    if len(param_dicts) != len(df_live):
        if len(param_dicts) == 1:
            param_dicts = param_dicts * len(df_live)
        else:
            pad_dict = {k: None for k in target_keys}
            target_n = len(df_live)
            param_dicts = (param_dicts[:target_n] + [pad_dict] * target_n)[:target_n]

    # ----------------------------------------------------------------------
    # ⭐ NEW PART: Combine model outputs with removed constant targets
    # ----------------------------------------------------------------------

    # Convert removed targets from "target_xxx" → "xxx"
    removed_clean = {}
    if removed_targets:
        for k, v in removed_targets.items():
            removed_clean[k] = v

    # Build the final full param dict for each row
    full_param_dicts = []

    for d in param_dicts:
        full = {}

        # Fill parameters in correct canonical order
        for key in target_keys:
            if key in d and d[key] is not None:
                # model predicted this parameter
                full[key] = d[key]
            elif key in removed_clean:
                # constant (removed) parameter value
                full[key] = removed_clean[key]
            else:
                # missing / unknown parameter → fallback
                full[key] = None

        full_param_dicts.append(full)

    # ----------------------------------------------------------------------
    # Use full params
    # ----------------------------------------------------------------------
    params_series = full_param_dicts

    df_result, metrics = simulate_trades_core(
        df=df_live,
        df_hist=df_hist,
        signal_col=signal_col,
        close_col=close_col,
        params=params_series,
    )

    df_result.attrs["sim_meta"] = {
        "model_type": model_type,
        "feature_cols": feature_cols,
        "target_keys": target_keys,
        "removed_targets": removed_clean,
        "n_live_rows": len(df_live),
        "n_hist_rows": len(df_hist),
    }

    return df_result, metrics

# ==================================================================================
# Configuration
# ==================================================================================

LOAD_HISTORICAL_DATA_FROM_CSV = True
USE_SAVED_PREDICTIONS = True   # ← switch this to False to retrain
USE_SAVED_LABELED_DATA = True
USE_TRAINED_MODEL = True

# ==================================================================================
# Main script
# ==================================================================================
if __name__ == "__main__":

    if LOAD_HISTORICAL_DATA_FROM_CSV and os.path.exists(os.path.join(LABEL_DIR, f"df_featured.csv")):
        print(f"[DEBUG] Loading historical data from CSV: {os.path.join(LABEL_DIR, f'df_featured.csv')}")
        df = pd.read_csv(
            os.path.join(LABEL_DIR, f"df_featured.csv"),
            parse_dates=['date'],
            index_col='date'
        )
        print(f"✅ Loaded {len(df)} candles from CSV")
    else:
        # Fetch historical data from exchange   
        exchange = ccxt.binance()
        timeframe = HISTORICAL_PRICES_TIMEFRAME
        limit = HISTORICAL_PRICES_LIMIT
        since = exchange.milliseconds() - TRAINING_WINDOW_MILLISECONDS

        print("Fetching data from Binance...")

        all_ohlcv = []

        while True:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv += ohlcv
            since = ohlcv[-1][0] + (5 * 60 * 1000)
            print(f"Fetched {len(all_ohlcv)} candles so far...")
            time.sleep(exchange.rateLimit / 1000)

        df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates('date').set_index('date').sort_index()
        print(f"✅ Loaded {len(df)} candles from Binance")
        df = make_features(df)
        df = make_labels(df, H=NUMBER_OF_CANDLES_AHEAD)
        print("Feature matrix shape:", df.shape)
        df.to_csv(os.path.join(LABEL_DIR, f"df_featured.csv"), index=True)

    def build_param_grid_with_relation(
            stakes, 
            stop_losses, 
            max_hold_hours
            ):
        """
        Build all combinations of stake_pct, stop_loss_pct, max_hold_hours,
        with the relation take_profit_pct = 2 * stop_loss_pct.
        """
        grid = []
        for s, sl, mh in product(stakes, stop_losses, max_hold_hours):
            grid.append({
                'stake_pct': float(s),
                'stop_loss_pct': float(sl),
                'take_profit_pct': float(2 * sl),
                'max_hold_hours': float(mh)
            })
        return grid
    
    # Define the ranges
    stake_values = [0.3, 0.5, 0.7]
    stop_loss_values = [0.005, 0.01, 0.02]
    max_hold_values = [1, 2, 4, 8, 16, 24]

    # Build all combinations keeping take_profit_pct = 2 * stop_loss_pct
    param_grid = build_param_grid_with_relation(
        stakes=stake_values,
        stop_losses=stop_loss_values,
        max_hold_hours=max_hold_values
    )

    MODEL_TYPE = 'cat'
    MODEL_CLS = CatBoostRegressor

    label_csv_path = os.path.join(LABEL_DIR, f"df_{MODEL_TYPE}_predictions.csv")

    if USE_SAVED_PREDICTIONS and os.path.exists(label_csv_path):
        print(f"[DEBUG] Loading predicted_dfs from: {label_csv_path}")
        predicted_dfs = pd.read_csv(
            label_csv_path,
            parse_dates=['date'],
            index_col='date'
        )
    else:
        # # Run rolling per-candle predictions
        predicted_dfs = rolling_train_predict_multi(
            df = df,
            model_cls=MODEL_CLS,
            model_type=MODEL_TYPE,
            model_params={'iterations': 500, 'verbose': False},
            features=get_features(df),
            target_col=TARGET_COLUMN,
            signal_col=SIGNAL_COLUMN,
            window=MAX_HISTORY_SIZE
            )
        
        # Define the path to save
        output_path = os.path.join(LABEL_DIR, f"df_{MODEL_TYPE}_predictions.csv")

        # Save the DataFrame
        predicted_dfs.to_csv(output_path, index=True)  # keep the index (dates)

        print(f"Saved rolling predictions to {output_path}")

    # -------------------------
    # Walk-forward labeling using model signals (pred column)
    # -------------------------
    
    dyn_path = f"{LABEL_DIR}/{MODEL_TYPE}_dyn_best_params_label.csv"
    df_dyn_best = None
    
    if USE_SAVED_LABELED_DATA and os.path.exists(dyn_path):
        if os.path.exists(dyn_path):
            print(f"[DEBUG] Loading df_dyn_best from: {dyn_path}")
            df_dyn_best = pd.read_csv(
                dyn_path,
                parse_dates=['date'],
                index_col='date'
            )
    else:
        # define the intervals you want to test (hours). Keep 24 for daily to compare
        intervals_to_test = (24,)

        #DEBUG
        # param_grid = param_grid[:1]  # limit to first 10 for faster testing
        ########

        summary_df, all_labels, all_results, best_hours, best_labels = label_and_evaluate_intervals(
            df=predicted_dfs,
            model_type=MODEL_TYPE,
            param_grid=param_grid,
            intervals_hours=intervals_to_test,
            timeframe_minutes=int(HISTORICAL_PRICES_TIMEFRAME.strip('m')),
        )

        # Save the summary table
        summary_path = f"{LABEL_DIR}/{MODEL_TYPE}_best_params_label_summary.csv"
        summary_df.to_csv(summary_path, index=True)
        print(f"\nSaved interval summary to {summary_path}")

        # Save the best interval labels (walk-forward labels DataFrame)
        if best_labels is not None:
            label_path = f"{LABEL_DIR}/{MODEL_TYPE}_best_params_label.csv"
            best_labels.to_csv(label_path, index=True)
            print(f"Saved best-interval labels to {label_path}")
        if best_hours is not None and best_hours in all_results:
            _, df_dyn_best, _ = all_results[best_hours]
            df_dyn_best.to_csv(dyn_path, index=True)
            print(f"Saved best-interval simulation DataFrame to {dyn_path}")

    multi_model, metadata = None, None
    
    if USE_TRAINED_MODEL:
        # Load model and metadata
        multi_model, metadata, _ = load_model(MODEL_TYPE)
    else:
        # -------------------------
        # Train multi-output model to predict best parameters
        # -------------------------
        multi_model, predictor, rmse_per_target, metadata = train_best_param_multi_model(df_dyn_best)
        # Save the trained multi-output model and metadata
        try:
            saved_path = save_model(multi_model, model_type=MODEL_TYPE, keep_count=2, metadata=metadata)
            print(f"Model saved to: {saved_path}")
        except Exception as e:
            print(f"Warning: failed to save model: {e}")

    # Determine target keys dynamically from metadata
    target_keys = metadata["target_keys"]

    removed_targets = metadata.get("removed_targets", {})

    # Determine features to use
    feature_cols = metadata["feature_cols"]

    df_result, metrics = run_simulation_from_predicted_dfs(
        df_dyn_best, multi_model, metadata, model_type=MODEL_TYPE,
        signal_col=SIGNAL_COLUMN, close_col="close",
        feature_cols=feature_cols, target_keys=target_keys,removed_targets=removed_targets)
    print(metrics)
