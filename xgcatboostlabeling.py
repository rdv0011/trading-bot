# xgcatboostlabeling.py
# Produces trained XGBoost/CatBoost models and a labeled historical price dataset

import pandas as pd
import numpy as np
import warnings
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
from xgcatboostcore import make_features, make_labels, predict_param_dicts_from_model, resolve_model_class
from xgcatboostcore import get_features, simulate_trades_core, create_model
from xgcatboostcore import TARGET_COLUMN, SEED_BASE, SIGNAL_COLUMN
from xgcatboostcore import PREDICT_WITH_SIGNAL_NUM_CANDLES, OBJECTIVE_METRIC, NUMBER_OF_CANDLES_AHEAD
from mlio import LABEL_DIR, download_historical_prices, load_featured_df, load_labels
from mlio import load_model, save_model, save_featured_df, save_labels, get_latest_model_paths
from itertools import product
import ast
import os
from typing import Tuple

warnings.filterwarnings("ignore")

SYMBOL = 'BTC/USDT'
WHOLE_WINDOW_DAYS = 60  # total days to fetch (train + test)
TRAINING_FRACTION = 0.8  # fraction used for training after labeling
WHOLE_WINDOW_MILLISECONDS = WHOLE_WINDOW_DAYS * 24 * 60 * 60 * 1000

SAVE_FINAL_MODEL = True # Models are saved to models/ directory
# Metaparameters as constants
MAX_HISTORY_SIZE = 600
HISTORICAL_PRICES_LENGTH = 500
HISTORICAL_PRICES_LIMIT = 1000
HISTORICAL_PRICES_TIMEFRAME = "5m"
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
            "best_metric": round(best_metric, 5),
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

def rolling_train_predict_multi(
        df,
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

    model_cls = resolve_model_class(model_type)
    preds = []
    mdl = None
    retrain_every = 1  # Retrain every N candles for speed

    # Train model on rolling window
    for i in tqdm(range(w, n), desc=f"Labeling with predictions {model_type.upper()}", unit="candle"):
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

    df_out = df_out.round(5)

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
        step = min(12, window_size)
        
        print(f"\n--- Labeling for {hours}h horizon ({window_size} candles), step={step} ---")

        # --- Create labels df (date-index from the beginning) ---
        labels_df = walkforward_label_forward_windows(
            df=df,
            param_grid=param_grid,
            signal_col=signal_col,
            window_size=window_size,
            timeframe_minutes=timeframe_minutes,
            step=step,
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
            f"{model_type}_objective": round(metrics_dyn.get(OBJECTIVE_METRIC, np.nan), 5),
            f"{model_type}_final_wallet": round(metrics_dyn.get('final_wallet', np.nan), 5),
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
    target_keys = []
    df["best_param"].dropna().apply(
        lambda d: [target_keys.append(k) for k in d.keys() if k not in target_keys]
    )

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

def run_simulation_from_predicted_dfs(
    predicted_dfs: pd.DataFrame,
    model,
    metadata,
    model_type: str = "cat",
    signal_col: str = SIGNAL_COLUMN,
    close_col: str = "close",
) -> Tuple[pd.DataFrame, dict]:

    df_hist = predicted_dfs[:PREDICT_WITH_SIGNAL_NUM_CANDLES].copy()
    df_live = predicted_dfs[PREDICT_WITH_SIGNAL_NUM_CANDLES:].copy()

    target_keys = metadata["target_keys"]
    removed_targets = metadata.get("removed_targets", {})
    feature_cols = metadata["feature_cols"]

    X_live = df_live[feature_cols].fillna(0)

    # 1) Predict variable targets
    param_dicts = predict_param_dicts_from_model(
        model, metadata, X_live
    )

    df_result, metrics = simulate_trades_core(
        df=df_live,
        df_hist=df_hist,
        signal_col=signal_col,
        close_col=close_col,
        params=param_dicts,
    )

    df_result.attrs["sim_meta"] = {
        "model_type": model_type,
        "feature_cols": feature_cols,
        "target_keys": target_keys,
        "removed_targets": removed_targets,
        "n_live_rows": len(df_live),
        "n_hist_rows": len(df_hist),
    }

    return df_result, metrics

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

def build_feature_dataset(df_raw, horizon):
    df = make_features(df_raw)
    df = make_labels(df, H=horizon)
    return df

# ==================================================================================
# Configuration
# ==================================================================================

LOAD_HISTORICAL_DATA_FROM_CSV = False
USE_SAVED_ROLLING_PREDICTIONS = False
USE_SAVED_LABELED_DATA = False
USE_SAVED_TRAINED_MODEL = False
MODEL_TYPE = 'cat'

# ==================================================================================
# Main script
# ==================================================================================
if __name__ == "__main__":

    # Load or fetch full historical data
    featured_filename = "df_featured.csv"
    df_full = None
    if LOAD_HISTORICAL_DATA_FROM_CSV:
        df_full = load_featured_df(featured_filename)

    if df_full is None:
        df_raw = download_historical_prices(
            SYMBOL,
            HISTORICAL_PRICES_TIMEFRAME,
            HISTORICAL_PRICES_LIMIT,
            WHOLE_WINDOW_MILLISECONDS
        )

        df_full = build_feature_dataset(df_raw, NUMBER_OF_CANDLES_AHEAD)
        save_featured_df(df_full, featured_filename)

    # Generate rolling predictions on FULL dataset (Label engine)
    rolling_pred_filename = f"df_{MODEL_TYPE}_predictions_full.csv"

    if USE_SAVED_ROLLING_PREDICTIONS:
        predicted_dfs_full = load_featured_df(rolling_pred_filename)
    else:
        predicted_dfs_full = None

    if predicted_dfs_full is None:
        predicted_dfs_full = rolling_train_predict_multi(
            df=df_full,
            model_type=MODEL_TYPE,
            model_params={'iterations': 500, 'verbose': False},
            features=get_features(df_full),
            target_col=TARGET_COLUMN,
            signal_col=SIGNAL_COLUMN,
            window=MAX_HISTORY_SIZE
        )

        save_featured_df(predicted_dfs_full, rolling_pred_filename)

    # Meta-parameter optimization (walk-forward) on FULL predicted dataset ---
    dyn_filename = f"{MODEL_TYPE}_dyn_best_params_label_full.csv"

    if USE_SAVED_LABELED_DATA:
        print(f"[DEBUG] Loading labeled dataset from: {dyn_filename}")
        df_dyn_best_full = load_labels(dyn_filename)
    else:
        # build param grid
        stake_values = [0.3, 0.5, 0.7]
        stop_loss_values = [
            0.0025,
            0.005,
            0.01,
            0.015,
            0.02,
            0.03,
            0.05,
            0.075,
            0.10
        ]
        max_hold_values = [1, 2, 4, 8, 16, 24]
        param_grid = build_param_grid_with_relation(
            stakes=stake_values,
            stop_losses=stop_loss_values,
            max_hold_hours=max_hold_values
        )

        summary_df, all_labels, all_results, best_hours, best_labels = label_and_evaluate_intervals(
            df=predicted_dfs_full,
            model_type=MODEL_TYPE,
            param_grid=param_grid,
            intervals_hours=(24,),
            timeframe_minutes=int(HISTORICAL_PRICES_TIMEFRAME.strip('m')),
        )

        summary_filename = f"{MODEL_TYPE}_best_params_label_summary_full.csv"
        save_labels(summary_df, summary_filename)
        print(f"Saved labeling summary to {summary_filename}")

        if best_labels is not None:
            labels_filename = f"{MODEL_TYPE}_best_params_label_full.csv"
            save_labels(best_labels, labels_filename)
            print(f"Saved best-interval labels to {labels_filename}")
        if best_hours is not None and best_hours in all_results:
            _, df_dyn_best_full, _ = all_results[best_hours]
            dyn_filename = f"{MODEL_TYPE}_dyn_best_params_label_full.csv"
            save_labels(df_dyn_best_full, dyn_filename)
            print(f"Saved full dyn best params DF to {dyn_filename}")

    # At this point df_dyn_best_full is the fully labeled dataset (features + rolling predictions + best_param per row)

    # Split labeled dataset into train / test (chronological split) ---
    df_labeled = df_dyn_best_full.copy()
    n_total = len(df_labeled)
    n_train = int(np.floor(n_total * TRAINING_FRACTION))

    df_train = df_labeled.iloc[:n_train].copy()
    df_test = df_labeled.iloc[n_train:].copy()

    print(f"Split labeled dataset: total={n_total}, train={len(df_train)}, test={len(df_test)}")

    # Train meta-model on TRAIN portion (predict best_param) ---
    multi_model = None
    metadata = None

    if USE_SAVED_TRAINED_MODEL:
        try:
            model_path, meta_path = get_latest_model_paths(MODEL_TYPE)
            multi_model, metadata = load_model(model_path, meta_path)
            print("Loaded trained meta-model from disk")
        except Exception as e:
            print(f"Could not load saved model: {e} — will train new one.")
            multi_model = None

    if multi_model is None:
        multi_model, predictor, rmse_per_target, metadata = train_best_param_multi_model(df_train)
        try:
            saved_path = save_model(model=multi_model, metadata=metadata, model_type=MODEL_TYPE)
            print(f"Model saved to: {saved_path}")
        except Exception as e:
            print(f"Warning: failed to save model: {e}")

    # Predict meta-parameters on TEST set and simulate trading on TEST only ---
    df_result_test, metrics_test = run_simulation_from_predicted_dfs(
        predicted_dfs=df_test,
        model=multi_model,
        metadata=metadata,
        model_type=MODEL_TYPE,
        signal_col=SIGNAL_COLUMN,
        close_col="close",
    )

    print("=== Final TEST set simulation metrics ===")
    print(metrics_test)

    # Save final test sim
    try:
        out_sim = os.path.join(LABEL_DIR, f"{MODEL_TYPE}_final_test_sim.csv")
        df_result_test.to_csv(out_sim)
        print(f"Saved final test simulation to {out_sim}")
    except Exception:
        pass

    print("Pipeline completed: labeling (full) → split → train → test simulation (final)")
