import argparse
import os
import warnings

import numpy as np
import pandas as pd
from binance.client import Client
from joblib import Parallel, delayed
from tqdm import tqdm

from mltrainingcore import (
    TARGET_COLUMN,
    SIGNAL_COLUMN,
    build_feature_dataset,
    get_features,
    simulate_trades_core,
    calculate_metrics,
)
from mlio import (
    MODEL_DIR,
    download_historical_prices,
    load_featured_df,
    save_featured_df,
    load_labels,
    save_labels,
)
from fancontrol.fanctl import fan_control
from tactical.tacticalml import TacticalML
from strategic.strategicml import StrategicML
from timeframe_config import TIMEFRAMES

warnings.filterwarnings("ignore")

USE_SAVED_FEATURED = False
USE_SAVED_PREDICTIONS = False

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_DAYS = 45
DEFAULT_TIMEFRAME = "5m"
TRAINING_FRACTION = 0.8

DEFAULT_PARAMS = {
    "stake_long_frac": 0.1,
    "stake_short_frac": 0.05,
    "stop_loss_frac": 0.02,
    "take_profit_frac": 0.04,
    "max_hold_hours": 4.0,
}


def _predict_chunk(df_full, features, tf_cfg, model_params, indices):
    from tactical.tacticalml import TacticalML as _TacticalML
    tactical = _TacticalML(model_params=model_params, tf_cfg=tf_cfg)
    window = tf_cfg.max_history_candles
    results = []
    for i in indices:
        df_train = df_full.iloc[i - window : i - 1]
        df_pred_row = df_full.iloc[[i]]
        sig = tactical.fit_and_predict(df_train, df_pred_row, features)
        results.append((i, sig.prediction))
    return results


def _rolling_tactical_predictions(df_full: pd.DataFrame, tf_cfg) -> pd.DataFrame:
    # iterations=100 (vs 300 live) is sufficient for offline backtest and gives ~3x speedup.
    # thread_count=1 prevents core thrashing when N workers each spawn N CatBoost threads.
    model_params = {"iterations": 100, "verbose": False, "thread_count": 1}

    features = get_features(df_full)
    n = len(df_full)
    window = tf_cfg.max_history_candles

    if window >= n:
        window = max(50, n // 3)
        print(f"[TacticalML] Auto-adjusted window={window} for dataset length={n}")

    all_indices = list(range(window, n))
    n_jobs = min(os.cpu_count() or 4, 8)
    chunks = [arr.tolist() for arr in np.array_split(all_indices, n_jobs)]

    print(f"Running {len(all_indices)} walk-forward predictions on {n_jobs} workers...")

    results_nested = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_predict_chunk)(df_full, features, tf_cfg, model_params, chunk)
        for chunk in chunks
    )

    flat = sorted(
        [item for chunk_result in results_nested for item in chunk_result],
        key=lambda x: x[0],
    )
    preds = [p for _, p in flat]

    print("Predictions complete.")

    df_out = df_full.iloc[window:].copy()
    df_out[SIGNAL_COLUMN] = preds
    return df_out.round(5)


def _build_strategic_param_list(
    df_test: pd.DataFrame,
    df_raw_5m: pd.DataFrame,
    strategic: StrategicML,
    strategic_tf_cfg,
) -> list:
    if not strategic.is_ready:
        print("WARNING: StrategicML has no model loaded — using DEFAULT_PARAMS for all test candles.")
        return [DEFAULT_PARAMS.copy() for _ in range(len(df_test))]

    df_1h = (
        df_raw_5m.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )

    max_hist = strategic_tf_cfg.max_history_candles
    min_rows_for_features = 250
    hist_window = max(max_hist, min_rows_for_features)
    param_list = []

    for ts in tqdm(df_test.index, desc="Strategic decisions", unit="candle"):
        df_hist_1h = df_1h[df_1h.index <= ts].tail(hist_window)

        if len(df_hist_1h) < min_rows_for_features:
            param_list.append(DEFAULT_PARAMS.copy())
            continue

        try:
            decision = strategic.predict(df_hist_1h)
        except Exception as exc:
            print(f"WARNING: StrategicML.predict failed at {ts}: {exc} — using defaults")
            param_list.append(DEFAULT_PARAMS.copy())
            continue

        params = {
            "stake_long_frac": decision.stake_long_frac if decision.allow_trading else 0.0,
            "stake_short_frac": decision.stake_short_frac if decision.allow_trading else 0.0,
            "stop_loss_frac": decision.stop_loss_frac,
            "take_profit_frac": decision.take_profit_frac,
            "max_hold_hours": decision.max_hold_hours,
        }
        param_list.append(params)

    return param_list


def run_predictions_only(
    symbol: str,
    days: int,
    timeframe: str,
) -> tuple:
    tf_cfg = TIMEFRAMES[timeframe]

    featured_file = f"dual_{symbol}_{timeframe}_{days}d_featured.csv"
    df_full = load_featured_df(featured_file) if USE_SAVED_FEATURED else None

    client = Client()
    df_raw = download_historical_prices(symbol, tf_cfg.binance_interval, days, client)

    if df_full is None:
        df_full = build_feature_dataset(df_raw, tf_cfg)
        save_featured_df(df_full, featured_file)
        print(f"Features saved: {featured_file}")

    print(f"Full featured dataset: {len(df_full)} rows")

    pred_file = f"dual_{symbol}_{timeframe}_{days}d_predictions.csv"
    df_predictions = load_featured_df(pred_file) if USE_SAVED_PREDICTIONS else None

    if df_predictions is None:
        df_predictions = _rolling_tactical_predictions(df_full, tf_cfg)
        save_featured_df(df_predictions, pred_file)
        print(f"Predictions saved: {pred_file}")

    print(f"Predictions dataset: {len(df_predictions)} rows")
    return df_predictions, df_raw


def run_simulation(symbol: str, days: int, timeframe: str, model_dir=MODEL_DIR):
    tf_cfg = TIMEFRAMES[timeframe]
    strategic_tf_cfg = TIMEFRAMES["1h"]

    df_predictions, df_raw = run_predictions_only(symbol, days, timeframe)

    n_total = len(df_predictions)
    n_train = int(np.floor(n_total * TRAINING_FRACTION))
    df_test = df_predictions.iloc[n_train:].copy()
    print(f"Split: total={n_total}, train={n_train}, test={len(df_test)}")

    strategic = StrategicML(model_dir=model_dir, tf_cfg=strategic_tf_cfg)
    param_list = _build_strategic_param_list(df_test, df_raw, strategic, strategic_tf_cfg)

    df_hist = df_predictions.iloc[: tf_cfg.adaptive_history_candles].copy()

    df_result, metrics = simulate_trades_core(
        df=df_test,
        df_hist=df_hist,
        signal_col=SIGNAL_COLUMN,
        tf_cfg=tf_cfg,
        param_list=param_list,
        close_col="close",
    )

    sim_file = f"dual_{symbol}_{timeframe}_{days}d_final_test_sim.csv"
    save_labels(df_result, sim_file)
    print(f"Simulation saved: {sim_file}")

    trades = df_result.attrs.get("trades", [])
    _, full_metrics = calculate_metrics(trades, metrics.get("final_wallet", 1.0))

    print("\n" + "=" * 60)
    print("DUAL-ML SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Final wallet:      {full_metrics.get('final_wallet', 1.0):.4f}")
    print(f"  Trades:            {full_metrics.get('trades_count', 0)}")
    print(f"  Win rate:          {full_metrics.get('win_rate', 0.0):.2%}")
    print(f"  Mean return/trade: {full_metrics.get('mean_return', 0.0):.4%}")
    print(f"  Objective score:   {full_metrics.get('objective_score', 0.0):.4f}")
    print("=" * 60)

    return df_result, full_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-ML walk-forward backtest.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument(
        "--timeframe", default=DEFAULT_TIMEFRAME, choices=list(TIMEFRAMES.keys())
    )
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument(
        "--fan-control",
        action="store_true",
        help="Enable GPIO fan control for CPU cooling during simulation.",
    )
    parser.add_argument(
        "--fan-temp-threshold",
        type=float,
        default=None,
        help="CPU temperature threshold (°C) to trigger the fan. "
        "Set low (e.g. 30) for testing. Requires --fan-control.",
    )
    args = parser.parse_args()

    with fan_control(
        enable=args.fan_control,
        temp_threshold=args.fan_temp_threshold,
    ):
        run_simulation(
            symbol=args.symbol,
            days=args.days,
            timeframe=args.timeframe,
            model_dir=args.model_dir,
        )
