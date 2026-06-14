"""
Strategic ML training pipeline.

Usage:
    python strategic/strategictraining.py
    python strategic/strategictraining.py --symbol BTCUSDT --days 365 --timeframe 1h

Designed to be run from cron:
    0 2 * * * conda run -n tradingbot python /path/to/strategic/strategictraining.py >> training.log 2>&1
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from binance.client import Client
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlio import (
    MODEL_DIR,
    download_historical_prices,
    load_featured_df,
    save_featured_df,
    load_labels,
    save_labels,
    save_model,
    get_latest_model_paths,
    load_model,
)
from mltrainingcore import SIGNAL_COLUMN
from mltraining import walkforward_label_forward_windows, build_param_grid
from fancontrol.fanctl import fan_control
from strategic.strategicfeatures import (
    make_strategic_features,
    get_strategic_features,
    EXTREME_VOL_RATIO,
    HIGH_VOL_RATIO,
)
from timeframe_config import TIMEFRAMES, TimeframeConfig
from strategic.strategicml import STRATEGIC_MODEL_TYPE_PREFIX

warnings.filterwarnings("ignore")

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_DAYS = 365
DEFAULT_TIMEFRAME = "1h"
TRAINING_FRACTION = 0.8


def _build_strategic_labels(df: pd.DataFrame, tf_cfg: TimeframeConfig) -> pd.DataFrame:
    df = df.copy()

    vol_ratio = df["vol_short"] / df["vol_long"].clip(lower=1e-8)
    df["allow_trading"] = (vol_ratio < EXTREME_VOL_RATIO).astype(float)

    regime_leverage = {"trend": 5.0, "high_vol": 2.0, "chop": 1.0}
    df["recommended_leverage"] = df["regime"].map(regime_leverage).fillna(1.0)

    df["max_exposure_frac"] = np.where(
        vol_ratio >= HIGH_VOL_RATIO, 0.3, np.where(vol_ratio >= 1.0, 0.6, 1.0)
    )

    df["stake_long_frac"] = np.where(df["regime"] == "trend", 0.2, 0.1)
    df["stake_short_frac"] = np.where(df["regime"] == "trend", 0.1, 0.05)

    df["stop_loss_frac"] = np.where(
        df["regime"] == "high_vol", 0.03, np.where(df["regime"] == "trend", 0.015, 0.02)
    )
    df["take_profit_frac"] = df["stop_loss_frac"] * 2.0

    df["max_hold_hours"] = np.where(
        df["regime"] == "trend", 8.0, np.where(df["regime"] == "high_vol", 2.0, 4.0)
    )

    return df.dropna()


def _build_strategic_labels_from_simulation(
    df: pd.DataFrame,
    df_5m_predictions: pd.DataFrame,
    tf_cfg: TimeframeConfig,
) -> pd.DataFrame:
    df = df.copy()

    vol_ratio = df["vol_short"] / df["vol_long"].clip(lower=1e-8)
    df["allow_trading"] = (vol_ratio < EXTREME_VOL_RATIO).astype(float)

    regime_leverage = {"trend": 5.0, "high_vol": 2.0, "chop": 1.0}
    df["recommended_leverage"] = df["regime"].map(regime_leverage).fillna(1.0)

    df["max_exposure_frac"] = np.where(
        vol_ratio >= HIGH_VOL_RATIO, 0.3, np.where(vol_ratio >= 1.0, 0.6, 1.0)
    )

    pred_1h = df_5m_predictions[SIGNAL_COLUMN].resample("1h").last().rename(SIGNAL_COLUMN)
    df = df.join(pred_1h, how="inner")
    df = df.dropna(subset=[SIGNAL_COLUMN])

    param_grid = build_param_grid(
        stake_short=[0.05, 0.10, 0.15],
        stake_long=[0.10, 0.15, 0.25],
        stop_loss=[0.01, 0.015, 0.02, 0.03, 0.05],
        max_hold_hours=[2, 4, 8, 12, 24],
        take_profit_mult=2.0,
    )

    labels_df = walkforward_label_forward_windows(
        df=df,
        param_grid=param_grid,
        signal_col=SIGNAL_COLUMN,
        window_hours=24.0,
        step_hours=2.4,
        tf_cfg=tf_cfg,
    )

    if labels_df.empty:
        raise RuntimeError(
            f"Walk-forward param search produced no labels — dataset too short "
            f"({len(df)} rows). Need at least {tf_cfg.adaptive_history_candles + 24} rows."
        )

    labels_for_merge = labels_df.reset_index()[["date", "best_param"]]

    df.index.name = "date"
    df_reset = df.reset_index()

    df_merged = pd.merge_asof(
        df_reset.sort_values("date"),
        labels_for_merge.sort_values("date"),
        on="date",
        direction="backward",
    ).set_index("date").sort_index()

    df_merged = df_merged.dropna(subset=["best_param"])

    param_keys = ["stake_long_frac", "stake_short_frac", "stop_loss_frac", "take_profit_frac", "max_hold_hours"]
    for key in param_keys:
        df_merged[key] = df_merged["best_param"].apply(
            lambda d: d.get(key) if isinstance(d, dict) else np.nan
        )

    df_merged = df_merged.drop(columns=["best_param"])
    return df_merged.dropna(subset=param_keys)


def _train_strategic_model(df_train: pd.DataFrame):
    target_cols = [
        "recommended_leverage",
        "max_exposure_frac",
        "stake_long_frac",
        "stake_short_frac",
        "stop_loss_frac",
        "take_profit_frac",
        "max_hold_hours",
    ]

    available_targets = [c for c in target_cols if c in df_train.columns]

    valid_targets = []
    removed_targets = {}
    for col in available_targets:
        if df_train[col].nunique() > 1:
            valid_targets.append(col)
        else:
            removed_targets[col] = float(df_train[col].iloc[0])

    if not valid_targets:
        raise RuntimeError("All strategic targets are constant — check label generation.")

    feature_cols = get_strategic_features(df_train)
    feature_cols = [c for c in feature_cols if c not in target_cols]

    X = df_train[feature_cols].fillna(0)
    Y = df_train[valid_targets].fillna(0)

    n_train = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
    Y_train, Y_val = Y.iloc[:n_train], Y.iloc[n_train:]

    base = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train, Y_train)

    preds = model.predict(X_val)
    print("\n📈 Validation RMSE per strategic target:")
    for i, key in enumerate(valid_targets):
        rmse = np.sqrt(np.mean((Y_val.iloc[:, i].values - preds[:, i]) ** 2))
        print(f"   {key}: {rmse:.6f}")

    metadata = {
        "feature_cols": feature_cols,
        "target_keys": available_targets,
        "valid_targets": valid_targets,
        "removed_targets": removed_targets,
    }

    return model, metadata


def run_training(
    symbol: str,
    days: int,
    timeframe: str,
    model_dir=MODEL_DIR,
    df_5m_predictions: pd.DataFrame = None,
):
    tf_cfg = TIMEFRAMES[timeframe]

    featured_file = f"strategic_{symbol}_{timeframe}_{days}d_featured.csv"
    df_full = load_featured_df(featured_file)

    if df_full is None:
        client = Client()

        df_raw = download_historical_prices(symbol, tf_cfg.binance_interval, days, client)
        df_full = make_strategic_features(df_raw, tf_cfg)
        save_featured_df(df_full, featured_file)
        print(f"✅ Features saved: {featured_file}")

    if df_5m_predictions is not None:
        labeled_file = f"strategic_{symbol}_{timeframe}_{days}d_sim_labeled.csv"
        df_labeled = load_labels(labeled_file)

        if df_labeled is None:
            df_labeled = _build_strategic_labels_from_simulation(df_full, df_5m_predictions, tf_cfg)
            save_labels(df_labeled, labeled_file)
            print(f"✅ Simulation-driven labels saved: {labeled_file}")
    else:
        labeled_file = f"strategic_{symbol}_{timeframe}_{days}d_labeled.csv"
        df_labeled = load_labels(labeled_file)

        if df_labeled is None:
            df_labeled = _build_strategic_labels(df_full, tf_cfg)
            save_labels(df_labeled, labeled_file)
            print(f"✅ Labels saved: {labeled_file}")

    print(f"📊 Labeled dataset: {len(df_labeled)} rows")

    n_train = int(len(df_labeled) * TRAINING_FRACTION)
    df_train = df_labeled.iloc[:n_train]
    df_test = df_labeled.iloc[n_train:]
    print(f"   Train: {len(df_train)}, Test: {len(df_test)}")

    model, metadata = _train_strategic_model(df_train)

    saved_path = save_model(
        model=model,
        metadata=metadata,
        model_type=STRATEGIC_MODEL_TYPE_PREFIX,
        model_dir=model_dir,
    )
    print(f"\n✅ Strategic model saved: {saved_path}")
    return saved_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the StrategicML model.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, choices=list(TIMEFRAMES.keys()))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument(
        "--fan-control",
        action="store_true",
        help="Enable GPIO fan control for CPU cooling during training.",
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
        run_training(
            symbol=args.symbol,
            days=args.days,
            timeframe=args.timeframe,
            model_dir=args.model_dir,
        )
