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
from dotenv import load_dotenv
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
    api_key: str,
    api_secret: str,
    model_dir=MODEL_DIR,
):
    tf_cfg = TIMEFRAMES[timeframe]

    featured_file = f"strategic_{symbol}_{timeframe}_{days}d_featured.csv"
    df_full = load_featured_df(featured_file)

    if df_full is None:
        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret must be provided")
        client = Client(api_key, api_secret, testnet=True)

        df_raw = download_historical_prices(symbol, tf_cfg.binance_interval, days, client)
        df_full = make_strategic_features(df_raw, tf_cfg)
        save_featured_df(df_full, featured_file)
        print(f"✅ Features saved: {featured_file}")

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
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train the StrategicML model.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, choices=list(TIMEFRAMES.keys()))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args()

    _api_key = os.getenv("BINANCE_TESTNET_FUTURES_API_KEY")
    _api_secret = os.getenv("BINANCE_TESTNET_FUTURES_API_SECRET")
    if not _api_key or not _api_secret:
        raise ValueError("BINANCE_TESTNET_FUTURES_API_KEY and BINANCE_TESTNET_FUTURES_API_SECRET must be set")

    run_training(
        symbol=args.symbol,
        days=args.days,
        timeframe=args.timeframe,
        api_key=_api_key,
        api_secret=_api_secret,
        model_dir=args.model_dir,
    )
