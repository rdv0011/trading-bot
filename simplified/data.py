"""
Data pipeline for Dual-ML Bitcoin Trading Bot.
Handles: historical data download, feature engineering, label generation,
train/validation chronological split, and data persistence.
Self-contained - no dependencies on original modules.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from datetime import datetime
import logging

from binance.client import Client

from config import (
    SYMBOL, TACTICAL_TF, STRATEGIC_TF,
    HISTORY_DAYS, TRAIN_FRACTION, LABEL_HORIZON,
    FEATURE_LAGS, EMA_SPANS, ATR_PERIOD,
    MODEL_DIR, STRATEGIC_TARGET_COLS,
    REGIME_LEVERAGE, REGIME_STAKE_LONG, REGIME_STAKE_SHORT,
    REGIME_STOP_LOSS, TAKE_PROFIT_MULT, REGIME_MAX_HOLD,
)

# ── Constants ───────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Timeframe config (inlined from original timeframe_config.py)
TIMEFRAME_CONFIG = {
    "15m": {
        "binance_interval": Client.KLINE_INTERVAL_15MINUTE,
        "minutes": 15,
        "candles_per_day": 96,
        "label_horizon_candles": 4,
        "adaptive_history_candles": 200,
        "ema_spans": [5, 10, 20, 50, 100],
    },
    "1h": {
        "binance_interval": Client.KLINE_INTERVAL_1HOUR,
        "minutes": 60,
        "candles_per_day": 24,
        "label_horizon_candles": 1,
        "adaptive_history_candles": 100,
        "ema_spans": [5, 10, 20, 50, 100],
    },
}


# ── Download Historical Prices ─────────────────────────────────────────
def download_historical(
    symbol: str = SYMBOL,
    days: int = HISTORY_DAYS,
    timeframe: str = TACTICAL_TF,
    testnet: bool = False,
) -> pd.DataFrame:
    """
    Download historical prices from Binance.
    Returns DataFrame with raw OHLCV data.

    Defaults to the LIVE Binance API (public klines, no auth needed) because
    Binance TESTNET only retains ~28 days of history regardless of request.
    Pass testnet=True only when deep history is not required (live trading).
    """
    tf_cfg = TIMEFRAME_CONFIG[timeframe]
    interval = tf_cfg["binance_interval"]

    target_candles = days * tf_cfg["candles_per_day"]
    print(f"Downloading {days} days of {timeframe} data for {symbol} ...")
    if testnet:
        client = Client(testnet=True)
    else:
        # Public live Binance client: public klines need no API keys and
        # expose far deeper history than testnet (~90+ days vs ~28 days).
        client = Client("", "")

    interval_ms = {
        Client.KLINE_INTERVAL_1MINUTE: 60_000,
        Client.KLINE_INTERVAL_15MINUTE: 15 * 60_000,
        Client.KLINE_INTERVAL_1HOUR: 3600_000,
        Client.KLINE_INTERVAL_4HOUR: 4 * 3600_000,
        Client.KLINE_INTERVAL_1DAY: 86_400_000,
    }.get(interval, 60_000)

    # Paginate backwards from now in batches of up to 1000 candles
    # (Binance single-request limit). Each batch ends at `end_ms`, then we
    # step back to the candle before this batch's first open and repeat.
    batch_limit = 1000
    end_ms = int(datetime.now().timestamp() * 1000)
    all_rows = []
    fetched = 0

    while fetched < target_candles and end_ms > 0:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=batch_limit,
            endTime=end_ms,
        )
        if not klines:
            break
        all_rows.extend(klines)
        fetched += len(klines)
        end_ms = int(klines[0][0]) - interval_ms
        if len(klines) < batch_limit:
            break

    df = pd.DataFrame(all_rows, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    if df.empty:
        raise RuntimeError(f"No data returned for {symbol} {timeframe}")

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Keep only OHLCV
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    # Deduplicate overlapping batch boundaries, sort chronologically,
    # then trim to the most recent `target_candles`.
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df = df.tail(target_candles)

    print(f"  Downloaded {len(df)} candles: {df.index[0].date()} to {df.index[-1].date()}")
    return df


# ── Feature Engineering ────────────────────────────────────────────────
def make_features_df(
    df: pd.DataFrame,
    timeframe: str = TACTICAL_TF,
) -> pd.DataFrame:
    """
    Engineer features: returns, EMAs, ATR, volatility, cyclical time, regime.
    Based on original mltrainingcore.make_features.
    """
    df = df.copy()
    tf_cfg = TIMEFRAME_CONFIG[timeframe]
    ema_spans = tf_cfg["ema_spans"]

    # Returns and lags
    df['ret1'] = df['close'].pct_change(1)
    for l in FEATURE_LAGS:
        df[f'ret_lag_{l}'] = df['ret1'].shift(l)

    # EMA features for trend detection
    for span in ema_spans:
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        df[f'ema_diff_{span}'] = df[f'ema_{span}'] - df['close']

    # ATR and volatility
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.abs(df['high'] - df['close'].shift(1)),
        np.abs(df['low'] - df['close'].shift(1)),
    )
    df['atr14'] = df['tr'].rolling(ATR_PERIOD).mean()
    df['vol_12'] = df['ret1'].rolling(12).std()
    df['vol_48'] = df['ret1'].rolling(48).std()

    # Cyclical time encoding
    hours = df.index.hour + df.index.minute / 60
    dows = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dows / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dows / 7)

    # Regime detection
    df['regime'] = df.apply(_detect_regime, axis=1)

    # Drop NaN rows from feature engineering
    df = df.dropna().round(5)

    return df


def _detect_regime(row) -> str:
    """Detect regime: trend, chop, high_vol (from original mltrainingcore)."""
    atr = max(row["atr14"], 1e-8)
    trend_strength = abs(row["ema_20"] - row["ema_100"]) / atr
    vol_ratio = row["vol_12"] / max(row["vol_48"], 1e-8)

    if trend_strength < 0.4:
        return "chop"
    if vol_ratio > 1.4:
        return "high_vol"
    return "trend"


# ── Strategic Feature & Label Engineering (1h) ─────────────────────────
def add_strategic_features_df(
    df: pd.DataFrame,
    timeframe: str = STRATEGIC_TF,
) -> pd.DataFrame:
    """
    Add strategic features (volatility ratio, drawdown, ATR%) on top of the
    base features. Mirrors legacy strategic/strategicfeatures.make_strategic_features.
    Returns df with base features + strategic features, NaN rows dropped.
    """
    df = df.copy()

    df["vol_short"] = df["ret1"].rolling(24).std()
    df["vol_long"] = df["ret1"].rolling(168).std()
    df["vol_ratio_strategic"] = df["vol_short"] / df["vol_long"].clip(lower=1e-8)

    rolling_max = df["close"].rolling(48).max()
    df["drawdown"] = (df["close"] - rolling_max) / rolling_max.clip(lower=1e-8)
    df["max_drawdown_window"] = df["drawdown"].rolling(48).min()

    df["atr_pct"] = df["atr14"] / df["close"].clip(lower=1e-8)
    df["atr_pct_ma"] = df["atr_pct"].rolling(168).mean()
    df["atr_pct_ratio"] = df["atr_pct"] / df["atr_pct_ma"].clip(lower=1e-8)

    return df.dropna().round(5)


def make_strategic_labels_df(
    df: pd.DataFrame,
    timeframe: str = STRATEGIC_TF,
) -> pd.DataFrame:
    """
    Build deterministic trade-parameter labels for the strategic model from
    regime + volatility. Mirrors legacy _build_strategic_labels. Non-leaking:
    each row's params depend only on that row's regime/volatility.
    """
    df = df.copy()

    vol_ratio = df["vol_ratio_strategic"]

    df["recommended_leverage"] = df["regime"].map(REGIME_LEVERAGE).fillna(1.0)
    df["max_exposure_frac"] = np.where(
        vol_ratio >= 1.6, 0.3, np.where(vol_ratio >= 1.0, 0.6, 1.0)
    )
    df["stake_long_frac"] = df["regime"].map(REGIME_STAKE_LONG).fillna(0.1)
    df["stake_short_frac"] = df["regime"].map(REGIME_STAKE_SHORT).fillna(0.05)
    df["stop_loss_frac"] = df["regime"].map(REGIME_STOP_LOSS).fillna(0.02)
    df["take_profit_frac"] = df["stop_loss_frac"] * TAKE_PROFIT_MULT
    df["max_hold_hours"] = df["regime"].map(REGIME_MAX_HOLD).fillna(4.0)

    return df.dropna()


# ── Label Generation (Future Return) ───────────────────────────────────
def make_labels_df(
    df: pd.DataFrame,
    timeframe: str = TACTICAL_TF,
) -> pd.DataFrame:
    """
    Add future return column for supervised learning.
    Based on original mltrainingcore.make_labels.
    """
    df = df.copy()
    horizon = TIMEFRAME_CONFIG[timeframe]["label_horizon_candles"]

    df['future_close'] = df['close'].shift(-horizon)
    df['future_ret'] = (df['future_close'] / df['close']) - 1.0

    # Drop last H rows (no future data)
    df = df.iloc[:-horizon] if horizon > 0 else df

    return df.dropna().round(5)


# ── Train / Validation Split ───────────────────────────────────────────
def train_val_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRACTION,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/validation split.
    No shuffling - preserves time order.
    """
    n = len(df)
    n_train = int(np.floor(n * train_frac))

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:].copy()

    print(f"Train/Val split: total={n}, train={len(df_train)}, val={len(df_val)}")
    if len(df_train) > 0:
        print(f"  Train: {df_train.index[0].date()} to {df_train.index[-1].date()}")
    if len(df_val) > 0:
        print(f"  Val:   {df_val.index[0].date()} to {df_val.index[-1].date()}")

    return df_train, df_val


# ── Feature Column Selection ───────────────────────────────────────────
def get_feature_cols(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Return list of numeric column names to use as features.
    Based on original mltrainingcore.get_features.
    """
    if exclude is None:
        exclude = ['future_close', 'future_ret', 'regime']

    if df is None or len(df) == 0:
        return []

    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]


# ── Data Persistence ───────────────────────────────────────────────────
def save_featured_df(df: pd.DataFrame, filename: str) -> Path:
    """Save featured DataFrame to CSV."""
    path = DATA_DIR / filename
    df.to_csv(path)
    print(f"  Saved: {path}")
    return path


def load_featured_df(filename: str) -> Optional[pd.DataFrame]:
    """Load featured DataFrame from CSV."""
    path = DATA_DIR / filename
    if not path.exists():
        print(f"  File not found: {path}")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"  Loaded: {path} ({len(df)} rows)")
    return df


# ── Build Feature Dataset (mirrors original) ────────────────────────────
def build_feature_dataset(
    df_raw: pd.DataFrame,
    timeframe: str = TACTICAL_TF,
) -> pd.DataFrame:
    """Build complete feature dataset: features + labels."""
    df = make_features_df(df_raw, timeframe)
    df = make_labels_df(df, timeframe)
    return df


# ── Full Pipeline ──────────────────────────────────────────────────────
def run_full_pipeline(
    symbol: str = SYMBOL,
    whole_days: int = HISTORY_DAYS,
    timeframe: str = TACTICAL_TF,
    train_frac: float = TRAIN_FRACTION,
    save_intermediate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline:
    1. Download historical prices
    2. Build feature dataset
    3. Train/validation split
    4. Save all artifacts

    Returns (df_train, df_val)
    """
    print(f"\n{'='*60}")
    print(f"DATA PIPELINE: {symbol} {timeframe}")
    print(f"{'='*60}")
    print(f"Days: {whole_days}, Train frac: {train_frac}")

    # Step 1: Download
    print(f"\n[1/4] Downloading historical prices...")
    df_raw = download_historical(symbol=symbol, days=whole_days, timeframe=timeframe)

    # Step 2: Features + Labels
    print(f"\n[2/4] Engineering features and labels...")
    df_labeled = build_feature_dataset(df_raw, timeframe)
    print(f"  Labeled data: {len(df_labeled)} rows")

    # Step 3: Split
    print(f"\n[3/4] Chronological train/validation split...")
    df_train, df_val = train_val_split(df_labeled, train_frac=train_frac)

    # Step 4: Save
    if save_intermediate:
        print(f"\n[4/4] Saving artifacts...")
        train_f = f"df_{symbol}_{timeframe}_train.csv"
        val_f = f"df_{symbol}_{timeframe}_val.csv"
        save_featured_df(df_train, train_f)
        save_featured_df(df_val, val_f)
    else:
        print(f"\n[4/4] Skipping save (save_intermediate=False)")

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Train: {len(df_train)} candles")
    print(f"  Val:   {len(df_val)} candles")
    print(f"{'='*60}\n")

    return df_train, df_val