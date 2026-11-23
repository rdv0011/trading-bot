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
    ABS_VOL_RATIO_CRITICAL, ABS_VOL_RATIO_ROLLING_WINDOW,
)

# ── Constants ───────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Timeframe config (inlined from original timeframe_config.py)
TIMEFRAME_CONFIG = {
    "15m": {
        "binance_interval": Client.KLINE_INTERVAL_15MINUTE,  # type: ignore[attr-defined]
        "minutes": 15,
        "candles_per_day": 96,
        "label_horizon_candles": 4,
        "adaptive_history_candles": 200,
        "ema_spans": [5, 10, 20, 50, 100],
    },
    "1h": {
        "binance_interval": Client.KLINE_INTERVAL_1HOUR,  # type: ignore[attr-defined]
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
    Download historical FUTURES prices from Binance (the bot trades BTCUSDT
    perpetuals; spot prices/volume diverge from futures, especially on
    testnet). Uses the public futures klines endpoint via a no-auth client.

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
        Client.KLINE_INTERVAL_1MINUTE: 60_000,  # type: ignore[attr-defined]
        Client.KLINE_INTERVAL_15MINUTE: 15 * 60_000,  # type: ignore[attr-defined]
        Client.KLINE_INTERVAL_1HOUR: 3600_000,  # type: ignore[attr-defined]
        Client.KLINE_INTERVAL_4HOUR: 4 * 3600_000,  # type: ignore[attr-defined]
        Client.KLINE_INTERVAL_1DAY: 86_400_000,  # type: ignore[attr-defined]
    }.get(interval, 60_000)

    # Paginate backwards from now in batches of up to 1000 candles
    # (Binance single-request limit). Each batch ends at `end_ms`, then we
    # step back to the candle before this batch's first open and repeat.
    batch_limit = 1000
    end_ms = int(datetime.now().timestamp() * 1000)
    all_rows = []
    fetched = 0

    while fetched < target_candles and end_ms > 0:
        # Use the FUTURES klines endpoint (not spot get_klines): the bot
        # trades BTCUSDT perpetual futures, and spot prices/volume diverge
        # from futures (especially on testnet where volume is synthetic).
        # Replays must see the same series the live broker sees.
        klines = client.futures_klines(
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

    # Keep OHLCV + quote_volume (absolute volume measure for Oct 5 post-mortem gating)
    df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].astype(float)

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

    # Regime detection (old logic based on vol_ratio only)
    df['regime'] = df.apply(_detect_regime, axis=1)

    # Absolute volume ratio (Oct 5 post-mortem: relative vol_ratio masks
    # volume vacuums where both short and long vol collapse together).
    # abs_vol_ratio = current bar quote_volume / rolling median quote_volume
    rolling_median_vol = df['quote_volume'].rolling(ABS_VOL_RATIO_ROLLING_WINDOW, min_periods=1).median()
    df['abs_vol_ratio'] = df['quote_volume'] / rolling_median_vol.clip(lower=1e-8)

    # Override regime to "halt" where absolute volume is critically low
    df['abs_vol_state'] = df['abs_vol_ratio'].apply(
        lambda x: classify_abs_vol_state(x, ABS_VOL_RATIO_CRITICAL)
    )
    df.loc[df['abs_vol_state'] == 'abs_vol_dropped', 'regime'] = 'halt' 

    # Drop NaN rows from feature engineering
    df = df.dropna().round(5)

    return df


def _detect_regime(row, abs_vol_state: str = "normal") -> str:
    """Detect regime: trend, chop, high_vol, or halt."""
    if abs_vol_state == "abs_vol_dropped":
        return "halt"
    atr = max(row["atr14"], 1e-8)
    trend_strength = abs(row["ema_20"] - row["ema_100"]) / atr
    vol_ratio = row["vol_12"] / max(row["vol_48"], 1e-8)

    if trend_strength < 0.4:
        return "chop"
    if vol_ratio > 1.4:
        return "high_vol"
    return "trend"


# ── Adaptive Threshold & Vol-State Helpers (Phase 2) ────────────────────
def adaptive_threshold(
    preds: pd.Series,
    lookback: int = 200,
    quantile: float = 0.95,
    min_threshold: float = 0.002,
    max_threshold: float = 0.02,
    fallback: float = 0.006,
) -> float:
    """
    Compute an adaptive signal threshold from recent prediction magnitudes.

    Takes the last `lookback` |predictions|, drops NaN, and uses the
    `quantile`-th percentile of their absolute values as the threshold,
    clamped into [min_threshold, max_threshold]. Mirrors the legacy
    adaptive_thresholding(): threshold adapts to recent signal strength
    instead of using a fixed ABSOLUTE_THRESHOLD.

    Falls back to `fallback` when fewer than 20 valid predictions are
    available (warmup period).

    Returns:
        float: threshold value rounded to 6 decimals.
    """
    if preds is None or len(preds) == 0:
        return round(float(fallback), 6)

    recent = pd.Series(preds).tail(lookback).dropna().abs()
    if len(recent) < 20:
        return round(float(fallback), 6)

    thr = float(np.percentile(recent.values, quantile * 100.0))
    thr = float(np.clip(thr, min_threshold, max_threshold))
    return round(thr, 6)


def adaptive_liquidity_threshold(
    intensities: pd.Series,
    lookback: int = 200,
    quantile: float = 0.25,          # use lower percentile (e.g., 25th) as floor
    min_threshold: float = 10000.0,  # absolute floor
    max_threshold: float = 100000.0, # absolute ceiling
    fallback: float = 25000.0,
) -> float:
    """
    Compute an adaptive LIQ_EXIT trade intensity threshold from recent values.

    Uses the `quantile`-th percentile (default 25th) of recent trade_intensity
    values as the minimum acceptable intensity. This ensures the threshold
    adapts to market conditions (low volume periods get lower threshold).

    Clamped into [min_threshold, max_threshold]. Falls back to `fallback`
    when fewer than 20 valid samples are available.

    Returns:
        float: threshold value in USD/s.
    """
    if intensities is None or len(intensities) == 0:
        return round(float(fallback), 1)

    recent = pd.Series(intensities).tail(lookback).dropna()
    if len(recent) < 20:
        return round(float(fallback), 1)

    # Use lower percentile as floor - we want to exit when intensity drops
    # BELOW what's typical for this market regime
    thr = float(np.percentile(recent.values, quantile * 100.0))
    thr = float(np.clip(thr, min_threshold, max_threshold))
    return round(thr, 1)


def adaptive_trailing_threshold(
    atr_values: pd.Series,
    price: float,
    lookback: int = 50,
    quantile: float = 0.5,          # median ATR as baseline
    mult: float = 0.1,              # fraction of ATR for update threshold
    min_threshold: float = 0.001,   # 0.1% floor
    max_threshold: float = 0.01,    # 1% ceiling
    fallback: float = 0.002,
) -> float:
    """
    Compute an adaptive TRAILING_SL_UPDATE_THRESHOLD from recent ATR values.

    Uses `mult` * `quantile`-th percentile of recent ATR (as fraction of price)
    as the minimum price move to trigger an SL/TP update. This makes the
    update threshold scale with volatility.

    Clamped into [min_threshold, max_threshold]. Falls back to `fallback`
    when fewer than 10 valid samples are available.

    Args:
        atr_values: Series of ATR values (absolute price units)
        price: Current price to convert ATR to fraction
        lookback: Rolling window size
        quantile: Percentile to use (default 0.5 = median)
        mult: Fraction of ATR to use as threshold (default 0.1 = 10%)
        min_threshold: Minimum threshold as price fraction (default 0.1%)
        max_threshold: Maximum threshold as price fraction (default 1%)
        fallback: Fallback value when insufficient data

    Returns:
        float: threshold as fraction of price (e.g., 0.002 = 0.2%).
    """
    if atr_values is None or len(atr_values) == 0:
        return round(float(fallback), 6)

    if price is None or price <= 0:
        return round(float(fallback), 6)

    recent = pd.Series(atr_values).tail(lookback).dropna()
    if len(recent) < 10:
        return round(float(fallback), 6)

    median_atr = float(np.percentile(recent.values, quantile * 100.0))
    # Convert ATR to fraction of price: (median_atr * mult) / price
    thr = (median_atr * mult) / price
    thr = float(np.clip(thr, min_threshold, max_threshold))
    return round(thr, 6)


def adaptive_vol_threshold(
    vol_ratios: pd.Series,
    lookback: int = 200,
    quantile: float = 0.95,          # 95th percentile as extreme threshold
    min_threshold: float = 1.2,      # absolute floor
    max_threshold: float = 3.0,      # absolute ceiling
    fallback: float = 1.8,
) -> float:
    """
    Compute an adaptive extreme volatility threshold from recent vol_ratio values.
    
    Uses the `quantile`-th percentile of recent vol_ratio values as the
    threshold for "extreme" volatility. This adapts to market conditions.
    
    Clamped into [min_threshold, max_threshold]. Falls back to `fallback`
    when fewer than 20 valid samples are available.
    
    Returns:
        float: vol_ratio threshold for extreme classification.
    """
    if vol_ratios is None or len(vol_ratios) == 0:
        return round(float(fallback), 3)
    
    recent = pd.Series(vol_ratios).tail(lookback).dropna()
    if len(recent) < 20:
        return round(float(fallback), 3)
    
    thr = float(np.percentile(recent.values, quantile * 100.0))
    thr = float(np.clip(thr, min_threshold, max_threshold))
    return round(thr, 3)


def adaptive_spread_threshold(
    spreads: pd.Series,
    lookback: int = 200,
    quantile: float = 0.75,          # 75th percentile as max spread
    min_threshold: float = 2.0,      # absolute floor (bps)
    max_threshold: float = 20.0,     # absolute ceiling (bps)
    fallback: float = 10.0,
) -> float:
    """
    Compute an adaptive max spread threshold from recent spread values.
    
    Uses the `quantile`-th percentile of recent spread_bps values as the
    maximum allowed spread for entries.
    
    Clamped into [min_threshold, max_threshold]. Falls back to `fallback`
    when fewer than 20 valid samples are available.
    
    Returns:
        float: max spread threshold in bps.
    """
    if spreads is None or len(spreads) == 0:
        return round(float(fallback), 2)
    
    recent = pd.Series(spreads).tail(lookback).dropna()
    if len(recent) < 20:
        return round(float(fallback), 2)
    
    thr = float(np.percentile(recent.values, quantile * 100.0))
    thr = float(np.clip(thr, min_threshold, max_threshold))
    return round(thr, 2)


def adaptive_depth_threshold(
    depths: pd.Series,
    lookback: int = 200,
    quantile: float = 0.25,          # 25th percentile as min depth
    min_threshold: float = 20000.0,  # absolute floor (USD)
    max_threshold: float = 500000.0, # absolute ceiling (USD)
    fallback: float = 50000.0,
) -> float:
    """
    Compute an adaptive min depth threshold from recent depth values.
    
    Uses the `quantile`-th percentile of recent depth_total_usd values as the
    minimum required depth for entries.
    
    Clamped into [min_threshold, max_threshold]. Falls back to `fallback`
    when fewer than 20 valid samples are available.
    
    Returns:
        float: min depth threshold in USD.
    """
    if depths is None or len(depths) == 0:
        return round(float(fallback), 1)
    
    recent = pd.Series(depths).tail(lookback).dropna()
    if len(recent) < 20:
        return round(float(fallback), 1)
    
    thr = float(np.percentile(recent.values, quantile * 100.0))
    thr = float(np.clip(thr, min_threshold, max_threshold))
    return round(thr, 1)


def classify_vol_state(vol_ratio: float, extreme_ratio: float = 1.8) -> str:
    """
    Classify the short/long volatility ratio into a vol state.

    Returns "extreme" when vol_ratio >= extreme_ratio (used by the regime
    gate to block entries during extreme volatility), else "normal".
    """
    if not np.isfinite(vol_ratio):
        return "extreme" if extreme_ratio <= 0 else "normal"
    return "extreme" if vol_ratio >= extreme_ratio else "normal"


def classify_abs_vol_state(abs_vol_ratio: float, critical_ratio: float = 0.10) -> str:
    """
    Classify absolute volume ratio into a state.

    Returns "abs_vol_dropped" when abs_vol_ratio < critical_ratio
    (actual trading activity collapsed below rolling median), else "normal".
    """
    if not np.isfinite(abs_vol_ratio):
        return "abs_vol_dropped"
    return "abs_vol_dropped" if abs_vol_ratio < critical_ratio else "normal"


def compute_atr_from_klines(
    df: pd.DataFrame,
    period: int = 14,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """Compute ATR14 from OHLCV klines DataFrame.
    
    Uses True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    Then ATR = rolling mean of TR over `period`.
    
    Args:
        df: DataFrame with OHLCV columns
        period: ATR period (default 14)
        price_col: Close price column name
        high_col: High price column name
        low_col: Low price column name
    
    Returns:
        Series of ATR values (aligned with df index, first `period` values are NaN)
    """
    if df is None or len(df) < period + 1:
        return pd.Series(dtype=float)
    
    high = df[high_col]
    low = df[low_col]
    close = df[price_col]
    prev_close = close.shift(1)
    
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(period, min_periods=period).mean()
    return atr


def compute_vol_ratio_from_klines(
    df: pd.DataFrame,
    short_window: int = 12,
    long_window: int = 48,
    price_col: str = "close",
) -> float:
    """Compute volatility ratio (short_window / long_window) from klines.
    
    Uses rolling standard deviation of log returns.
    
    Args:
        df: DataFrame with close prices
        short_window: Short volatility window (default 12)
        long_window: Long volatility window (default 48)
        price_col: Close price column name
    
    Returns:
        vol_ratio = vol_short / vol_long, or 1.0 if insufficient data
    """
    if df is None or len(df) < long_window + 1:
        return 1.0
    
    close = df[price_col]
    log_ret = np.log(close / close.shift(1))
    
    vol_short = log_ret.rolling(short_window).std().iloc[-1]
    vol_long = log_ret.rolling(long_window).std().iloc[-1]
    
    if not np.isfinite(vol_long) or vol_long == 0:
        return 1.0
    
    return float(vol_short / vol_long)


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