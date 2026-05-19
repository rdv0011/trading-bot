import numpy as np
import pandas as pd

from mltrainingcore import make_features, detect_regime
from timeframe_config import TimeframeConfig


STRATEGIC_VOLATILITY_WINDOW_SHORT = 24
STRATEGIC_VOLATILITY_WINDOW_LONG = 168
EXTREME_VOL_RATIO = 2.5
HIGH_VOL_RATIO = 1.6
DRAWDOWN_WINDOW = 48


def make_strategic_features(df: pd.DataFrame, tf_cfg: TimeframeConfig) -> pd.DataFrame:
    df = make_features(df, tf_cfg)

    df["vol_short"] = df["ret1"].rolling(STRATEGIC_VOLATILITY_WINDOW_SHORT).std()
    df["vol_long"] = df["ret1"].rolling(STRATEGIC_VOLATILITY_WINDOW_LONG).std()
    df["vol_ratio_strategic"] = df["vol_short"] / df["vol_long"].clip(lower=1e-8)

    rolling_max = df["close"].rolling(DRAWDOWN_WINDOW).max()
    df["drawdown"] = (df["close"] - rolling_max) / rolling_max.clip(lower=1e-8)
    df["max_drawdown_window"] = df["drawdown"].rolling(DRAWDOWN_WINDOW).min()

    df["atr_pct"] = df["atr14"] / df["close"].clip(lower=1e-8)
    df["atr_pct_ma"] = df["atr_pct"].rolling(STRATEGIC_VOLATILITY_WINDOW_LONG).mean()
    df["atr_pct_ratio"] = df["atr_pct"] / df["atr_pct_ma"].clip(lower=1e-8)

    df["vol_state"] = df["vol_ratio_strategic"].apply(_classify_vol_state)

    return df.dropna().round(5)


def _classify_vol_state(vol_ratio: float) -> float:
    if vol_ratio >= EXTREME_VOL_RATIO:
        return 3.0
    if vol_ratio >= HIGH_VOL_RATIO:
        return 2.0
    if vol_ratio >= 1.0:
        return 1.0
    return 0.0


def get_strategic_features(df: pd.DataFrame) -> list:
    exclude = {"future_close", "future_ret", "regime", "vol_state"}
    return [
        c
        for c in df.columns
        if c not in exclude and np.issubdtype(df[c].dtype, np.number)
    ]
