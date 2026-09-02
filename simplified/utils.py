"""
Shared utilities for Dual-ML Bitcoin Trading Bot.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


# ── Metrics Calculation ─────────────────────────────────────────────────
def calculate_metrics(trades_df: pd.DataFrame, equity_curve: pd.Series = None) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics from trades DataFrame.
    """
    if trades_df.empty:
        return {
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "num_trades": 0,
            "avg_trade_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_hold_hours": 0.0,
        }

    # Basic stats
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    total_pnl = trades_df['pnl'].sum()
    win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0.0

    avg_trade = trades_df['pnl'].mean()
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0.0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0.0

    gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 1.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Drawdown from equity curve
    if equity_curve is not None and len(equity_curve) > 0:
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = abs(drawdown.min())
    else:
        max_dd = 0.0

    # Sharpe ratio (annualized, assuming hourly returns)
    if equity_curve is not None and len(equity_curve) > 1:
        returns = equity_curve.pct_change().dropna()
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(8760)  # hourly to annual
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Average hold time
    if 'timestamp' in trades_df.columns and 'exit_timestamp' in trades_df.columns:
        try:
            trades_df['hold_hours'] = (
                pd.to_datetime(trades_df['exit_timestamp']) -
                pd.to_datetime(trades_df['timestamp'])
            ).dt.total_seconds() / 3600
            avg_hold = trades_df['hold_hours'].mean()
        except:
            avg_hold = 0.0
    else:
        avg_hold = 0.0

    return {
        "total_return": round(total_pnl, 6),
        "total_return_pct": round(total_pnl, 6),  # Assuming normalized equity
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_pct": round(max_dd, 6),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "num_trades": len(trades_df),
        "avg_trade_pnl": round(avg_trade, 6),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "avg_hold_hours": round(avg_hold, 2),
    }


# ── CSV Operations ──────────────────────────────────────────────────────
def save_trades_csv(trades: pd.DataFrame, path: str) -> None:
    """Save trades DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(path, index=False)
    print(f"Saved {len(trades)} trades to {path}")


def load_trades_csv(path: str) -> pd.DataFrame:
    """Load trades from CSV."""
    if not Path(path).exists():
        print(f"File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Parse timestamps
    for col in ['timestamp', 'exit_timestamp']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


def merge_trade_csvs(pattern: str = "logs/trades_*.csv") -> pd.DataFrame:
    """Merge multiple trade CSV files."""
    csv_files = sorted(Path(".").glob(pattern))

    if not csv_files:
        print(f"No CSV files found matching {pattern}")
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"Merged {len(csv_files)} files: {len(merged)} total trades")
    return merged


# ── Feature Helpers ─────────────────────────────────────────────────────
def get_feature_cols(df: pd.DataFrame, exclude: list = None) -> List[str]:
    """
    Get numeric feature columns, excluding target/regime columns.
    """
    if exclude is None:
        exclude = ['future_close', 'future_ret', 'regime', 'target', 'label']

    return [c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])]


def normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Normalize features to [0, 1] range using min-max scaling.
    """
    df_norm = df.copy()
    for col in feature_cols:
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            range_val = max_val - min_val
            if range_val > 0:
                df_norm[col] = (df_norm[col] - min_val) / range_val
            else:
                df_norm[col] = 0.5
    return df_norm


# ── Model Utilities ─────────────────────────────────────────────────────
def model_summary(model) -> Dict[str, Any]:
    """Get summary of CatBoost model."""
    if model is None:
        return {"error": "model is None"}

    try:
        return {
            "iterations": model.get_param('iterations'),
            "depth": model.get_param('depth'),
            "learning_rate": model.get_param('learning_rate'),
            "loss_function": model.get_param('loss_function'),
            "feature_importance": dict(zip(
                model.feature_names_,
                model.feature_importances_.tolist()
            )) if hasattr(model, 'feature_importances_') else None,
        }
    except Exception as e:
        return {"error": str(e)}


def save_model_metadata(
    path: str,
    metadata: Dict[str, Any],
) -> None:
    """Save model metadata to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=convert)

    print(f"Model metadata saved: {path}")


def load_model_metadata(path: str) -> Dict[str, Any]:
    """Load model metadata from JSON."""
    if not Path(path).exists():
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ── Date/Time Utilities ─────────────────────────────────────────────────
def format_timestamp(ts: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime to string."""
    return ts.strftime(fmt)


def parse_timestamp(s: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """Parse string to datetime."""
    return datetime.strptime(s, fmt)


# ── Signal Utilities ────────────────────────────────────────────────────
def signal_to_int(signal: str) -> int:
    """Convert signal string to int: long=1, hold=0, short=-1."""
    return {"long": 1, "hold": 0, "short": -1}.get(signal.lower(), 0)


def int_to_signal(i: int) -> str:
    """Convert int to signal string."""
    return {1: "long", 0: "hold", -1: "short"}.get(i, "hold")


# ── File Utilities ──────────────────────────────────────────────────────
def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    """Convert string to safe filename."""
    return re.sub(r'[^\w\-.]', '_', name)


# ── Comparison Helpers ──────────────────────────────────────────────────
def calculate_slippage(
    expected_price: float,
    actual_price: float,
    side: str,
) -> float:
    """Calculate slippage in basis points."""
    if side == "long":
        return (actual_price - expected_price) / expected_price * 10000
    else:
        return (expected_price - actual_price) / expected_price * 10000


def fee_estimate(
    price: float,
    qty: float,
    fee_rate: float = 0.0004,
) -> float:
    """Estimate trade fee."""
    return price * qty * fee_rate


# ── Data Validation ─────────────────────────────────────────────────────
def validate_dataframe(
    df: pd.DataFrame,
    required_cols: List[str],
    name: str = "DataFrame",
) -> bool:
    """Validate DataFrame has required columns and rows."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"{name} missing columns: {missing}")
        return False
    if len(df) == 0:
        print(f"{name} is empty")
        return False
    return True


def validate_prices(df: pd.DataFrame, name: str = "DataFrame") -> bool:
    """Validate price data (OHLCV)."""
    required = ['open', 'high', 'low', 'close', 'volume']
    return validate_dataframe(df, required, name)