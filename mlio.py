from pathlib import Path
import pandas as pd
import os
import sys
import os, json
from datetime import datetime
import joblib, glob

# =============================================
# IO directories configuration
# =============================================
# Detect the directory of the main script
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
MODEL_DIR = Path(script_dir) / "models"
MODEL_DIR.mkdir(exist_ok=True)

LABEL_DIR = Path(script_dir) / "labeleddata"
LABEL_DIR.mkdir(exist_ok=True)

# =============================================
# Model Persistence Functions
# =============================================
def save_model(model, metadata, model_type, model_dir=MODEL_DIR, keep_count=2):
    """
    Save trained model to disk with timestamp.
    Keeps only the most recent models to save disk space.

    Args:
        model: Trained model object (e.g., MultiOutputRegressor wrapping CatBoost)
        model_type: 'xgb' or 'cat' (used in filename)
        model_dir: Directory to save models
        keep_count: Number of recent models to keep (default: 2)
        metadata: optional dict to save alongside the model (e.g., feature_cols, valid_targets)
    """
    os.makedirs(model_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_type = str(model_type)
    model_fname = os.path.join(model_dir, f"{safe_type}_meta_model_{ts}.pkl")
    meta_fname = os.path.join(model_dir, f"{safe_type}_meta_model_{ts}.meta.json")

    # Use joblib for generic pickling (works for sklearn wrappers & catboost objects)
    joblib.dump(model, model_fname)

    if metadata is not None:
        try:
            with open(meta_fname, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception:
            # Best-effort metadata saving; don't fail the pipeline for this.
            pass

    # Prune older models (keep only 'keep_count' most recent)
    pattern = os.path.join(model_dir, f"{safe_type}_meta_model_*.pkl")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    # Keep keep_count newest
    for old in files[keep_count:]:
        try:
            os.remove(old)
            # Also try to remove associated metadata file if exists
            old_meta = os.path.splitext(old)[0] + ".meta.json"
            if os.path.exists(old_meta):
                os.remove(old_meta)
        except Exception:
            pass

    print(f"[save_model] Saved model to {model_fname} (kept {min(keep_count, len(files))} newest).")
    return model_fname

def load_model(model_path, model_meta_path):
    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    # Load metadata
    metadata = None
    if os.path.exists(model_meta_path):
        try:
            with open(model_meta_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            # Non-fatal – model still loads
            metadata = None

    print(f"[load_model] Loaded model: {model_path}, meta info: {model_meta_path}")

    return model, metadata

def get_latest_model_paths(model_type, model_dir=MODEL_DIR):
    """
    Load the most recent saved model of the given type ('xgb' or 'cat').

    Returns:
        model: the loaded model object
        metadata: dict or None
        model_path: path to the loaded .pkl file
    """
    pattern = os.path.join(model_dir, f"{model_type}_meta_model_*.pkl")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    if not files:
        raise FileNotFoundError(f"No saved models for type '{model_type}' in {model_dir}")

    # Most recent model file
    model_path = files[0]
    meta_path = os.path.splitext(model_path)[0] + ".meta.json"
    
    return model_path, meta_path

def load_featured_df(filename):
    """
    Loads featured dataframe from LABEL_DIR/filename.
    Returns df or None.
    """
    path = LABEL_DIR / filename

    if path.exists():
        print(f"[DEBUG] Loading data from: {path}")
        df = pd.read_csv(path, parse_dates=['date'], index_col='date')
        print(f"✅ Loaded {len(df)} candles from CSV")
        return df

    return None

def save_featured_df(df, filename):
    """
    Saves dataframe into LABEL_DIR/filename.
    """
    path = LABEL_DIR / filename
    df.to_csv(path, index=True)
    print(f"✅ Saved {len(df)} candles to {path}")

def download_historical_prices(
    symbol: str,
    interval: str,
    lookback_days: int,
    client
):
    """
    symbol: 'BTCUSDT'
    interval: Client.KLINE_INTERVAL_15MINUTE
    lookback_days: int
    """

    start_str = f"{lookback_days} days ago UTC"

    print(f"📥 Fetching OHLCV from Binance ({symbol}, {interval})...")

    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_str,
    )

    if not klines:
        raise RuntimeError("No OHLCV data returned from Binance")

    df = pd.DataFrame(
        klines,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = (
        df[["date", "open", "high", "low", "close", "volume"]]
        .set_index("date")
        .astype(float)
        .sort_index()
    )

    print(f"✅ Retrieved {len(df)} candles")
    return df

def save_labels(df, filename):
    path = LABEL_DIR / filename
    df.to_csv(path, index=True)
    return path

def load_labels(filename):
    path = LABEL_DIR / filename
    return pd.read_csv(path, parse_dates=['date'], index_col='date')