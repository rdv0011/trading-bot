import pickle
from pathlib import Path
import pandas as pd
import os
import sys
import os, json, hashlib
import numpy as np
import time
import threading
from typing import Any, Dict, List
from datetime import datetime
import joblib, glob

# Detect the directory of the main script (e.g., xgcatboost.py)
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
MODEL_DIR = Path(script_dir) / "models"
MODEL_DIR.mkdir(exist_ok=True)

LABEL_DIR = Path(script_dir) / "labeleddata"
LABEL_DIR.mkdir(exist_ok=True)

PREDICTIONS_DIR = Path(script_dir) / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)


# =============================================
# Model Persistence Functions
# =============================================
def _cleanup_old_files(pattern, model_dir, keep_count=2):
    """
    Remove old files, keeping only the most recent ones.
    
    Args:
        pattern: Glob pattern to match files
        model_dir: Directory containing files
        keep_count: Number of most recent files to keep
    """
    files = list(model_dir.glob(pattern))
    if len(files) <= keep_count:
        return
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Remove older files
    for old_file in files[keep_count:]:
        old_file.unlink()
        print(f"🗑️  Removed old file: {old_file.name}")

# =============================================
# Model Persistence Functions
# =============================================
def save_model(model, model_type, model_dir=MODEL_DIR, keep_count=2, metadata=None):
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

def load_model(model_type, model_dir=MODEL_DIR):
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

    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    # Load metadata
    metadata = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            # Non-fatal – model still loads
            metadata = None

    print(f"[load_model] Loaded model: {model_path}")

    return model, metadata, model_path

def save_feature_columns(features, model_type, model_dir=MODEL_DIR, keep_count=2):
    """
    Save feature column names for consistent prediction.
    Keeps only the most recent feature files to save disk space.
    
    Args:
        features: List of feature column names
        model_type: 'xgb' or 'cat'
        model_dir: Directory to save feature list
        keep_count: Number of recent feature files to keep (default: 2)
    
    Returns:
        Path to saved feature file
    """
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_type}_features_{timestamp}.pkl"
    filepath = model_dir / filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(features, f)
    
    print(f"✅ Features saved: {filepath}")
    
    # Cleanup old feature files
    _cleanup_old_files(f"{model_type}_features_*.pkl", model_dir, keep_count)
    
    return filepath

def load_feature_columns(filepath=None, model_type=None, model_dir=MODEL_DIR):
    """
    Load feature column names.
    
    Args:
        filepath: Path to saved feature file (optional)
        model_type: 'xgb' or 'cat' - required if filepath is None
        model_dir: Directory containing feature files
    
    Returns:
        List of feature column names
    """
    if filepath is None:
        if model_type is None:
            raise ValueError("Either filepath or model_type must be provided")
        
        # Find the most recent feature file
        pattern = f"{model_type}_features_*.pkl"
        files = list(model_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No feature files found matching pattern: {pattern}")
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        filepath = files[0]
        print(f"📂 Loading latest features: {filepath.name}")
    
    with open(filepath, 'rb') as f:
        features = pickle.load(f)
    
    return features

def get_latest_model_paths(model_type, model_dir=MODEL_DIR):
    """
    Get paths to the latest model and features files.
    
    Args:
        model_type: 'xgb' or 'cat'
        model_dir: Directory containing model files
    
    Returns:
        tuple: (model_path, features_path)
    """
    # Find latest model
    model_files = list(model_dir.glob(f"{model_type}_model_*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model files found for type: {model_type}")
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    model_path = model_files[0]
    
    # Find latest features
    feature_files = list(model_dir.glob(f"{model_type}_features_*.pkl"))
    if not feature_files:
        raise FileNotFoundError(f"No feature files found for type: {model_type}")
    feature_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    features_path = feature_files[0]
    
    return str(model_path), str(features_path)

def hash_model_params(params: Dict[str, Any]) -> str:
    """Stable hash for model parameters."""
    s = json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def hash_df_deterministic(df: pd.DataFrame, feature_cols: list[str]) -> str:
    """
    Deterministic hash of a DataFrame using only specified feature columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_cols (list[str]): Columns to include in the hash.

    Returns:
        str: Deterministic hash string.
    """
    if df is None or df.empty or not feature_cols:
        return "none"

    # Select only feature columns
    df_features = df[feature_cols].copy()

    # Ensure deterministic column order
    df_features = df_features[sorted(df_features.columns)]

    # Normalize dtypes
    for c in df_features.columns:
        if pd.api.types.is_numeric_dtype(df_features[c]):
            df_features[c] = df_features[c].astype(np.float64)
        elif pd.api.types.is_datetime64_any_dtype(df_features[c]):
            df_features[c] = df_features[c].astype("int64")  # nanoseconds since epoch
        else:
            df_features[c] = df_features[c].astype("string")

    # Convert to deterministic bytes
    hash_vals = pd.util.hash_pandas_object(df_features, index=False).values
    return hashlib.md5(hash_vals.tobytes()).hexdigest()


def make_cache_key(timestamp: Any, df_window: pd.DataFrame, feature_cols: List[str]) -> str:
    """
    Generate a deterministic cache key using the timestamp and feature columns of a DataFrame.

    Args:
        timestamp (Any): Prediction timestamp.
        df_window (pd.DataFrame): Rolling window of data.
        feature_cols (list[str]): Columns to include in the hash.

    Returns:
        str: Deterministic cache key.
    """
    # Normalize timestamp
    if timestamp is None:
        ts_str = "none"
    elif isinstance(timestamp, pd.Timestamp):
        ts_str = timestamp.isoformat()  # deterministic ISO format
    else:
        ts_str = str(timestamp)

    # Compute deterministic hash of features
    data_hash = hash_df_deterministic(df_window, feature_cols)[:8]  # short hash
    return f"{ts_str}_{data_hash}"

class JSONModelCache:
    def __init__(self, model_type: str, model_params: Dict[str, Any], autosave_interval: int = 120):
        """
        In-memory JSON cache for model predictions, one file per model configuration.
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model_hash = hash_model_params(model_params)
        self.path = PREDICTIONS_DIR / f"cache_{model_type}_{self.model_hash}.json"
        self.autosave_interval = autosave_interval
        self.cache = {}
        self.lock = threading.Lock()
        self.meta = {
            "model_type": model_type,
            "params": model_params
        }

        self._load()
        if autosave_interval > 0:
            self._start_autosave_thread()

    # Core

    def _load(self):
        """Load cache file if exists."""
        if not os.path.exists(self.path):
            print(f"🆕 New cache for {self.model_type} ({self.model_hash})")
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.cache = data.get("data", {})
            self.meta = data.get("meta", self.meta)
            print(f"✅ Loaded {len(self.cache)} cached entries from {self.path}")
        except Exception as e:
            print(f"⚠️ Failed to load cache {self.path}: {e}")

    def _save(self):
        """Save to JSON atomically."""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with self.lock:
            payload = {"meta": self.meta, "data": self.cache}
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)

    def _start_autosave_thread(self):
        """Background thread to autosave cache."""
        def autosaver():
            while True:
                time.sleep(self.autosave_interval)
                try:
                    self._save()
                except Exception as e:
                    print(f"⚠️ Autosave failed: {e}")
        t = threading.Thread(target=autosaver, daemon=True)
        t.start()

    # API

    def get(self, key: str) -> Any:
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        with self.lock:
            self.cache[key] = value

    def flush(self):
        """Force immediate save."""
        self._save()

    def __contains__(self, key):
        return key in self.cache

    def __len__(self):
        return len(self.cache)

    def __repr__(self):
        return f"<JSONModelCache {self.model_type}:{self.model_hash} ({len(self.cache)} entries)>"