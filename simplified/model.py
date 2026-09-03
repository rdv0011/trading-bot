"""
Model pipeline for Dual-ML Bitcoin Trading Bot.
Supports: CatBoost model training, persistence, and prediction.
KEEPS dual-ML architecture: Tactical (15m) + Strategic (1h).
Self-contained - no dependencies on original modules.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import logging

from catboost import CatBoostRegressor

from config import (
    MODEL_DIR, TACTICAL_MODEL_PARAMS, STRATEGIC_MODEL_PARAMS,
    WALKFORWARD_RETRAIN_EVERY, TACTICAL_TF, STRATEGIC_TARGET_COLS,
)

# ── Constants ───────────────────────────────────────────────────────────
TARGET_COLUMN = 'future_ret'
SEED_BASE = 42


# ── CatBoost Model Wrapper ─────────────────────────────────────────────
class CatBoostModel:
    """
    Wrapper for CatBoostRegressor with train, save, load, predict.
    Supports both tactical and strategic models.
    """

    def __init__(
        self,
        model_type: str = "tactical",
        model_params: dict = None,
        model_dir: str = MODEL_DIR,
    ):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model_dir = Path(model_dir)
        self.model = None
        self.metadata = None
        self.model_path = None
        self.meta_path = None

        # Select params based on model type
        if self.model_type == "tactical":
            self.default_params = TACTICAL_MODEL_PARAMS.copy()
        elif self.model_type == "strategic":
            self.default_params = STRATEGIC_MODEL_PARAMS.copy()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Merge defaults with provided params
        self.params = {**self.default_params, **self.model_params}

    def _get_path(self, prefix: str = "model") -> Tuple[Path, Path]:
        """Generate model file paths."""
        self.model_path = self.model_dir / f"{prefix}_{self.model_type}.cbm"
        self.meta_path = self.model_dir / f"{prefix}_{self.model_type}_meta.json"
        return self.model_path, self.meta_path

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = TARGET_COLUMN,
        target_cols: List[str] = None,
        save: bool = True,
    ) -> "CatBoostModel":
        """
        Train CatBoost model on provided data.

        If target_cols is given, trains a multi-output regressor on those
        target columns (used by the strategic model for trade parameters).
        """
        print(f"\n{'='*60}")
        print(f"TRAINING {self.model_type.upper()} MODEL")
        print(f"{'='*60}")
        print(f"  Rows: {len(df)}, Features: {len(feature_cols)}")
        print(f"  Params: {self.params}")

        # Prepare data
        X = df[feature_cols].fillna(0)
        if target_cols:
            y = df[target_cols].fillna(0)
            multi = True
        else:
            y = df[target_col].fillna(0)
            multi = False

        # Internal train/validation split (80/20) for early stopping
        n = len(X)
        n_train = int(np.floor(n * 0.8))
        X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_val = y.iloc[:n_train], y.iloc[n_train:]

        # Create and train model
        base = CatBoostRegressor(**self.params)

        print(f"\n  Training with {n_train} train, {n - n_train} val samples...")
        if multi:
            from sklearn.multioutput import MultiOutputRegressor
            self.model = MultiOutputRegressor(base)
            self.model.fit(X_train, y_train)
            best_score_val = None
        else:
            self.model = base
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=100,
            )
            best_score_val = None
            if hasattr(self.model, 'best_score_'):
                bs = self.model.best_score_
                if isinstance(bs, dict):
                    for dataset_key in ('test', 'validation', 'learn'):
                        if dataset_key in bs:
                            dataset_val = bs[dataset_key]
                            best_score_val = (
                                dataset_val.get('best') or dataset_val.get('min')
                                if isinstance(dataset_val, dict)
                                else dataset_val
                            )
                            break
                else:
                    best_score_val = bs
                if best_score_val is not None:
                    best_score_val = float(best_score_val)

        self.metadata = {
            "feature_cols": feature_cols,
            "n_features": len(feature_cols),
            "n_train_rows": n_train,
            "n_val_rows": n - n_train,
            "model_type": self.model_type,
            "params": self.params,
            "best_score": best_score_val,
            "multi_output": multi,
        }
        if multi:
            self.metadata["target_cols"] = target_cols

        # Save if requested
        if save:
            self.save()

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"  Best val score: {best_score_val}")
        print(f"{'='*60}")

        return self

    def save(self, prefix: str = "model") -> Tuple[Path, Path]:
        """Save current model and metadata to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")

        model_path, meta_path = self._get_path(prefix)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if self.metadata.get("multi_output"):
            import joblib
            model_path = self.model_dir / f"{prefix}_{self.model_type}.joblib"
            joblib.dump(self.model, str(model_path))
            self.meta_path = meta_path
        else:
            self.model.save_model(str(model_path))

        self.metadata["saved_at"] = pd.Timestamp.now().isoformat()
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"  Model saved: {model_path}")
        print(f"  Meta saved:  {meta_path}")
        return model_path, meta_path

    def load(
        self,
        model_path: str = None,
        meta_path: str = None,
    ) -> "CatBoostModel":
        """Load model from disk."""
        if model_path is None:
            # Auto-detect latest model
            model_path, meta_path = self._find_latest_model()

        print(f"\nLoading {self.model_type.upper()} model...")
        print(f"  Model: {model_path}")

        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path) if meta_path else None

        # Load metadata first to know how to deserialize
        loaded_meta = {}
        if self.meta_path and self.meta_path.exists():
            with open(self.meta_path, 'r') as f:
                loaded_meta = json.load(f)
            self.metadata = loaded_meta

        if loaded_meta.get("multi_output"):
            import joblib
            self.model = joblib.load(str(self.model_path))
        else:
            self.model = CatBoostRegressor(**self.params)
            self.model.load_model(str(self.model_path))

        if not self.metadata:
            self.metadata = {}

        print(f"  Model loaded successfully")
        return self

    def _find_latest_model(self) -> Tuple[Path, Path]:
        """Find latest model file in model directory."""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        pattern = f"model_{self.model_type}.cbm"
        model_path = self.model_dir / pattern
        joblib_pattern = self.model_dir / f"model_{self.model_type}.joblib"

        if joblib_pattern.exists():
            model_path = joblib_pattern
        elif not model_path.exists():
            raise FileNotFoundError(
                f"No model found at {model_path}. Run 'train' mode first."
            )

        meta_path = self.model_dir / f"model_{self.model_type}_meta.json"
        return model_path, meta_path

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.Series:
        """Make predictions on new data."""
        if self.model is None:
            raise RuntimeError(f"{self.model_type.upper()} model not loaded")

        X = df[feature_cols].fillna(0)
        preds = self.model.predict(X)

        if self.metadata and self.metadata.get("multi_output"):
            cols = self.metadata.get("target_cols", [f"y{i}" for i in range(np.asarray(preds).shape[1])])
            return pd.DataFrame(preds, index=df.index, columns=cols)

        return pd.Series(
            preds, index=df.index, name=f"{self.model_type}_prediction"
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.model is None:
            return {}
        if not hasattr(self.model, 'feature_importances_'):
            return {}

        return dict(zip(
            self.metadata.get("feature_cols", []),
            self.model.feature_importances_.tolist(),
        ))


# ── Walk-Forward Tactical Prediction ───────────────────────────────────
def rolling_tactical_predict(
    df: pd.DataFrame,
    model: CatBoostModel,
    feature_cols: List[str],
    retrain_every: int = WALKFORWARD_RETRAIN_EVERY,
    window: int = 500,
) -> pd.Series:
    """
    Walk-forward tactical predictions.
    Retrains tactical model every 'retrain_every' candles on rolling window.
    """
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD TACTICAL PREDICTION")
    print(f"{'='*60}")
    print(f"  Retrain every: {retrain_every} candles")
    print(f"  Window: {window} candles")

    n = len(df)
    preds = np.full(n, np.nan)
    current_model = None

    for i in range(window, n):
        # Retrain every retrain_every candles
        if (i - window) % retrain_every == 0 or i == window:
            train_df = df.iloc[i - window:i].copy()

            # Create fresh model for this window
            current_model = CatBoostRegressor(**model.params)
            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df[TARGET_COLUMN].fillna(0)
            current_model.fit(X_train, y_train, verbose=0)

        # Predict current candle
        if current_model is not None:
            X_pred = df.iloc[[i]][feature_cols].fillna(0)
            preds[i] = float(current_model.predict(X_pred)[0])

        # Progress
        if i % 100 == 0:
            print(f"  Progress: {i}/{n} candles ({i*100//n}%)")

    result = pd.Series(preds, index=df.index, name="tactical_prediction")

    print(f"\n  Completed: {np.sum(~np.isnan(preds))} predictions out of {n}")
    print(f"{'='*60}\n")

    return result


# ── Strategic Prediction (Batch) ───────────────────────────────────────
def strategic_batch_predict(
    df: pd.DataFrame,
    model: CatBoostModel,
    feature_cols: List[str],
) -> pd.Series:
    """Batch prediction using strategic model on entire dataset."""
    return model.predict(df, feature_cols)


# ── Meta-Parameter Prediction from Strategic Model ──────────────────────
def predict_strategic_meta_params(
    df: pd.DataFrame,
    model: CatBoostModel,
    feature_cols: List[str],
) -> List[Dict[str, Any]]:
    """
    Predict strategic meta-parameters (stake, SL, TP, max_hold, leverage).
    Returns list of param dicts, one per row in df.

    The strategic model is a multi-output regressor over STRATEGIC_TARGET_COLS.
    Each predicted value is clamped to a sane range; if a target is absent
    (older single-output model), it falls back to the safe default.
    """
    if model.model is None:
        raise RuntimeError("Strategic model not loaded")

    X = df[feature_cols].fillna(0)
    raw_preds = model.model.predict(X)

    target_cols = (
        list(model.metadata.get("target_cols", []))
        if model.metadata
        else list(STRATEGIC_TARGET_COLS)
    )

    bounds = {
        "recommended_leverage": (1.0, 10.0, 1.0),
        "max_exposure_frac": (0.0, 1.0, 0.5),
        "stake_long_frac": (0.01, 0.3, 0.1),
        "stake_short_frac": (0.01, 0.2, 0.05),
        "stop_loss_frac": (0.005, 0.1, 0.02),
        "take_profit_frac": (0.005, 0.2, 0.04),
        "max_hold_hours": (0.5, 48.0, 4.0),
    }

    # Column order in the prediction matrix follows target_cols if the model
    # is multi-output; default to ordered bounds otherwise.
    pred_arrs = np.asarray(raw_preds)
    is_multi_col = pred_arrs.ndim > 1 and pred_arrs.shape[1] > 1
    param_list = []

    for i in range(len(df)):
        row = {}
        if target_cols and is_multi_col:
            for j, key in enumerate(target_cols):
                default = bounds.get(key, (0.0, 1.0, 0.0))[2]
                if j < pred_arrs.shape[1]:
                    row[key] = _clamp_float(pred_arrs, (i, j), *bounds.get(key, (0.0, 1.0, default)))
                else:
                    row[key] = default
        else:
            for key in STRATEGIC_TARGET_COLS:
                row[key] = bounds[key][2]

        param_list.append({
            "stake_long_frac": row.get("stake_long_frac", 0.1),
            "stake_short_frac": row.get("stake_short_frac", 0.05),
            "stop_loss_frac": row.get("stop_loss_frac", 0.02),
            "take_profit_frac": row.get("take_profit_frac", 0.04),
            "max_hold_hours": row.get("max_hold_hours", 4.0),
            "max_exposure_frac": row.get("max_exposure_frac", 0.5),
            "recommended_leverage": row.get("recommended_leverage", 1.0),
            "regime": "trend",
        })

    return param_list


def _clamp_float(pred_arr, idx, lo, hi, default):
    value = default
    try:
        raw = float(pred_arr[idx])
        if np.isfinite(raw):
            value = raw
    except (IndexError, ValueError, TypeError):
        pass
    return float(max(lo, min(hi, value)))


# ── Convenience Functions ──────────────────────────────────────────────
def train_tactical_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_params: dict = None,
) -> CatBoostModel:
    """Train tactical (15m) model."""
    model = CatBoostModel(model_type="tactical", model_params=model_params)
    model.train(df, feature_cols, target_col=TARGET_COLUMN, save=True)
    return model


def train_strategic_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_params: dict = None,
) -> CatBoostModel:
    """Train strategic (1h) multi-output trade-parameter model."""
    model = CatBoostModel(model_type="strategic", model_params=model_params)
    model.train(df, feature_cols, target_cols=list(STRATEGIC_TARGET_COLS), save=True)
    return model


def load_tactical_model(model_dir: str = MODEL_DIR) -> CatBoostModel:
    """Load latest tactical model from disk."""
    model = CatBoostModel(model_type="tactical", model_dir=model_dir)
    model.load()
    return model


def load_strategic_model(model_dir: str = MODEL_DIR) -> CatBoostModel:
    """Load latest strategic model from disk."""
    model = CatBoostModel(model_type="strategic", model_dir=model_dir)
    model.load()
    return model