import gc
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd

from mlio import get_latest_model_paths, load_model
from positionmanager import StrategicDecision
from strategic.strategicfeatures import make_strategic_features, get_strategic_features
from timeframe_config import TimeframeConfig

STRATEGIC_MODEL_TYPE_PREFIX = "strategic"
_RELOAD_CHECK_INTERVAL = 300

_REGIME_TO_LEVERAGE = {
    "trend": 5.0,
    "high_vol": 2.0,
    "chop": 1.0,
}

_VOL_STATE_ALLOW = {
    0.0: True,
    1.0: True,
    2.0: True,
    3.0: False,
}

_FALLBACK_DECISION = StrategicDecision(
    allow_trading=False,
    market_regime="chop",
    volatility_state="high",
    recommended_leverage=1.0,
    max_exposure_frac=0.0,
    stake_long_frac=0.1,
    stake_short_frac=0.05,
    stop_loss_frac=0.02,
    take_profit_frac=0.04,
    max_hold_hours=4.0,
    confidence=0.0,
)


class StrategicML:
    def __init__(
        self,
        model_dir,
        tf_cfg: TimeframeConfig,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.model_dir = model_dir
        self.tf_cfg = tf_cfg
        self.log = logger if logger is not None else print
        self._model = None
        self._metadata = None
        self._current_model_path = None
        self._current_meta_path = None
        self._last_reload_check: float = 0.0
        self._load_latest()

    def predict(self, df: pd.DataFrame) -> StrategicDecision:
        self._check_and_reload()

        if self._model is None or self._metadata is None:
            self.log("⚠️ StrategicML: no model loaded, using fallback decision")
            return _FALLBACK_DECISION

        try:
            df_feat = make_strategic_features(df, self.tf_cfg)
            feature_cols = self._metadata.get("feature_cols", get_strategic_features(df_feat))

            for col in feature_cols:
                if col not in df_feat.columns:
                    df_feat[col] = 0.0

            X_last = df_feat[feature_cols].fillna(0).iloc[[-1]]
            raw = self._model.predict(X_last)

            return self._decode_prediction(raw, df_feat.iloc[-1])

        except Exception as e:
            self.log(f"⚠️ StrategicML prediction failed: {e}")
            return _FALLBACK_DECISION

    def _decode_prediction(self, raw_preds, last_row) -> StrategicDecision:
        target_keys = self._metadata.get("valid_targets", [])
        removed = self._metadata.get("removed_targets", {})

        pred_arr = np.atleast_1d(np.asarray(raw_preds[0]) if hasattr(raw_preds, "__len__") else raw_preds)
        result = {k: removed[k] for k in removed}
        for i, key in enumerate(target_keys[: len(pred_arr)]):
            result[key] = float(pred_arr[i])

        regime = str(last_row.get("regime", "chop"))
        vol_state_raw = float(last_row.get("vol_state", 2.0))
        vol_label = {0.0: "low", 1.0: "normal", 2.0: "high", 3.0: "extreme"}.get(
            vol_state_raw, "high"
        )
        allow = _VOL_STATE_ALLOW.get(vol_state_raw, False)

        return StrategicDecision(
            allow_trading=allow,
            market_regime=regime,
            volatility_state=vol_label,
            recommended_leverage=float(result.get("recommended_leverage", _REGIME_TO_LEVERAGE.get(regime, 1.0))),
            max_exposure_frac=float(result.get("max_exposure_frac", 0.5)),
            stake_long_frac=float(result.get("stake_long_frac", 0.1)),
            stake_short_frac=float(result.get("stake_short_frac", 0.05)),
            stop_loss_frac=float(result.get("stop_loss_frac", 0.02)),
            take_profit_frac=float(result.get("take_profit_frac", 0.04)),
            max_hold_hours=float(result.get("max_hold_hours", 4.0)),
            confidence=1.0,
        )

    def _load_latest(self):
        try:
            model_path, meta_path = get_latest_model_paths(
                STRATEGIC_MODEL_TYPE_PREFIX, self.model_dir
            )
            model, metadata = load_model(model_path, meta_path)
            self._model = model
            self._metadata = metadata
            self._current_model_path = model_path
            self._current_meta_path = meta_path
            self.log(f"✅ StrategicML loaded: {model_path}")
        except FileNotFoundError:
            self.log("⚠️ StrategicML: no persisted model found — will use fallback until trained")
        except Exception as e:
            self.log(f"⚠️ StrategicML load error: {e}")

    def _check_and_reload(self):
        now = time.time()
        if now - self._last_reload_check < _RELOAD_CHECK_INTERVAL:
            return
        self._last_reload_check = now
        try:
            latest_path, latest_meta = get_latest_model_paths(
                STRATEGIC_MODEL_TYPE_PREFIX, self.model_dir
            )
            if latest_path != self._current_model_path:
                self.log(f"🔄 StrategicML hot-swap: {latest_path}")
                old = self._model
                self._model = None
                del old
                gc.collect()
                self._model, self._metadata = load_model(latest_path, latest_meta)
                self._current_model_path = latest_path
                self._current_meta_path = latest_meta
        except FileNotFoundError:
            self.log("⚠️ StrategicML: no model file found during reload check")
        except Exception as e:
            self.log(f"⚠️ StrategicML reload check failed: {e}")

    @property
    def is_ready(self) -> bool:
        return self._model is not None
