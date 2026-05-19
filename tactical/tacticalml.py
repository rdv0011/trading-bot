from dataclasses import dataclass, field
from collections import deque
from typing import Callable, Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from binancebasebroker import SIGNAL_HOLD, SIGNAL_LONG, SIGNAL_SHORT
from mltrainingcore import (
    TARGET_COLUMN,
    SEED_BASE,
    create_model,
    resolve_model_class,
    adaptive_thresholding,
)
from timeframe_config import TimeframeConfig


@dataclass
class TacticalSignal:
    signal: str
    prediction: float
    min_threshold: float
    max_threshold: float


class TacticalML:
    """
    Ephemeral 5m predictor. Retrained on every candle from a rolling window.
    Never persisted to disk — purely in-memory.
    Produces LONG / SHORT / HOLD signals via adaptive thresholding.
    """

    def __init__(
        self,
        model_type: str,
        model_params: dict,
        tf_cfg: TimeframeConfig,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.model_type = model_type
        self.model_cls = resolve_model_class(model_type)
        self.model_params = model_params
        self.tf_cfg = tf_cfg
        self.log = logger if logger is not None else print
        self._pred_history: deque = deque(maxlen=tf_cfg.max_history_candles)

    def warmup(self, df: pd.DataFrame, features: List[str]):
        n = len(df)
        window = max(50, self.tf_cfg.min_feature_candles)
        mdl = None
        retrain_every = 12

        for i in tqdm(range(window, n), desc="TacticalML warmup", unit="candle"):
            if i % retrain_every == 0 or mdl is None:
                train_df = df.iloc[i - window : i]
                X_train = train_df[features]
                y_train = train_df[TARGET_COLUMN]
                seed = SEED_BASE + i - window
                mdl = create_model(self.model_cls, seed, self.model_params)
                mdl.fit(X_train, y_train)

            X_pred = df.iloc[[i]][features]
            p = float(mdl.predict(X_pred)[0])
            self._pred_history.append(p)

        self.log(
            f"🧠 TacticalML warmup complete: {len(self._pred_history)} predictions cached"
        )

    def fit_and_predict(self, df: pd.DataFrame, features: List[str]) -> TacticalSignal:
        X = df[features]
        y = df[TARGET_COLUMN]

        seed = SEED_BASE + len(self._pred_history)
        mdl = create_model(self.model_cls, seed, self.model_params)
        mdl.fit(X, y)

        last_row = X.iloc[[-1]]
        prediction = float(mdl.predict(last_row)[0])

        hist_len = self.tf_cfg.adaptive_history_candles
        pred_series = pd.Series(list(self._pred_history) + [prediction])

        if len(pred_series) < hist_len:
            max_th, min_th = np.nan, np.nan
        else:
            max_th, min_th = adaptive_thresholding(
                pred_series.iloc[-hist_len:], self.tf_cfg
            )

        if np.isnan(max_th):
            signal = SIGNAL_HOLD
        elif prediction > max_th:
            signal = SIGNAL_LONG
        elif prediction < min_th:
            signal = SIGNAL_SHORT
        else:
            signal = SIGNAL_HOLD

        self._pred_history.append(prediction)

        return TacticalSignal(
            signal=signal,
            prediction=prediction,
            min_threshold=min_th,
            max_threshold=max_th,
        )

    @property
    def pred_history_len(self) -> int:
        return len(self._pred_history)
