"""
Dual-ML Strategy logic for Bitcoin Trading Bot.
Combines tactical (15m) signals with strategic (1h) meta-parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import time

from config import (
    ABSOLUTE_THRESHOLD,
    STAKE_LONG_FRAC_DEFAULT, STAKE_SHORT_FRAC_DEFAULT,
    STOP_LOSS_FRAC_DEFAULT, TAKE_PROFIT_FRAC_DEFAULT,
    MAX_HOLD_HOURS_DEFAULT, LEVERAGE_DEFAULT,
)
from logging import log_info, log_debug, log_warning, log_error, log_trade_entry, log_trade_exit, log_equity
from model import CatBoostModel, rolling_tactical_predict, strategic_batch_predict, predict_strategic_meta_params


# ── Signal Conversion ───────────────────────────────────────────────────
def prediction_to_signal(pred: float, threshold: float = ABSOLUTE_THRESHOLD) -> str:
    """Convert tactical prediction to trading signal."""
    if np.isnan(pred):
        return "hold"
    if pred > threshold:
        return "long"
    if pred < -threshold:
        return "short"
    return "hold"


# ── Default Meta-Parameters ─────────────────────────────────────────────
DEFAULT_META = {
    "stake_long_frac": STAKE_LONG_FRAC_DEFAULT,
    "stake_short_frac": STAKE_SHORT_FRAC_DEFAULT,
    "stop_loss_frac": STOP_LOSS_FRAC_DEFAULT,
    "take_profit_frac": TAKE_PROFIT_FRAC_DEFAULT,
    "max_hold_hours": MAX_HOLD_HOURS_DEFAULT,
    "recommended_leverage": LEVERAGE_DEFAULT,
    "regime": "trend",
}


# ── DualMLStrategy Class ────────────────────────────────────────────────
class DualMLStrategy:
    """
    Dual-ML Strategy combining:
    - Tactical ML (15m): Entry/exit signals via walk-forward prediction
    - Strategic ML (1h): Meta-parameters (stake, SL, TP, max_hold, leverage, regime)
    """

    def __init__(
        self,
        broker: Any,  # BinanceBroker or MockBroker
        config: Any = None,
        tactical_model: Optional[CatBoostModel] = None,
        strategic_model: Optional[CatBoostModel] = None,
        feature_cols: Optional[List[str]] = None,
        tactical_tf_cfg: Any = None,
        strategic_tf_cfg: Any = None,
    ):
        self.broker = broker
        self.config = config

        # Models
        self.tactical_model = tactical_model
        self.strategic_model = strategic_model

        # Feature columns
        self.feature_cols = feature_cols or []

        # Timeframe configs (for walk-forward window sizes)
        self.tactical_tf_cfg = tactical_tf_cfg
        self.strategic_tf_cfg = strategic_tf_cfg

        # State
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_time: Optional[datetime] = None
        self.current_meta = DEFAULT_META.copy()
        self.current_trade: Optional[Dict] = None

        # Threshold
        self.threshold = getattr(config, 'ABSOLUTE_THRESHOLD', ABSOLUTE_THRESHOLD) if config else ABSOLUTE_THRESHOLD

        # Walk-forward retraining
        self.retrain_every = getattr(config, 'WALKFORWARD_RETRAIN_EVERY', 100) if config else 100
        self.candles_since_retrain = 0

    # ── Simulation Mode ────────────────────────────────────────────────
    def run_simulation(
        self,
        df_val: pd.DataFrame,
        tactical_preds: pd.Series,
        strategic_meta_params: List[Dict],
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Run simulation on validation data using pre-computed predictions.
        This is the primary simulation path (called from simulate.py).
        """
        log_info("Running DualMLStrategy simulation...")

        # Reset state
        self._reset_state()

        # Delegate to broker's simulation (which handles position management)
        # This method exists for interface compatibility
        # The actual simulation logic is in simulate.py's run_simulation

        trades_df = self.broker.get_trades_df() if hasattr(self.broker, 'get_trades_df') else pd.DataFrame()
        metrics = self.broker.get_metrics() if hasattr(self.broker, 'get_metrics') else {}

        return trades_df, metrics

    def _reset_state(self) -> None:
        """Reset strategy state."""
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.current_meta = DEFAULT_META.copy()
        self.current_trade = None
        self.candles_since_retrain = 0

    # ── Live Trading Mode ──────────────────────────────────────────────
    def run_live_loop(
        self,
        sleep_seconds: int = 60,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Run live trading loop.
        Fetches data, makes predictions, executes trades.
        """
        log_info("=" * 60)
        log_info("Starting LIVE trading loop")
        log_info(f"  Sleep: {sleep_seconds}s")
        log_info("=" * 60)

        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            try:
                iteration += 1
                self._live_iteration()
            except KeyboardInterrupt:
                log_info("Interrupted by user")
                break
            except Exception as e:
                log_error(f"Live iteration error: {e}")
                log_error(f"Retrying in {sleep_seconds}s...")

            time.sleep(sleep_seconds)

    def _live_iteration(self) -> None:
        """Single live trading iteration."""
        # 1. Fetch latest data for both timeframes
        # This would call broker.get_historical_prices() for tactical and strategic
        # For now, this is a placeholder - requires BinanceBroker implementation

        # 2. Build features for latest candle
        # df_tactical = self._build_tactical_features()
        # df_strategic = self._build_strategic_features()

        # 3. Get tactical prediction (walk-forward)
        # pred = self.tactical_model.predict(df_tactical, self.feature_cols).iloc[-1]

        # 4. Get strategic meta-params (batch, less frequent)
        # if iteration % (60 * 60 // sleep_seconds) == 0:  # Every hour
        #     meta_list = predict_strategic_meta_params(
        #         df_strategic, self.strategic_model, self.feature_cols
        #     )
        #     self.current_meta = meta_list[-1]  # Latest

        # 5. Convert to signal
        # signal = prediction_to_signal(pred, self.threshold)

        # 6. Check exits
        # self._check_exits()

        # 7. Execute entry/exit
        # self._execute_signal(signal)

        # 8. Log equity
        # log_equity(datetime.now(), self.broker.get_equity(), self.position, ...)

        log_debug(f"Live iteration {iteration} - placeholder (needs BinanceBroker)")

    # ── Position Management ─────────────────────────────────────────────
    def _check_exits(self, current_price: float, current_time: datetime) -> Optional[str]:
        """Check if position should be exited. Returns exit reason or None."""
        if self.position == 0:
            return None

        # Time-based exit
        if self.entry_time is not None:
            hours_held = (current_time - self.entry_time).total_seconds() / 3600
            if hours_held >= self.current_meta["max_hold_hours"]:
                return "max_hold"

        # Stop loss / Take profit
        if self.position > 0:  # Long
            sl_price = self.entry_price * (1 - self.current_meta["stop_loss_frac"])
            if current_price <= sl_price:
                return "sl"
            tp_price = self.entry_price * (1 + self.current_meta["take_profit_frac"])
            if current_price >= tp_price:
                return "tp"
        else:  # Short
            sl_price = self.entry_price * (1 + self.current_meta["stop_loss_frac"])
            if current_price >= sl_price:
                return "sl"
            tp_price = self.entry_price * (1 - self.current_meta["take_profit_frac"])
            if current_price <= tp_price:
                return "tp"

        return None

    def _execute_signal(self, signal: str, current_price: float, current_time: datetime) -> None:
        """Execute trading signal."""
        # Exit if reversal
        if signal == "long" and self.position < 0:
            self._exit_position("reversal", current_price, current_time)
        elif signal == "short" and self.position > 0:
            self._exit_position("reversal", current_price, current_time)

        # Enter if flat
        if self.position == 0 and signal in ("long", "short"):
            self._enter_position(signal, current_price, current_time)

    def _enter_position(self, side: str, price: float, timestamp: datetime) -> None:
        """Open new position."""
        # Apply slippage (broker handles this in reality)
        # For strategy, we just log intent
        stake_frac = (
            self.current_meta["stake_long_frac"]
            if side == "long"
            else self.current_meta["stake_short_frac"]
        )
        leverage = self.current_meta["recommended_leverage"]

        # Delegate to broker
        if hasattr(self.broker, 'open_position'):
            self.broker.open_position(
                side=side,
                stake_frac=stake_frac,
                leverage=leverage,
                stop_loss_frac=self.current_meta["stop_loss_frac"],
                take_profit_frac=self.current_meta["take_profit_frac"],
            )

        # Update local state
        self.position = 1.0 if side == "long" else -1.0  # Direction only
        self.entry_price = price
        self.entry_time = timestamp

        # Create trade record
        self.current_trade = log_trade_entry(
            timestamp=timestamp,
            symbol="BTCUSDT",
            side=side,
            entry_price=price,
            qty=0.0,  # Filled by broker
            stake_frac=stake_frac,
            leverage=leverage,
            stop_loss=self.current_meta["stop_loss_frac"],
            take_profit=self.current_meta["take_profit_frac"],
            max_hold_hours=self.current_meta["max_hold_hours"],
            regime=self.current_meta["regime"],
            tactical_pred=0.0,  # Updated by caller
            strategic_params=self.current_meta.copy(),
            equity_before=0.0,  # Filled by broker
        )

        log_info(f"LIVE ENTRY {side.upper()} @ {price:.2f} | Meta: {json.dumps(self.current_meta, default=str)}")

    def _exit_position(self, reason: str, price: float, timestamp: datetime) -> None:
        """Close current position."""
        if self.position == 0:
            return

        side = "long" if self.position > 0 else "short"

        # Delegate to broker
        if hasattr(self.broker, 'close_position'):
            self.broker.close_position()

        # Log exit
        if self.current_trade:
            log_trade_exit(
                trade=self.current_trade,
                exit_price=price,
                exit_reason=reason,
                pnl=0.0,  # Filled by broker
                pnl_pct=0.0,
                equity_after=0.0,
                fee_paid=0.0,
                slippage_paid=0.0,
            )

        log_info(f"LIVE EXIT {reason.upper()} @ {price:.2f}")

        # Reset
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.current_trade = None


# ── Utility Functions ───────────────────────────────────────────────────
def get_default_meta() -> Dict[str, Any]:
    """Return default meta-parameters."""
    return DEFAULT_META.copy()


def validate_meta_params(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize meta-parameters."""
    validated = DEFAULT_META.copy()
    for key, default_val in DEFAULT_META.items():
        if key in meta:
            val = meta[key]
            # Type conversion and bounds checking
            if isinstance(default_val, float):
                validated[key] = float(val)
            elif isinstance(default_val, int):
                validated[key] = int(val)
            else:
                validated[key] = val

    # Bounds
    validated["stake_long_frac"] = np.clip(validated["stake_long_frac"], 0.01, 0.5)
    validated["stake_short_frac"] = np.clip(validated["stake_short_frac"], 0.01, 0.5)
    validated["stop_loss_frac"] = np.clip(validated["stop_loss_frac"], 0.005, 0.1)
    validated["take_profit_frac"] = np.clip(validated["take_profit_frac"], 0.01, 0.2)
    validated["max_hold_hours"] = np.clip(validated["max_hold_hours"], 0.5, 24.0)
    validated["recommended_leverage"] = np.clip(validated["recommended_leverage"], 1.0, 10.0)

    return validated