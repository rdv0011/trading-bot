from basestrategy import BaseStrategy
import time
from mlio import MODEL_DIR
from timeframe_config import TIMEFRAMES
from mlpredictor import MlPredictor
from mltrainingcore import TARGET_COLUMN
from mltrainingcore import get_features, make_features, make_labels

MIN_TRADEABLE_QUANTITY = 0.001  # Minimum tradeable quantity for BTC on Binance
TRADEABLE_QUANTITY_PRECISION = 3  # Binance allows BTC quantities to 3 decimal places
BUY_SLIPPAGE = 1.005  # 0.5% slippage on buy orders
SELL_SLIPPAGE = 0.995  # 0.5% slippage on sell orders

class MissingHistoricalDataError(Exception):
    """Raised when get_historical_prices() returns None"""
    pass

class XGCatBoostStrategy(BaseStrategy):
    '''
    Machine Learning Trading Strategy using XGBoost/CatBoost
    '''
    
    def initialize(self):
        # Trading parameters
        self.asset = self.parameters.get("asset_symbol", "BTC")
        
        self.historical_prices_unit = self.parameters.get("historical_prices_unit", "5m")
        self.tf_cfg = TIMEFRAMES[self.historical_prices_unit]

        # Historical candles for fetching data (can be larger than adaptive history)
        self.historical_prices_length = self.parameters.get(
            "historical_prices_length",
            min(500, self.tf_cfg.max_history_candles)  # fallback capped by max history
        )

        # Adaptive thresholds & prediction windows
        self.max_history_size = self.parameters.get(
            "max_history_size",
            self.tf_cfg.adaptive_history_candles
        )
        self.predict_with_signal_num_candles = self.parameters.get(
            "predict_with_signal_num_candles",
            self.tf_cfg.adaptive_history_candles  # number of past predictions to track
        )
        self.predict_with_signal_label_window = self.parameters.get(
            "predict_with_signal_label_window",
            self.tf_cfg.label_window_candles
        )

        self.log_message(f"⏱ TimeframeConfig loaded: {self.tf_cfg.name} ({self.tf_cfg.minutes} min candles)")
        self.log_message(f"  Historical candles: {self.historical_prices_length} @ {self.historical_prices_unit}")
        self.log_message(
            f"  Predictor windows | AdaptiveHistory={self.predict_with_signal_num_candles}, "
            f"LabelWindow={self.predict_with_signal_label_window}, "
            f"MaxHistory={self.max_history_size}"
        )
        self.log_message(f"  Min feature candles: {self.tf_cfg.min_feature_candles}")
        self.log_message(f"  Label horizon: {self.tf_cfg.label_horizon_candles} candles ({self.tf_cfg.label_horizon_minutes} minutes)")
        self.log_message(f"  EMA spans (candles): {list(self.tf_cfg.ema_spans)}")
        
        # Position tracking
        self.entry_price = None
        self.entry_time = None

        # Model reload tracking
        self.last_model_check = time.time()
        self.model_check_interval = self.parameters.get("model_check_interval", 300)  # Check every 5 minutes
        
        # Sleep time between iterations
        self.sleeptime = self.parameters.get("sleeptime", 300)  # 5 minutes to match training data

        # Prediction history for adaptive thresholding
        init_length = self.compute_required_history(self.tf_cfg)
        self.log_message(
            f"📊 Fetching {init_length} historical candles "
            f"(features={self.tf_cfg.min_feature_candles}, "
            f"adaptive={self.tf_cfg.adaptive_history_candles}, "
            f"label_window={self.tf_cfg.label_window_candles})"
        )
        df_hist = self.get_historical_prices(self.asset, init_length, self.historical_prices_unit)
        if df_hist is None or len(df_hist) == 0:
            msg = (
                f"❌ Initialization aborted: get_historical_prices() returned None or empty "
                f"for asset={self.asset}, candles={init_length}, interval={self.historical_prices_unit}"
            )
            self.log_message(msg)
            raise MissingHistoricalDataError(msg)
        
        df_hist = make_features(df_hist, tf_cfg=self.tf_cfg)
        df_hist = make_labels(df_hist, tf_cfg=self.tf_cfg)
        features = get_features(df_hist)

        self.predictor = MlPredictor(
            model_dir=self.parameters.get("model_dir", MODEL_DIR),
            model_type=self.parameters.get("model_type", "cat"),
            model_params=self.parameters.get("model_params", {'iterations': 500, 'verbose': False}),
            df_hist=df_hist,
            features=features,
            tf_cfg=self.tf_cfg,
            target_col=TARGET_COLUMN,
            logger=self.log_message,
        )

        # Log initial model info
        model_info = self.predictor.get_model_info()
        self.log_message(f"🤖 Strategy initialized")
        for key, value in model_info.items():
            pretty_key = key.replace("_", " ").title()
            self.log_message(f"   {pretty_key}: {value}")

    def on_trading_iteration(self):
        """
        Live trading iteration with:
        - Regime-aware stake scaling
        - Adaptive threshold signals
        - Time-based exits (max_hold_hours)
        - Regime-aware stop/take scaling
        - Trade marker logging
        """

        # ---------------------------------------------------------
        # 1️⃣ Fetch historical candles
        # ---------------------------------------------------------
        df = self.get_historical_prices(
            self.asset,
            self.historical_prices_length,
            self.historical_prices_unit,
        )

        if df is None or len(df) == 0:
            self.log_message("❌ No historical data, skipping iteration")
            return

        # ---------------------------------------------------------
        # 2️⃣ Update meta parameters (stake, SL, TP, max hold)
        # ---------------------------------------------------------
        try:
            meta = self.predictor.predict_meta_params(df)

            self.stake_long_frac = meta["stake_long_frac"]
            self.stake_short_frac = meta["stake_short_frac"]
            self.stop_loss_frac = meta["stop_loss_frac"]
            self.take_profit_frac = meta["take_profit_frac"]
            self.max_hold_hours = meta["max_hold_hours"]

            self.log_message(
                f"MetaParams | "
                f"Long={self.stake_long_frac:.3f}, "
                f"Short={self.stake_short_frac:.3f}, "
                f"SL={self.stop_loss_frac:.3f}, "
                f"TP={self.take_profit_frac:.3f}, "
                f"Hold={self.max_hold_hours:.1f}h"
            )
        except Exception as e:
            self.log_message(f"⚠️ Meta-param update failed: {e}")

        # ---------------------------------------------------------
        # 3️⃣ Feature engineering
        # ---------------------------------------------------------
        min_required = self.tf_cfg.min_feature_candles
        if len(df) < min_required:
            self.log_message(
                f"⚠️ Need {min_required} candles, got {len(df)}"
            )
            return

        df = make_features(df, tf_cfg=self.tf_cfg)
        df = make_labels(df, tf_cfg=self.tf_cfg)
        features = get_features(df)

        # ---------------------------------------------------------
        # 4️⃣ Predict signal (regime-aware)
        # ---------------------------------------------------------
        result = self.predictor.predict_with_signal(df, features, tf_cfg=self.tf_cfg)

        prediction = result["prediction"]
        signal = result["signal"]
        regime = result["regime"]
        stake_mult = result["stake_mult"]

        self.last_regime = regime
        self.last_stake_mult = stake_mult

        self.log_message(
            f"Signal | Pred={prediction:.6f} | MinThr={result['min_threshold']:.6f} | MaxThr={result['max_threshold']:.6f} | "
            f"Signal={signal.upper()} | "
            f"Regime={regime} | "
            f"StakeMult={stake_mult:.2f}"
        )

        # ---------------------------------------------------------
        # 5️⃣ Skip trading entirely in chop regime
        # ---------------------------------------------------------
        if stake_mult == 0.0:
            self.log_message("⏸ Chop regime detected — trading paused")
            return

        # ---------------------------------------------------------
        # 6️⃣ Apply regime scaling (mirrors simulator)
        # ---------------------------------------------------------
        stake_long = self.stake_long_frac * stake_mult
        stake_short = self.stake_short_frac * stake_mult

        scaled_sl, scaled_tp = self._scaled_risk_params(stake_mult)

        # ---------------------------------------------------------
        # 7️⃣ Current position state
        # ---------------------------------------------------------
        position = self.get_position(self.asset)
        current_price = self.get_last_price(self.asset)
        has_position = position is not None and abs(position) >= MIN_TRADEABLE_QUANTITY

        # ---------------------------------------------------------
        # 8️⃣ ENTRY LOGIC
        # ---------------------------------------------------------
        if not has_position and signal != "hold":
            # Clean up stale state
            if self.entry_price is not None:
                self.cancel_open_orders(self.asset)
                self.entry_price = None
                self.entry_time = None

            # Store scaled params for entry
            self.stake_long_frac = stake_long
            self.stake_short_frac = stake_short
            self.stop_loss_frac = scaled_sl
            self.take_profit_frac = scaled_tp

            self._enter_position(current_price, signal)
            return

        # ---------------------------------------------------------
        # 9️⃣ EXIT LOGIC (time-based, reversal-based)
        # ---------------------------------------------------------
        if has_position and self.entry_price is not None:
            self._check_exit_conditions(position, current_price, signal)

    def _enter_position(self, current_price: float, signal: str):
        signal = signal.lower()

        stake_frac = self.stake_long_frac if signal == "long" else self.stake_short_frac
        cash = self.get_cash()
        qty = round((cash * stake_frac) / current_price, TRADEABLE_QUANTITY_PRECISION)

        if qty < MIN_TRADEABLE_QUANTITY:
            self.log_message("⚠️ Quantity below minimum, skipping entry")
            return

        sl, tp = self._scaled_risk_params(self.last_stake_mult)

        res = self.open_position_with_bracket(
            self.asset,
            signal,
            qty,
            sl_frac=sl,
            tp_frac=tp,
        )

        if not res.success:
            self.log_message(f"❌ ENTRY FAILED: {res.error}")
            return

        self.entry_price = res.data["entry_price"]
        self.entry_time = self.get_datetime()

        self.log_message(f"🟢 ENTRY {signal.upper()} @ {self.entry_price}")
        
    def _check_exit_conditions(self, position, current_price: float, signal: str):
        is_long = position > 0

        # -------------------------------------------------
        # Performance calculation (direction-aware)
        # -------------------------------------------------
        perf = (
            (current_price / self.entry_price - 1.0)
            if is_long
            else (self.entry_price / current_price - 1.0)
        )

        # -------------------------------------------------
        # Time-based exit (SIMULATOR-FAITHFUL)
        # -------------------------------------------------
        now = self.get_datetime()

        elapsed_minutes = (
            now - self.entry_time
        ).total_seconds() / 60.0

        max_hold_minutes = self.max_hold_hours * 60.0

        exit_reason = None

        # 1️⃣ Time-based exit
        if elapsed_minutes >= max_hold_minutes:
            exit_reason = "MAX_HOLD_TIME"

        # -------------------------------------------------
        # Signal reversal exit
        # -------------------------------------------------
        elif is_long and signal == "short":
            exit_reason = "REVERSAL_LONG_TO_SHORT"
        elif not is_long and signal == "long":
            exit_reason = "REVERSAL_SHORT_TO_LONG"

        # -------------------------------------------------
        # Execute exit
        # -------------------------------------------------
        if exit_reason:
            self._close_orders_position(
                position=position,
                current_price=current_price,
                reason=exit_reason,
                perf=perf,
            )
        
    def _close_orders_position(self, position, current_price, reason, perf):
        
        self.log_message(f"🔵 Closing orders and position due to: {reason}")
        
        self.cancel_open_orders(self.asset)

        if position is not None and abs(position) >= MIN_TRADEABLE_QUANTITY:
            self.close_position(self.asset, position)

        self.log_message(
            f"🔵 EXIT {reason} @ {current_price} | PnL {perf:.2%}"
        )

        self.entry_price = None
        self.entry_time = None

    def on_abrupt_closing(self):
        """
        Close all positions on strategy shutdown or error.
        """
        try:
            self.log_message("⚠️  Abrupt closing triggered - closing all positions...")
            
            # Cancel all pending orders
            self.cancel_open_orders(self.asset)
            
            # Get current position
            position = self.get_position(self.asset)

            if position is not None and abs(position) >= MIN_TRADEABLE_QUANTITY:
                current_price = self.get_last_price(self.asset)
                is_long = position > 0
                position_type = "LONG" if is_long else "SHORT"
                self.log_message(f"🔴 EMERGENCY EXIT ({position_type}) at {current_price}")
                self.close_position(self.asset, position)

                # Calculate final performance if entry price is known
                if self.entry_price is not None:
                    if is_long:
                        perf = (current_price / self.entry_price - 1)
                    else:
                        perf = (self.entry_price / current_price - 1)
                    self.log_message(f"   Final PnL: {perf:.2%}")
            else:
                self.log_message("✅ Position was already closed or too small (possibly by stop/take profit)")
                
        except Exception as e:
            self.log_message(f"❌ Error during abrupt closing: {e}")
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}")

    def _scaled_risk_params(self, stake_mult: float):
        """
        Regime-aware stop/take scaling.
        High volatility => tighter exits.
        """
        if stake_mult <= 0:
            return None, None

        sl = self.stop_loss_frac / stake_mult
        tp = self.take_profit_frac / stake_mult

        # Safety clamp
        sl = min(sl, 0.10)
        tp = min(tp, 0.20)

        return sl, tp
