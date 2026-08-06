from basestrategy import BaseStrategy
import time as _time
import traceback
from collections import deque
from dataclasses import replace
from datetime import datetime
from binancebasebroker import SIGNAL_HOLD, SIGNAL_LONG, SIGNAL_SHORT, MARKET_TYPE_SPOT
from mlio import MODEL_DIR
from mltrainingcore import make_features, make_labels, get_features
from timeframe_config import TIMEFRAMES
from tactical.tacticalml import TacticalML, TacticalSignal
from strategic.strategicml import StrategicML
from positionmanager import PositionManager
from riskguard import RiskGuard

STRATEGIC_HISTORY_CANDLES = 300
TACTICAL_HISTORY_CANDLES_MULTIPLIER = 2


class MissingHistoricalDataError(Exception):
    pass


class DualMLStrategy(BaseStrategy):

    def initialize(self):
        self.asset = self.parameters.get("asset_symbol", "BTC")
        self.market_type = self.parameters.get("market_type", "futures")
        self._iteration_count = 0

        self.tf_cfg_tactical = TIMEFRAMES[self.parameters.get("tactical_timeframe", "15m")]
        self.tf_cfg_strategic = TIMEFRAMES[self.parameters.get("strategic_timeframe", "1h")]

        tactical_init_length = self.compute_required_history(self.tf_cfg_tactical)

        self.log_message(
            f"⏱ DualMLStrategy | tactical={self.tf_cfg_tactical.name} "
            f"strategic={self.tf_cfg_strategic.name}"
        )

        df_hist = self.get_historical_prices(
            self.asset, tactical_init_length, self.tf_cfg_tactical.name
        )
        if df_hist is None or len(df_hist) == 0:
            raise MissingHistoricalDataError(
                f"No historical data for {self.asset} @ {self.tf_cfg_tactical.name}"
            )

        df_hist = make_features(df_hist, self.tf_cfg_tactical)
        df_hist = make_labels(df_hist, self.tf_cfg_tactical)
        features = get_features(df_hist)

        model_params = self.parameters.get("model_params", {"iterations": 300, "verbose": False})

        self.tactical_ml = TacticalML(
            model_params=model_params,
            tf_cfg=self.tf_cfg_tactical,
            logger=self.log_message,
        )
        self.tactical_ml.warmup(df_hist, features)

        model_dir = self.parameters.get("model_dir", MODEL_DIR)
        self.strategic_ml = StrategicML(
            model_dir=model_dir,
            tf_cfg=self.tf_cfg_strategic,
            logger=self.log_message,
        )

        if not self.strategic_ml.is_ready:
            self.log_message(
                "⚠️ StrategicML has no trained model yet. "
                "Run: python strategic/strategictraining.py"
            )

        self.position_manager = PositionManager(
            broker=self._broker,
            asset=self.asset,
            quote_symbol=self.quote_asset_symbol,
            logger=self.log_message,
        )

        self.risk_guard = RiskGuard(
            max_daily_loss_frac=self.parameters.get("max_daily_loss_frac", 0.05),
            max_drawdown_frac=self.parameters.get("max_drawdown_frac", 0.15),
            max_leverage=self.parameters.get("max_leverage", 20.0),
        )

        self._last_tactical_candle_ts = None
        self._cached_tactical_signal = TacticalSignal(
            signal=SIGNAL_HOLD,
            prediction=0.0,
            min_threshold=0.0,
            max_threshold=0.0,
        )
        self._gate_counter_day = None
        self._daily_gate_counters = {
            "volume_filter": 0,
            "htf_trend": 0,
            "adaptive_threshold": 0,
            "riskguard": 0,
            "chop_regime": 0,
            "strategic_veto": 0,
        }
        self._regime_counts = {"trend": 0, "chop": 0, "high_vol": 0}
        self._prediction_history = deque(maxlen=1000)

        self.log_message("✅ DualMLStrategy initialized")

    def _ensure_daily_gate_counters(self, candle_ts):
        counter_day = candle_ts.date() if hasattr(candle_ts, "date") else None
        if counter_day != self._gate_counter_day:
            # Log previous day's regime distribution before resetting
            if self._gate_counter_day is not None:
                total = sum(self._regime_counts.values()) or 1
                pct_str = " | ".join(
                    f"{k}={v/total*100:.0f}%"
                    for k, v in sorted(self._regime_counts.items())
                )
                self.log_message(
                    f"📊 Regime distribution ({self._gate_counter_day}): {pct_str}"
                )
            self._gate_counter_day = counter_day
            self._daily_gate_counters = {
                "volume_filter": 0,
                "htf_trend": 0,
                "adaptive_threshold": 0,
                "riskguard": 0,
                "chop_regime": 0,
                "strategic_veto": 0,
            }
            self._regime_counts = {"trend": 0, "chop": 0, "high_vol": 0}

    def _increment_gate_counter(self, gate_name, candle_ts):
        self._ensure_daily_gate_counters(candle_ts)
        self._daily_gate_counters[gate_name] = self._daily_gate_counters.get(gate_name, 0) + 1

    def _log_daily_gate_summary(self, candle_ts):
        self._ensure_daily_gate_counters(candle_ts)
        counters = self._daily_gate_counters
        # Prediction distribution stats
        pred_stats = ""
        if len(self._prediction_history) >= 10:
            arr = list(self._prediction_history)
            mean = sum(arr) / len(arr)
            near_zero = sum(1 for v in arr if abs(v) < 0.001) / len(arr)
            pred_stats = (
                f" | pred(mean={mean:.5f} near_zero={near_zero:.0%} "
                f"n={len(arr)})"
            )
        self.log_message(
            "ℹ️ Gate counter summary | "
            f"day={self._gate_counter_day} "
            f"vol_flt={counters.get('volume_filter', 0)} "
            f"htf_trd={counters.get('htf_trend', 0)} "
            f"adapt_thr={counters.get('adaptive_threshold', 0)} "
            f"riskguard={counters.get('riskguard', 0)} "
            f"chop={counters.get('chop_regime', 0)} "
            f"veto={counters.get('strategic_veto', 0)}"
            f"{pred_stats}"
        )

    def on_trading_iteration(self):
        self._iteration_count += 1
        self._iteration_start = _time.time()
        self.log_message(f"🔄 Iteration #{self._iteration_count} starting")
        current_equity = self.get_cash()
        if not self.risk_guard.update(current_equity):
            self.log_message(f"GATE: riskguard halted — equity={current_equity:.2f}")
            self._increment_gate_counter("riskguard", datetime.utcnow())
            if self.position_manager.has_position:
                self.position_manager.emergency_close_live()
            return

        lev = self._broker.get_position_leverage(self._pair_asset_symbol(self.asset))
        lev_str = f" | leverage={lev}x" if lev is not None else ""
        if self.position_manager.has_position:
            pos = self.get_position(self.asset)
            if pos is not None and pos.amount is not None:
                direction = "LONG" if pos.amount > 0 else "SHORT"
                self.log_message(
                    f"Position | {direction} {abs(pos.amount)} {self.asset} @ {pos.entry_price:.2f}{lev_str}"
                )
            else:
                self.log_message(f"Position | FLAT{lev_str}")
        else:
            self.log_message(f"Position | FLAT{lev_str}")

        df_tactical = self.get_historical_prices(
            self.asset,
            self.tf_cfg_tactical.max_history_candles,
            self.tf_cfg_tactical.name,
        )
        if df_tactical is None or len(df_tactical) < self.tf_cfg_tactical.min_feature_candles:
            self.log_message("❌ Insufficient tactical data, skipping")
            return

        df_strategic = self.get_historical_prices(
            self.asset,
            STRATEGIC_HISTORY_CANDLES,
            self.tf_cfg_strategic.name,
        )
        if df_strategic is None or len(df_strategic) < 50:
            self.log_message("❌ Insufficient strategic data, skipping")
            return

        strategic_decision = self.strategic_ml.predict(df_strategic)
        strategic_decision = replace(
            strategic_decision,
            recommended_leverage=self.risk_guard.clamp_leverage(strategic_decision.recommended_leverage),
        )

        # Only re-run the tactical model when a new 15m candle has formed.
        # On heartbeat iterations (every 5m) between 15m candles, reuse the
        # previous signal — avoids wasted CatBoost retrains and signal flapping.
        current_tactical_ts = df_tactical.index[-1]
        tactical_is_new = current_tactical_ts != self._last_tactical_candle_ts

        if tactical_is_new:
            self._last_tactical_candle_ts = current_tactical_ts

            df_tactical_raw = df_tactical
            df_tactical = make_features(df_tactical, self.tf_cfg_tactical)
            df_tactical = make_labels(df_tactical, self.tf_cfg_tactical)
            features = get_features(df_tactical)

            df_pred = make_features(df_tactical_raw, self.tf_cfg_tactical).iloc[[-1]]

            tactical_signal = self.tactical_ml.fit_and_predict(df_tactical, df_pred, features)

            self.log_debug(
                f"tactical: signal={tactical_signal.signal} "
                f"prediction={tactical_signal.prediction:.6f} "
                f"min_thr={tactical_signal.min_threshold:.6f} "
                f"max_thr={tactical_signal.max_threshold:.6f}"
            )
            self.log_debug(
                f"strategic: regime={strategic_decision.market_regime} "
                f"direction={strategic_decision.direction} "
                f"confidence={strategic_decision.confidence}"
            )

            if self.market_type.lower() == MARKET_TYPE_SPOT and tactical_signal.signal == SIGNAL_SHORT:
                tactical_signal = type(tactical_signal)(
                    signal=SIGNAL_HOLD,
                    prediction=tactical_signal.prediction,
                    min_threshold=tactical_signal.min_threshold,
                    max_threshold=tactical_signal.max_threshold,
                )

            # Count adaptive-threshold HOLD (signal was inside thresholds, no trade generated)
            if tactical_signal.signal == SIGNAL_HOLD:
                self._increment_gate_counter("adaptive_threshold", current_tactical_ts)
            # Collect prediction values for distribution analysis
            self._prediction_history.append(tactical_signal.prediction)

            # Track regime distribution from current tactical candle
            if "regime" in df_tactical.columns:
                current_regime = df_tactical["regime"].iloc[-1]
                self._regime_counts[current_regime] = self._regime_counts.get(current_regime, 0) + 1

            if tactical_signal.signal != SIGNAL_HOLD:
                market_regime = strategic_decision.market_regime
                volume_threshold = 0.5 if market_regime == "chop" else 0.8
                vol_sma20 = df_tactical_raw["volume"].rolling(20).mean().iloc[-1]
                current_vol = df_tactical_raw["volume"].iloc[-1]
                if vol_sma20 > 0 and current_vol < vol_sma20 * volume_threshold:
                    blocked_signal = tactical_signal.signal.upper()
                    self.log_message(f"GATE: volume_filter blocked {blocked_signal}")
                    self._increment_gate_counter("volume_filter", current_tactical_ts)
                    self.log_message(
                        f"🔇 Volume filter: regime={market_regime} vol={current_vol:.0f} "
                        f"< {volume_threshold:.0%} SMA20={vol_sma20:.0f} "
                        f"— overriding {blocked_signal} to HOLD"
                    )
                    tactical_signal = type(tactical_signal)(
                        signal=SIGNAL_HOLD,
                        prediction=tactical_signal.prediction,
                        min_threshold=tactical_signal.min_threshold,
                        max_threshold=tactical_signal.max_threshold,
                    )

            if tactical_signal.signal != SIGNAL_HOLD:
                df_strategic["ema50"] = df_strategic["close"].ewm(span=50, adjust=False).mean()
                current_1h_close = df_strategic["close"].iloc[-1]
                current_ema50 = df_strategic["ema50"].iloc[-1]
                above_ema50 = current_1h_close > current_ema50

                if tactical_signal.signal == SIGNAL_LONG and not above_ema50:
                    blocked_signal = tactical_signal.signal.upper()
                    self.log_message(f"GATE: htf_trend blocked {blocked_signal}")
                    self._increment_gate_counter("htf_trend", current_tactical_ts)
                    self.log_message(
                        f"🔇 HTF filter: LONG but 1h close={current_1h_close:.0f} < EMA50={current_ema50:.0f}"
                        f" — overriding to HOLD"
                    )
                    tactical_signal = type(tactical_signal)(
                        signal=SIGNAL_HOLD,
                        prediction=tactical_signal.prediction,
                        min_threshold=tactical_signal.min_threshold,
                        max_threshold=tactical_signal.max_threshold,
                    )
                elif tactical_signal.signal == SIGNAL_SHORT and above_ema50:
                    blocked_signal = tactical_signal.signal.upper()
                    self.log_message(f"GATE: htf_trend blocked {blocked_signal}")
                    self._increment_gate_counter("htf_trend", current_tactical_ts)
                    self.log_message(
                        f"🔇 HTF filter: SHORT but 1h close={current_1h_close:.0f} > EMA50={current_ema50:.0f}"
                        f" — overriding to HOLD"
                    )
                    tactical_signal = type(tactical_signal)(
                        signal=SIGNAL_HOLD,
                        prediction=tactical_signal.prediction,
                        min_threshold=tactical_signal.min_threshold,
                        max_threshold=tactical_signal.max_threshold,
                    )

            self._cached_tactical_signal = tactical_signal
        else:
            self.log_message(
                f"⏭ Same {self.tf_cfg_tactical.name} candle {current_tactical_ts} "
                f"— reusing previous tactical signal"
            )
            tactical_signal = self._cached_tactical_signal

        self.log_message(
            f"Tactical | signal={tactical_signal.signal.upper()} "
            f"pred={tactical_signal.prediction:.6f} "
            f"min={tactical_signal.min_threshold:.6f} max={tactical_signal.max_threshold:.6f}"
        )
        self.log_message(
            f"Strategic | allow={strategic_decision.allow_trading} "
            f"regime={strategic_decision.market_regime} "
            f"vol={strategic_decision.volatility_state} "
            f"leverage={strategic_decision.recommended_leverage:.1f}x "
            f"exposure={strategic_decision.max_exposure_frac:.2f}"
        )
        self._log_daily_gate_summary(current_tactical_ts)

        # Track regime distribution on non-new-candle iterations too
        if not tactical_is_new:
            self._regime_counts[strategic_decision.market_regime] = (
                self._regime_counts.get(strategic_decision.market_regime, 0) + 1
            )

        current_price = self.get_last_price(self.asset)

        # --- GATE: StrategicML veto ---
        if not strategic_decision.allow_trading:
            self.log_message(f"GATE: strategic_veto blocked — allow_trading=false")
            self._increment_gate_counter("strategic_veto", current_tactical_ts)
            if self.position_manager.has_position:
                self.position_manager.emergency_close_live()
            elapsed = _time.time() - self._iteration_start
            self.log_message(f"✅ Iteration #{self._iteration_count} complete ({elapsed:.1f}s)")
            return

        # --- GATE: Chop regime (no new entries) ---
        if strategic_decision.market_regime == "chop":
            self.log_message(f"GATE: chop_regime blocked — regime=chop")
            self._increment_gate_counter("chop_regime", current_tactical_ts)
            elapsed = _time.time() - self._iteration_start
            self.log_message(f"✅ Iteration #{self._iteration_count} complete ({elapsed:.1f}s)")
            return

        self.position_manager.on_signal(tactical_signal, strategic_decision, current_price)
        elapsed = _time.time() - self._iteration_start
        self.log_message(f"✅ Iteration #{self._iteration_count} complete ({elapsed:.1f}s)")

    def on_abrupt_closing(self):
        try:
            self.log_message("⚠️ Abrupt closing — emergency position close")
            self.position_manager.emergency_close_live()
        except Exception as e:
            self.log_message(f"❌ Emergency close error: {e}")
            self.log_message(traceback.format_exc())
