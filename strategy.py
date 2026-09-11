"""
Dual-ML Strategy logic for Bitcoin Trading Bot.
Combines tactical (15m) signals with strategic (1h) meta-parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

import config as _config
from config import (
    ABSOLUTE_THRESHOLD,
    STAKE_LONG_FRAC_DEFAULT, STAKE_SHORT_FRAC_DEFAULT,
    STOP_LOSS_FRAC_DEFAULT, TAKE_PROFIT_FRAC_DEFAULT,
    MAX_HOLD_HOURS_DEFAULT, LEVERAGE_DEFAULT,
    SYMBOL, TACTICAL_TF, STRATEGIC_TF,
    FEATURE_CACHE_TTL,
    ADAPTIVE_THRESHOLD_ENABLED, ADAPTIVE_LOOKBACK, ADAPTIVE_QUANTILE,
    ADAPTIVE_MIN_THRESHOLD, ADAPTIVE_MAX_THRESHOLD,
    TRAILING_STOP_ENABLED, TRAILING_ATR_MULT, TRAILING_BREAKEVEN_MULT,
    GATE_EXTREME_VOL, EXTREME_VOL_RATIO,
    SCALING_ENABLED, MAX_SCALE_COUNT, SCALE_CONFIRM_BARS, SCALE_STAKE_FRAC,
    PARTIAL_EXIT_ENABLED, PARTIAL_EXIT_FRACTION, REVERSAL_FULL_CLOSE_STREAK,
    TRADE_COOLDOWN_ENABLED, TRADE_COOLDOWN_MINUTES,
)
from logger import log_info, log_debug, log_warning, log_error, log_trade_entry, log_trade_exit, log_equity
from model import CatBoostModel, rolling_tactical_predict, strategic_batch_predict, predict_strategic_meta_params
from data import make_features_df, add_strategic_features_df, make_labels_df, adaptive_threshold, classify_vol_state


# ── Flag Reader (module-level config w/ optional injected cfg) ──────────
def _cfg_flag(name: str, default: Any, cfg: Any = None) -> Any:
    """Read a runtime flag from an injected config object (e.g. mock in tests),
    falling back to the module-level config module. Calling this at runtime
    (not import time) is what makes env-variable overrides effective after boot."""
    src = cfg if cfg is not None else _config
    return getattr(src, name, default)


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
        model_dir: Optional[str] = None,
        liquidity_monitor: Any = None,
    ):
        self.broker = broker
        self.config = config
        self.liquidity_monitor = liquidity_monitor

        # Models
        self.tactical_model = tactical_model
        self.strategic_model = strategic_model
        self.model_dir = model_dir
        self._last_saved_at = {
            mt: (getattr(m, "metadata", {}) or {}).get("saved_at")
            for mt, m in (("tactical", tactical_model), ("strategic", strategic_model))
            if m is not None
        }

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

        # Phase 2 adaptive threshold state
        self.pred_history: List[float] = []
        self._last_tactical_pred: float = 0.0

        # Phase 2/3 state (mirrors MockBroker in simulate.py)
        self.trail_extreme = 0.0
        self.scale_count = 0
        self.same_dir_streak = 0
        self.reversal_streak = 0
        self.last_entry_time: Optional[datetime] = None

        # Phase 1 feature-frame cache: {timeframe_key: (cached_at, df_feat, last_ts)}
        self._feature_cache: Dict[str, Tuple[float, Any, Any]] = {}
        self._last_atr14: float = 0.0
        self._initial_qty: float = 0.0

        # Phase 2: rolling raw prediction history (for retrain cadence only)

        # Threshold
        self.threshold = getattr(config, 'ABSOLUTE_THRESHOLD', ABSOLUTE_THRESHOLD) if config else ABSOLUTE_THRESHOLD

        # Walk-forward retraining
        self.retrain_every = getattr(config, 'WALKFORWARD_RETRAIN_EVERY', 100) if config else 100
        self.candles_since_retrain = 0
        self.model_refresh_every = getattr(config, 'MODEL_REFRESH_EVERY', 50) if config else 50

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

        # Phase 2/3 state
        self.pred_history = []
        self.trail_extreme = 0.0
        self.scale_count = 0
        self.same_dir_streak = 0
        self.reversal_streak = 0
        self.last_entry_time = None
        self._feature_cache = {}
        self._last_atr14 = 0.0
        self._initial_qty = 0.0

    def _retrain_tactical(self, df_raw: pd.DataFrame) -> None:
        """Retrain the tactical model on the recent window (mirrors walk-forward)."""
        from catboost import CatBoostRegressor
        from model import TARGET_COLUMN
        df_feat = make_features_df(df_raw, timeframe=TACTICAL_TF)
        df_lbl = make_labels_df(df_feat, timeframe=TACTICAL_TF).dropna()
        if len(df_lbl) < 100 or not self.feature_cols:
            return
        feats = [c for c in self.feature_cols if c in df_lbl.columns]
        base = CatBoostRegressor(**self.tactical_model.params)
        base.fit(
            df_lbl[feats].fillna(0),
            df_lbl[TARGET_COLUMN].fillna(0),
            verbose=False,
        )
        self.tactical_model.model = base
        log_debug(f"Tactical model retrained on {len(df_lbl)} recent candles")

    def _tactical_model_is_degenerate(self) -> bool:
        """True when the loaded tactical model is a degenerate artifact.

        The frozen model can be early-stopped at iteration 0 (1 tree), which
        predicts ~constant values that never cross the threshold. When
        detected, the live loop retrains immediately instead of being blind
        for the first retrain_every candles.
        """
        model = getattr(self.tactical_model, "model", None)
        if model is None:
            return False  # handled by the model-is-None branch in the caller
        n_trees = getattr(model, "tree_count_", None)
        if n_trees is None:
            return False  # attribute unavailable -> assume healthy
        return n_trees < 2

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

        self._sleep_seconds = sleep_seconds
        self._tactical_window = getattr(self, "_tactical_window", 200)
        self._strategic_window = getattr(self, "_strategic_window", 500)

        model_refresh_every = self.model_refresh_every
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            try:
                iteration += 1
                if self.model_dir and iteration % model_refresh_every == 0:
                    self._refresh_models()
                self._live_iteration()

                summary_every = int(
                    _cfg_flag("DECISION_LOG_INTERVAL", 300, self.config)
                )
                if summary_every > 0 and iteration % summary_every == 0:
                    self._log_state_summary(iteration, sleep_seconds)
            except KeyboardInterrupt:
                log_info("Interrupted by user")
                self._close_all_positions("manual_shutdown")
                break
            except Exception as e:
                log_error(f"Live iteration error: {e}")
                log_error(f"Retrying in {sleep_seconds}s...")

            try:
                time.sleep(sleep_seconds)
            except KeyboardInterrupt:
                log_info("Interrupted by user")
                self._close_all_positions("manual_shutdown")
                break

    def _refresh_models(self) -> None:
        """Hot-swap tactic/strategic models from disk when a newer model is found.

        Training may run in a separate process; reload when the on-disk meta
        'saved_at' is newer than the model currently loaded so live trading
        picks up freshly trained models without a restart.
        """
        from model import CatBoostModel
        if not self.model_dir:
            return
        for model_type in ("tactical", "strategic"):
            current = getattr(self, f"{model_type}_model")
            if current is None:
                continue
            meta_path = Path(self.model_dir) / f"model_{model_type}_meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    frozen_saved_at = json.load(f).get("saved_at")
            except (json.JSONDecodeError, OSError):
                continue
            if not frozen_saved_at or frozen_saved_at == self._last_saved_at.get(model_type):
                continue
            try:
                fresh = CatBoostModel(model_type=model_type, model_dir=self.model_dir)
                fresh.load()
            except Exception as e:
                log_warning(f"Hot-swap {model_type} model failed: {e}")
                continue
            setattr(self, f"{model_type}_model", fresh)
            self._last_saved_at[model_type] = frozen_saved_at
            log_info(f"Hot-swapped {model_type} model (saved_at={frozen_saved_at})")
            if model_type == "tactical":
                new_feats = list((fresh.metadata or {}).get("feature_cols", []))
                if new_feats:
                    self.feature_cols = new_feats
                    self.candles_since_retrain = self.retrain_every

    def _close_all_positions(self, reason: str = "manual_shutdown") -> None:
        """Force-close any open position and cancel open orders on shutdown."""
        if self.position != 0:
            price = self.broker.get_last_price()
            if price > 0:
                try:
                    self._exit_position(reason, price, datetime.now())
                except Exception as e:
                    log_error(f"Shutdown close failed: {e}")
            else:
                log_warning(f"Shutdown: could not fetch price to close {self.position:+}")
        try:
            self.broker.cancel_open_orders()
        except Exception as e:
            log_error(f"Shutdown cancel orders failed: {e}")
        log_info("All positions closed, open orders cancelled")

    def _cached_features(self, df_raw: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """Phase 1: TTL-cached feature-frame build.

        Reuses the previously built feature frame while it is fresh
        (FEATURE_CACHE_TTL seconds) AND the latest raw candle is unchanged.
        Live data is appended one candle at a time, so a fresh cache with the
        same last timestamp is always valid. Falls back to rebuilding.
        """
        now_ts = time.time()
        last_ts = df_raw.index[-1] if df_raw is not None and len(df_raw) > 0 else None

        cached = self._feature_cache.get(timeframe)
        if cached is not None:
            cached_at, cached_feat, cached_last = cached
            ttl = float(_cfg_flag("FEATURE_CACHE_TTL", FEATURE_CACHE_TTL, self.config))
            if (now_ts - cached_at) < ttl and cached_last == last_ts:
                return cached_feat

        df_feat = make_features_df(df_raw, timeframe=timeframe)
        if df_feat is None or len(df_feat) == 0:
            return None
        self._feature_cache[timeframe] = (now_ts, df_feat, last_ts)
        return df_feat

    def _current_threshold(self) -> float:
        """Phase 2: adaptive threshold from recent |preds| (fallback: static)."""
        if not _cfg_flag("ADAPTIVE_THRESHOLD_ENABLED", True, self.config):
            return self.threshold
        return adaptive_threshold(
            pd.Series(self.pred_history),
            lookback=int(_cfg_flag("ADAPTIVE_LOOKBACK", 200, self.config)),
            quantile=float(_cfg_flag("ADAPTIVE_QUANTILE", 0.95, self.config)),
            min_threshold=float(_cfg_flag("ADAPTIVE_MIN_THRESHOLD", 0.002, self.config)),
            max_threshold=float(_cfg_flag("ADAPTIVE_MAX_THRESHOLD", 0.02, self.config)),
            fallback=self.threshold,
        )

    def _live_iteration(self) -> None:
        """Fetch data, predict, and act on the latest candle."""
        now = datetime.now()
        iteration = getattr(self, "_iteration", 0)
        self._iteration = iteration + 1

        # Tactical (15m) data + feature frame (Phase 1: TTL cache)
        df_tactical = self.broker.get_historical_prices(
            self.broker.symbol, self._tactical_window, TACTICAL_TF
        )
        if df_tactical is None or len(df_tactical) < self._tactical_window:
            log_warning("Insufficient tactical data, skipping iteration")
            return

        df_feat = self._cached_features(df_tactical, TACTICAL_TF)
        if df_feat is None or len(df_feat) == 0 or not self.feature_cols:
            log_warning("Tactical features incomplete, skipping iteration")
            return

        # Walk-forward tactical retrain (mirrors rolling_tactical_predict used
        # by simulation): periodically rebuild the model on the recent window so
        # live matches sim. The frozen full-history model compresses predictions
        # to ~0 (never crosses threshold); the recent-window retrain recovers
        # the signal that simulation exploits.
        self.candles_since_retrain = getattr(self, "candles_since_retrain", 0) + 1
        if (self.candles_since_retrain >= self.retrain_every
                or self.tactical_model is None
                or self.tactical_model.model is None
                or self._tactical_model_is_degenerate()):
            self._retrain_tactical(df_tactical)
            self.candles_since_retrain = 0

        # Predict + adaptive threshold (Phase 2)
        row = df_feat.iloc[[-1]][[c for c in self.feature_cols if c in df_feat.columns]].fillna(0)
        pred = float(self.tactical_model.predict(row, list(row.columns)).iloc[0])
        if not np.isnan(pred):
            self.pred_history.append(float(pred))
            lookback = int(_cfg_flag("ADAPTIVE_LOOKBACK", 200, self.config))
            if len(self.pred_history) > lookback:
                self.pred_history = self.pred_history[-lookback:]
        self._last_tactical_pred = float(pred) if not np.isnan(pred) else 0.0
        thr = self._current_threshold()
        signal = prediction_to_signal(pred, thr)

        # Stash latest ATR14 for the trailing stop (Phase 2)
        self._last_atr14 = float(df_feat.iloc[-1].get("atr14", 0.0) or 0.0)
        last_row = df_feat.iloc[-1]

        # Strategic meta-params refresh roughly hourly
        strat_every = max(1, int(3600 / max(self._sleep_seconds, 1)))
        if (iteration % strat_every) == 0 or self.current_meta.get("_init") is None:
            df_strat = self.broker.get_historical_prices(
                self.broker.symbol, self._strategic_window, STRATEGIC_TF
            )
            if df_strat is not None and len(df_strat) > 50:
                df_strat_feat = add_strategic_features_df(
                    make_features_df(df_strat, timeframe=STRATEGIC_TF),
                    timeframe=STRATEGIC_TF,
                )
                strat_feats = list(
                    (self.strategic_model.metadata or {}).get("feature_cols", [])
                )
                strat_feats = [c for c in strat_feats if c in df_strat_feat.columns]
                try:
                    meta_list = predict_strategic_meta_params(
                        df_strat_feat, self.strategic_model, strat_feats,
                    )
                    meta = meta_list[-1]
                    meta["_init"] = True
                    self.current_meta = meta
                except Exception as e:
                    log_warning(f"Strategic prediction failed, using defaults: {e}")
                    self.current_meta = DEFAULT_META.copy()
            else:
                self.current_meta = DEFAULT_META.copy()

        current_price = self.broker.get_last_price()
        if current_price <= 0:
            log_warning("Price fetch failed, skipping iteration")
            return

        # Exits are ALWAYS evaluated before entry gating (trailing stop must
        # run during hold/chop too, matching simulate.step()).
        exit_reason = self._check_exits(current_price, now)
        if exit_reason is not None and self.position != 0:
            self._exit_position(exit_reason, current_price, now)

        # ── Decision logging (Phase 2/3 reasons) ───────────────────────
        regime = self.current_meta.get("regime", "unknown")
        adaptive_on = bool(_cfg_flag("ADAPTIVE_THRESHOLD_ENABLED", True, self.config))
        thr_mode = "adapt" if (adaptive_on and len(self.pred_history) >= 20) else "static"
        vol_12 = last_row.get("vol_12", 0.0)
        vol_48 = last_row.get("vol_48", 0.0)
        vol_ratio = vol_12 / max(vol_48, 1e-8)
        vol_state = classify_vol_state(
            vol_ratio, float(_cfg_flag("EXTREME_VOL_RATIO", 1.8, self.config))
        )
        cooldown_left = 0.0
        if (
            self.position == 0
            and self.last_entry_time is not None
        ):
            _cd_min = float(_cfg_flag("TRADE_COOLDOWN_MINUTES", 30, self.config))
            cooldown_left = max(
                0.0, _cd_min - (now - self.last_entry_time).total_seconds() / 60.0
            )

        # Phase 2: extreme-volatility gate blocks ENTRIES only (exits already done)
        vol_blocked = False
        if signal in ("long", "short") and _cfg_flag("GATE_EXTREME_VOL", True, self.config):
            if vol_state == "extreme":
                vol_blocked = True
                signal = "hold"
                log_warning(
                    f"GATE vol: extreme vol_ratio={vol_ratio:.2f} "
                    f"(>={_cfg_flag('EXTREME_VOL_RATIO', 1.8, self.config)}) "
                    f"blocked {pred:+.6f} entry"
                )

        # Phase 1-2: liquidity gate blocks ENTRIES only (exits ran above).
        # Fail-open: stale/missing book data does NOT block (LIQUIDITY_FAIL_OPEN_ON_STALE).
        liq_blocked = None
        if (signal in ("long", "short")
                and _cfg_flag("GATE_LIQUIDITY", True, self.config)
                and self.liquidity_monitor is not None):
            liq_blocked = self._liquidity_entry_gate(signal)
            if liq_blocked:
                signal = "hold"
                log_warning(
                    f"GATE liq: {liq_blocked} blocked {pred:+.6f} entry"
                )

        # Chop/hold gate blocks ENTRIES only (exits ran above)
        if self.current_meta.get("regime") == "chop" or signal == "hold":
            if self.current_meta.get("regime") == "chop" and signal in ("long", "short"):
                log_warning(
                    f"GATE regime: chop market blocked {signal} entry "
                    f"(pred={pred:+.6f} thr={thr:.6f})"
                )
            reason = (
                "vol_gate" if vol_blocked
                else "liq_gate" if liq_blocked
                else "chop_gate" if self.current_meta.get("regime") == "chop"
                else "below_thr" if not np.isnan(pred) and abs(pred) <= thr
                else "nan_pred"
            )
            log_debug(
                f"DECISION iter={self._iteration} pred={pred:+.6f} "
                f"thr={thr:.6f}({thr_mode}) regime={regime} vol={vol_ratio:.2f}({vol_state}) "
                f"pos={self.position:+.6f} price={current_price:.2f} "
                f"atr14={self._last_atr14:.4f} cooldown_left={cooldown_left:.1f}m "
                f"signal=hold reason={reason}"
            )
            return

        # Phase 3: cooldown blocks new FLAT entries only
        if (
            self.position == 0
            and signal in ("long", "short")
            and _cfg_flag("TRADE_COOLDOWN_ENABLED", True, self.config)
            and self.last_entry_time is not None
        ):
            elapsed_min = (now - self.last_entry_time).total_seconds() / 60.0
            if elapsed_min < float(_cfg_flag("TRADE_COOLDOWN_MINUTES", 30, self.config)):
                log_debug(
                    f"DECISION iter={self._iteration} pred={pred:+.6f} "
                    f"thr={thr:.6f}({thr_mode}) regime={regime} vol={vol_ratio:.2f}({vol_state}) "
                    f"pos={self.position:+.6f} price={current_price:.2f} "
                    f"atr14={self._last_atr14:.4f} cooldown_left={cooldown_left:.1f}m "
                    f"signal=hold reason=cooldown"
                )
                signal = "hold"

        if signal in ("long", "short"):
            log_debug(
                f"DECISION iter={self._iteration} pred={pred:+.6f} "
                f"thr={thr:.6f}({thr_mode}) regime={regime} vol={vol_ratio:.2f}({vol_state}) "
                f"pos={self.position:+.6f} price={current_price:.2f} "
                f"atr14={self._last_atr14:.4f} cooldown_left={cooldown_left:.1f}m "
                f"signal={signal}"
            )

        # Entry / same-direction scaling / opposite-direction partial exit
        if self.position == 0 and signal in ("long", "short"):
            self._enter_position(signal, current_price, now)
        elif signal in ("long", "short") and self.position != 0:
            same_dir = (
                (signal == "long" and self.position > 0)
                or (signal == "short" and self.position < 0)
            )
            if same_dir:
                self._scale_in_live(signal, current_price, now)
            else:
                self._handle_opposite_live(signal, current_price, now)

    def _log_state_summary(self, iteration: int, sleep_seconds: int) -> None:
        price = 0.0
        try:
            price = self.broker.get_last_price()
        except Exception:
            pass
        equity = 0.0
        try:
            equity = self.broker.get_equity()
        except Exception:
            pass
        hist = [abs(p) for p in self.pred_history if p == p]
        recent = hist[-200:]
        pred_stats = {
            "preds": len(self.pred_history),
            "recent_n": len(recent),
            "recent_mean": (sum(recent) / len(recent)) if recent else 0.0,
        }
        regime = self.current_meta.get("regime", "unknown")
        summary_every = int(_cfg_flag("DECISION_LOG_INTERVAL", 300, self.config))
        if self.position != 0:
            log_info(
                f"SUM iter={iteration} pos={self.position:+.6f} "
                f"entry={self.entry_price:.2f} price={price:.2f} equity={equity:.6f} "
                f"regime={regime} preds={pred_stats['preds']} "
                f"recent_|pred|_mean={pred_stats['recent_mean']:.6f} "
                f"trail_extreme={self.trail_extreme:.2f} scale_count={self.scale_count}"
            )
        else:
            log_info(
                f"SUM iter={iteration} FLAT pos=0.0 price={price:.2f} equity={equity:.6f} "
                f"regime={regime} preds={pred_stats['preds']} "
                f"recent_|pred|_mean={pred_stats['recent_mean']:.6f} "
                f"next_summary_in={sleep_seconds * summary_every}s"
            )

    # ── Position Management ─────────────────────────────────────────────
    def _liquidity_entry_gate(self, side: str) -> Optional[str]:
        """Return block reason if orderbook too thin for a new entry, else None."""
        mon = self.liquidity_monitor
        if mon is None:
            return None
        snap = mon.snapshot()
        if snap is None:
            return None if _cfg_flag("LIQUIDITY_FAIL_OPEN_ON_STALE", True, self.config) else "stale_book"
        if not snap["fresh"] and not _cfg_flag("LIQUIDITY_FAIL_OPEN_ON_STALE", True, self.config):
            return "stale_book"
        if snap["spread_bps"] > float(_cfg_flag("LIQ_MAX_SPREAD_BPS", 10.0, self.config)):
            return "spread"
        side_depth = snap["depth_ask_usd"] if side == "long" else snap["depth_bid_usd"]
        if side_depth < float(_cfg_flag("LIQ_MIN_DEPTH_USD", 50000.0, self.config)):
            return "depth"
        if snap["vanish_pct"] > float(_cfg_flag("LIQ_MAX_VANISH_PCT", 60.0, self.config)):
            return "vanish"
        buy_ratio = snap["aggressor_buy_ratio"]
        if side == "long" and buy_ratio < float(_cfg_flag("LIQ_MIN_AGGRESSOR_RATIO", 0.35, self.config)):
            return "aggressor"
        if side == "short" and buy_ratio > float(_cfg_flag("LIQ_MAX_AGGRESSOR_RATIO", 0.65, self.config)):
            return "aggressor"
        if snap["trade_intensity_usd_s"] < float(_cfg_flag("LIQ_MIN_TRADE_INTENSITY_USD", 50000.0, self.config)):
            return "intensity"
        return None

    def _liquidity_exit_gate(self) -> Optional[str]:
        """Return 'liq_exit' when the book thins dangerously while holding."""
        mon = self.liquidity_monitor
        if mon is None:
            return None
        snap = mon.snapshot()
        if snap is None or not snap["fresh"]:
            return None
        if snap["spread_bps"] > float(_cfg_flag("LIQ_EXIT_SPREAD_BPS", 20.0, self.config)):
            return "liq_exit"
        if snap["depth_total_usd"] < float(_cfg_flag("LIQ_EXIT_MIN_DEPTH_USD", 25000.0, self.config)):
            return "liq_exit"
        return None

    def _check_exits(self, current_price: float, current_time: datetime) -> Optional[str]:
        """Check if position should be exited. Returns exit reason or None."""
        if self.position == 0:
            return None

        # Phase 1-2 emergency liquidity exit: fires FIRST, before price-based
        # exits, so a thinning book dumps the position before SL/TP fills slip
        # (Sept 11: SL slipped 230+ pts in a thin book). Fail-open: stale or
        # missing book data never forces an exit.
        if _cfg_flag("GATE_LIQUIDITY_EXIT", True, self.config):
            liq_reason = self._liquidity_exit_gate()
            if liq_reason is not None:
                return liq_reason

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

        # Trailing stop (Phase 2): raise SL to breakeven, then trail at ATR multiple.
        # Mirrors MockBroker._check_exit_conditions in simulate.py. Live only
        # sees the last price (no intra-bar high/low), so trail_extreme is
        # updated from current_price each iteration.
        if _cfg_flag("TRAILING_STOP_ENABLED", True, self.config) and self.position != 0:
            atr14 = self._last_atr14
            if atr14 and atr14 > 0:
                sl_frac = self.current_meta.get("stop_loss_frac", STOP_LOSS_FRAC_DEFAULT)
                atr_mult = float(_cfg_flag("TRAILING_ATR_MULT", 1.5, self.config))
                be_mult = float(_cfg_flag("TRAILING_BREAKEVEN_MULT", 1.0, self.config))
                if self.trail_extreme <= 0:
                    self.trail_extreme = self.entry_price
                if self.position > 0:
                    self.trail_extreme = max(self.trail_extreme, current_price)
                    breakeven_price = self.entry_price * (1 + be_mult * sl_frac)
                    if self.trail_extreme >= breakeven_price:
                        base_sl = self.entry_price * (1 - sl_frac)
                        trail_sl = self.trail_extreme - atr_mult * atr14
                        eff_sl = max(base_sl, trail_sl)
                        if current_price <= eff_sl and eff_sl > base_sl:
                            return "trailing_sl"
                else:
                    self.trail_extreme = min(self.trail_extreme, current_price)
                    breakeven_price = self.entry_price * (1 - be_mult * sl_frac)
                    if self.trail_extreme <= breakeven_price:
                        base_sl = self.entry_price * (1 + sl_frac)
                        trail_sl = self.trail_extreme + atr_mult * atr14
                        eff_sl = min(base_sl, trail_sl)
                        if current_price >= eff_sl and eff_sl < base_sl:
                            return "trailing_sl"

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

        result = self.broker.open_position(
            side=side,
            stake_frac=stake_frac,
            leverage=leverage,
            stop_loss_frac=self.current_meta["stop_loss_frac"],
            take_profit_frac=self.current_meta["take_profit_frac"],
        )
        if result is None or not getattr(result, "success", False):
            log_warning(f"Entry {side.upper()} FAILED: {getattr(result, 'error', 'no result')}")
            return

        # Reconcile from broker's actual fill (source of truth)
        actual_pos = self.broker.get_position()
        self.position = 1.0 if side == "long" else -1.0
        self.entry_price = (actual_pos.entry_price if actual_pos else None) or price
        self.entry_time = timestamp
        qty = abs(actual_pos.amount) if actual_pos else 0.0
        self._initial_qty = qty

        # Phase 2/3 state: fresh entry resets streaks, trail, scale count
        self.last_entry_time = timestamp
        self.trail_extreme = 0.0
        self.scale_count = 0
        self.same_dir_streak = 0
        self.reversal_streak = 0

        # Create trade record
        self.current_trade = log_trade_entry(
            timestamp=timestamp,
            symbol="BTCUSDT",
            side=side,
            entry_price=self.entry_price,
            qty=qty,
            stake_frac=stake_frac,
            leverage=leverage,
            stop_loss=self.current_meta["stop_loss_frac"],
            take_profit=self.current_meta["take_profit_frac"],
            max_hold_hours=self.current_meta["max_hold_hours"],
            regime=self.current_meta["regime"],
            tactical_pred=self._last_tactical_pred,
            strategic_params=self.current_meta.copy(),
            equity_before=self.broker.get_equity(),
        )

        log_info(f"LIVE ENTRY {side.upper()} @ {self.entry_price:.2f} | Meta: {json.dumps(self.current_meta, default=str)}")

    def _scale_in_live(self, signal: str, price: float, timestamp: datetime) -> None:
        """Phase 3: scale into an open position on a same-direction signal.

        Adds SCALE_STAKE_FRAC of the initial stake via broker.scale_in, re-
        reconciles from the broker's actual fill, and re-places the TP/SL
        bracket on the enlarged position (weighted-average entry).
        """
        if self.position == 0 or self.current_trade is None:
            return
        if not _cfg_flag("SCALING_ENABLED", True, self.config):
            return

        self.same_dir_streak += 1
        confirm_bars = int(_cfg_flag("SCALE_CONFIRM_BARS", 2, self.config))
        if self.same_dir_streak < confirm_bars:
            return
        if self.scale_count >= int(_cfg_flag("MAX_SCALE_COUNT", 2, self.config)):
            self.same_dir_streak = 0
            return

        # Add a SCALE_STAKE_FRAC-sized stake to the existing position
        base_qty = self._initial_qty or abs(self.current_trade.get("qty", 0.0))
        add_qty = base_qty * float(_cfg_flag("SCALE_STAKE_FRAC", 0.5, self.config))
        if add_qty <= 0:
            return

        broker_side = "BUY" if self.position > 0 else "SELL"
        result = self.broker.scale_in(self.broker.symbol, broker_side, add_qty)
        if result is None or not getattr(result, "amount", None):
            log_warning(f"Scale-in {signal.upper()} FAILED: {getattr(result, 'error', 'no result')}")
            self.same_dir_streak = 0
            return

        # Reconcile from broker's actual fill
        actual_pos = self.broker.get_position()
        if actual_pos is not None:
            self.entry_price = actual_pos.entry_price or self.entry_price
            self._initial_qty = abs(actual_pos.amount)
            self.current_trade["qty"] = self._initial_qty
        self.scale_count += 1
        self.same_dir_streak = 0

        # Re-place TP/SL bracket on the enlarged position
        try:
            self.broker.replace_bracket_order(
                self.broker.symbol,
                self._initial_qty,
                broker_side,
                tp_frac=self.current_meta.get("take_profit_frac", TAKE_PROFIT_FRAC_DEFAULT),
                sl_frac=self.current_meta.get("stop_loss_frac", STOP_LOSS_FRAC_DEFAULT),
            )
        except Exception as e:
            log_warning(f"replace_bracket_order after scale-in failed: {e}")

        log_info(
            f"LIVE SCALE-IN {signal.upper()} +{add_qty:.6f} -> {self._initial_qty:.6f} "
            f"@ {self.entry_price:.2f} (scale #{self.scale_count})"
        )

    def _handle_opposite_live(self, signal: str, price: float, timestamp: datetime) -> None:
        """Phase 3: opposite-direction signal while in position.

        Disabled (legacy): no-op on opposite signal (live never flips).
        Enabled: first reversal partially closes PARTIAL_EXIT_FRACTION; on the
        REVERSAL_FULL_CLOSE_STREAK-th consecutive reversal, full close + flip.
        Mirrors MockBroker._handle_reversal in simulate.py.
        """
        if self.position == 0:
            return

        if not _cfg_flag("PARTIAL_EXIT_ENABLED", True, self.config):
            # Legacy: ignore opposite signals while in position (live never flips)
            self.same_dir_streak = 0
            self.reversal_streak = 0
            return

        self.same_dir_streak = 0
        self.reversal_streak += 1
        if self.reversal_streak >= int(_cfg_flag("REVERSAL_FULL_CLOSE_STREAK", 2, self.config)):
            # Persistent reversal: full close, then flip if signal persists
            self.reversal_streak = 0
            self._exit_position("reversal", price, timestamp)
            if self.position == 0 and signal in ("long", "short"):
                self._enter_position(signal, price, timestamp)
        else:
            # First reversal: close PARTIAL_EXIT_FRACTION, keep the rest
            fraction = float(_cfg_flag("PARTIAL_EXIT_FRACTION", 0.33, self.config))
            fill_price = self.broker.close_position_fraction(self.broker.symbol, fraction)
            if fill_price is not None and self.current_trade is not None:
                qty = abs(self.current_trade.get("qty", 0.0))
                self.current_trade["qty"] = qty * (1.0 - fraction)
                if self._initial_qty > 0:
                    self._initial_qty *= (1.0 - fraction)
                log_info(
                    f"LIVE PARTIAL EXIT {signal.upper()} {fraction:.0%} @ {fill_price:.2f} "
                    f"| qty {self.current_trade['qty']:.6f}"
                )
            else:
                log_warning(f"Partial exit {signal.upper()} failed (no fill)")

    def _exit_position(self, reason: str, price: float, timestamp: datetime) -> None:
        """Close current position."""
        if self.position == 0:
            return

        side = "long" if self.position > 0 else "short"

        # Delegate to broker; broker.close_position() uses its own symbol/position
        fill_price = self.broker.close_position()
        actual_fill = fill_price or price

        if self.current_trade:
            entry_price = self.entry_price or actual_fill
            notional = abs(self.current_trade.get("qty", 0.0)) or 1.0
            raw_pnl = (
                (actual_fill - entry_price) * notional
                if self.position > 0
                else (entry_price - actual_fill) * notional
            )
            equity_before = self.current_trade.get("equity_before") or 1.0
            log_trade_exit(
                trade=self.current_trade,
                exit_price=actual_fill,
                exit_reason=reason,
                pnl=raw_pnl,
                pnl_pct=raw_pnl / equity_before if equity_before else 0.0,
                equity_after=self.broker.get_equity(),
                fee_paid=0.0,
                slippage_paid=0.0,
            )

        log_info(f"LIVE EXIT {reason.upper()} @ {actual_fill:.2f}")

        self.position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.current_trade = None

        # Phase 3: reset streaks/trail/scale but KEEP last_entry_time
        # (cooldown basis persists after an exit)
        self.trail_extreme = 0.0
        self.scale_count = 0
        self.same_dir_streak = 0
        self.reversal_streak = 0
        self._initial_qty = 0.0


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