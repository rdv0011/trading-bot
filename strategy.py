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
import signal
import logging

logger = logging.getLogger(__name__)

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
    ADAPTIVE_LIQ_ENABLED, ADAPTIVE_LIQ_LOOKBACK, ADAPTIVE_LIQ_QUANTILE,
    ADAPTIVE_LIQ_MIN_USD, ADAPTIVE_LIQ_MAX_USD,
    ADAPTIVE_LIQ_COOLDOWN_MIN,
    ADAPTIVE_TRAIL_UPDATE_ENABLED, ADAPTIVE_TRAIL_UPDATE_LOOKBACK,
    ADAPTIVE_TRAIL_UPDATE_QUANTILE, ADAPTIVE_TRAIL_UPDATE_MULT,
    ADAPTIVE_TRAIL_UPDATE_MIN, ADAPTIVE_TRAIL_UPDATE_MAX,
    TRAILING_STOP_ENABLED, TRAILING_ATR_MULT, TRAILING_BREAKEVEN_MULT,
    GATE_EXTREME_VOL, EXTREME_VOL_RATIO,
    SCALING_ENABLED, MAX_SCALE_COUNT, SCALE_CONFIRM_BARS, SCALE_STAKE_FRAC,
    PARTIAL_EXIT_ENABLED, PARTIAL_EXIT_FRACTION, REVERSAL_FULL_CLOSE_STREAK,
    TRADE_COOLDOWN_ENABLED, TRADE_COOLDOWN_MINUTES,
    MIN_TREES,
    is_testnet_env,
)
from logger import log_info, log_debug, log_warning, log_error, log_trade_entry, log_trade_exit, log_equity, log_cost_summary, log_edge_summary
from model import CatBoostModel, rolling_tactical_predict, strategic_batch_predict, predict_strategic_meta_params
from data import (make_features_df, add_strategic_features_df, make_labels_df,
                  adaptive_threshold, adaptive_liquidity_threshold, adaptive_trailing_threshold,
                  compute_atr_from_klines, compute_vol_ratio_from_klines,
                  classify_vol_state, classify_abs_vol_state)


# ── Flag Reader (runtime overrides → injected cfg → module config) ─────
RUNTIME_OVERRIDES_PATH = Path(__file__).parent / "runtime_overrides.json"
_RUNTIME_OVERRIDES: Dict[str, Any] = {}
_RUNTIME_OVERRIDES_MTIME: float = 0.0


def reload_runtime_overrides(path: Path = RUNTIME_OVERRIDES_PATH) -> bool:
    """Refresh live-tunable flags from a JSON file (e.g. runtime_overrides.json).

    Call this repeatedly (the live loop does, each iteration); it only re-parses
    the file when mtime actually changes. Values are applied to every _cfg_flag()
    read, letting flags like LIQ_EXIT_SPREAD_BPS be tuned on a running bot
    without restart. Fail-open: a missing file, corrupt JSON, or non-dict content
    leaves the previous overrides unchanged and returns False (never halts).

    Returns:
        True if overrides were (re)loaded, False if the file was missing,
        unchanged, or unparsable.
    """
    global _RUNTIME_OVERRIDES_MTIME
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    if mtime == _RUNTIME_OVERRIDES_MTIME:
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        log_warning(
            f"Runtime overrides file unparsable ({path}): {exc}; keeping previous overrides"
        )
        _RUNTIME_OVERRIDES_MTIME = mtime  # don't re-parse the same broken file every tick
        return False
    if not isinstance(data, dict):
        log_warning(
            f"Runtime overrides file {path}: expected JSON object, got {type(data).__name__}; "
            "keeping previous overrides"
        )
        _RUNTIME_OVERRIDES_MTIME = mtime
        return False
    _RUNTIME_OVERRIDES.update(data)
    _RUNTIME_OVERRIDES_MTIME = mtime
    log_info(f"Runtime overrides loaded from {path}: {_RUNTIME_OVERRIDES}")
    return True


def _cfg_flag(name: str, default: Any, cfg: Any = None) -> Any:
    """Read a runtime flag: live file override (highest), then an injected
    config object (e.g. mock in tests), then the module-level config module."""
    if name in _RUNTIME_OVERRIDES:
        return _RUNTIME_OVERRIDES[name]
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

        # Adaptive liquidity threshold state
        self.trade_intensity_history: List[float] = []
        self._last_liq_exit_time: Optional[datetime] = None

        # Adaptive trailing SL update threshold state
        self.atr_history: List[float] = []

        # Phase 2/3 state (mirrors MockBroker in simulate.py)
        self.trail_extreme = 0.0
        self.scale_count = 0
        self.same_dir_streak = 0
        self.reversal_streak = 0
        self.last_entry_time: Optional[datetime] = None

        # Partial exit tracking (R-multiple based profit taking)
        self._partial_exits_hit: int = 0           # Number of R-levels taken (0, 1, 2, 3)
        self._next_partial_r_level: int = 1        # Next R-level to target (1, 2, 3)

        # Phase 1 feature-frame cache: {timeframe_key: (cached_at, df_feat, last_ts)}
        self._feature_cache: Dict[str, Tuple[float, Any, Any]] = {}
        self._last_atr14: float = 0.0
        self._initial_qty: float = 0.0

        # Hybrid trailing SL/TP sync state (exchange-side order prices)
        self._last_synced_sl_price: Optional[float] = None
        self._last_synced_tp_price: Optional[float] = None
        self._last_sl_update_ts: float = 0.0

        # Phase 2: rolling raw prediction history (for retrain cadence only)

        # Threshold
        self.threshold = getattr(config, 'ABSOLUTE_THRESHOLD', ABSOLUTE_THRESHOLD) if config else ABSOLUTE_THRESHOLD

        # Walk-forward retraining
        self.retrain_every = getattr(config, 'WALKFORWARD_RETRAIN_EVERY', 100) if config else 100
        self.candles_since_retrain = 0
        self.model_refresh_every = getattr(config, 'MODEL_REFRESH_EVERY', 50) if config else 50

    def seed_trade_intensity_history(self, initial_intensity: float, num_samples: int = 50) -> None:
        """Seed the trade_intensity_history with an estimated initial value.
        
        This gives the adaptive liquidity threshold a sensible starting point
        instead of using the hardcoded fallback during warmup.
        
        Args:
            initial_intensity: Estimated trade intensity in USD/s from recent market data
            num_samples: Number of history samples to pre-populate (default 50)
        """
        if initial_intensity is None or initial_intensity <= 0:
            log_warning("Invalid initial trade intensity, skipping history seeding")
            return
        
        # Clamp to configured bounds
        min_usd = float(_cfg_flag("ADAPTIVE_LIQ_MIN_USD", 5000.0, self.config))
        max_usd = float(_cfg_flag("ADAPTIVE_LIQ_MAX_USD", 50000.0, self.config))
        clamped = max(min_usd, min(max_usd, initial_intensity))
        
        # Pre-populate history with the clamped value
        # Use slightly varied values to avoid percentile calculation issues
        import random
        self.trade_intensity_history = [
            clamped * (1 + random.uniform(-0.05, 0.05)) 
            for _ in range(num_samples)
        ]
        
        log_info(
            f"[LIQUIDITY INIT] Seeded trade_intensity_history with {num_samples} samples "
            f"around {clamped:.0f} USD/s (raw={initial_intensity:.0f}, bounds=[{min_usd:.0f}, {max_usd:.0f}])"
        )

    def seed_atr_history(self, df_klines: pd.DataFrame, num_samples: int = 50) -> None:
        """Seed the atr_history with ATR values computed from historical klines.
        
        This gives the adaptive trailing update threshold a realistic starting point
        instead of using the hardcoded fallback during warmup.
        
        Args:
            df_klines: DataFrame with OHLCV columns (high, low, close)
            num_samples: Number of history samples to pre-populate (default 50)
        """
        if df_klines is None or len(df_klines) < 20:
            log_warning("Insufficient klines data for ATR history seeding")
            return
        
        # Compute ATR from klines
        atr_series = compute_atr_from_klines(df_klines, period=14)
        if atr_series is None or len(atr_series) == 0 or atr_series.dropna().empty:
            log_warning("ATR computation failed, skipping history seeding")
            return
        
        # Get recent valid ATR values
        valid_atr = atr_series.dropna().tail(min(num_samples, len(atr_series.dropna())))
        if len(valid_atr) < 5:
            log_warning(f"Only {len(valid_atr)} valid ATR values, need at least 5")
            return
        
        # Store raw ATR values (adaptive_trailing_threshold handles min/max internally)
        # Add slight variation to avoid percentile edge cases
        import random
        self.atr_history = [
            float(val * (1 + random.uniform(-0.02, 0.02))) 
            for val in valid_atr
        ]
        
        log_info(
            f"[ATR INIT] Seeded atr_history with {len(self.atr_history)} samples "
            f"median={valid_atr.median():.2f} (raw median={atr_series.dropna().median():.2f})"
        )

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

        # Hybrid trailing sync state
        self._last_synced_sl_price = None
        self._last_synced_tp_price = None
        self._last_sl_update_ts = 0.0

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
        n_trees = getattr(base, "tree_count_", 0)
        if n_trees < MIN_TREES:
            log_warning(
                f"Tactical retrain produced only {n_trees} tree(s) "
                f"(< {MIN_TREES}); retraining without early stopping..."
            )
            base2 = CatBoostRegressor(**self.tactical_model.params)
            base2.fit(
                df_lbl[feats].fillna(0),
                df_lbl[TARGET_COLUMN].fillna(0),
                verbose=False,
            )
            n_trees = getattr(base2, "tree_count_", 0)
            base = base2
        self.tactical_model.model = base
        log_debug(f"Tactical model retrained on {len(df_lbl)} recent candles ({n_trees} trees)")

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
        return n_trees < MIN_TREES

    # ── Live Trading Mode ──────────────────────────────────────────────
    def sync_exchange_position(self, symbol: Optional[str] = None) -> bool:
        """Adopt any open position already on the exchange at startup.

        Called once before ``run_live_loop`` begins. If a previous process
        left a position open (killed session, crash), the live bot would
        otherwise start flat and trade blindly beside it — opening a new
        (possibly opposite) position while the orphan is unmanaged. This
        mirrors the exchange position into strategy state so SL/TP,
        trailing, liquidity and manual-shutdown exits manage it instead.

        Fail-open: any query/parse error is treated as "no position" so a
        transient API hiccup never blocks startup.

        Returns:
            True  -> caller may proceed (position adopted, or account flat).
            False -> an open position exists but adoption is disabled
                     (``ADOPT_EXISTING_POSITION=False``); refuse to start.
        """
        symbol = symbol or getattr(self.broker, "symbol", "BTCUSDT")

        try:
            pos = self.broker.get_position(symbol)
        except Exception as e:
            log_warning(f"Position sync: query failed ({e}); starting flat")
            self.position = 0.0
            return True

        # Defensive parse: broker mocks / odd payloads must never crash startup.
        if pos is None:
            log_info("Position sync: no open position on exchange, starting flat")
            self.position = 0.0
            return True

        amount_raw = getattr(pos, "amount", None)
        entry_raw = getattr(pos, "entry_price", None)
        if not isinstance(amount_raw, (int, float)) or not isinstance(entry_raw, (int, float)):
            log_warning("Position sync: unparseable position payload, starting flat")
            self.position = 0.0
            return True

        amount = float(amount_raw)
        entry_price = float(entry_raw)

        if abs(amount) < 1e-9:
            log_info("Position sync: no open position on exchange, starting flat")
            self.position = 0.0
            return True

        if not _cfg_flag("ADOPT_EXISTING_POSITION", True, self.config):
            log_error(
                f"Position sync: exchange holds {amount:+.6f} {symbol} "
                f"@ {entry_price:.2f} but ADOPT_EXISTING_POSITION=False; "
                "refusing to start (close it manually or enable adoption)"
            )
            return False

        side = "long" if amount > 0 else "short"
        self.position = 1.0 if amount > 0 else -1.0
        self.entry_price = entry_price
        # Conservative: max_hold counts from adoption, not true fill time.
        self.entry_time = datetime.now()
        self._initial_qty = abs(amount)
        self.trail_extreme = entry_price
        self.last_entry_time = self.entry_time
        self.scale_count = 0
        self.same_dir_streak = 0
        self.reversal_streak = 0

        # Rebuild a trade record so exit accounting/PnL logging works normally.
        meta = self.current_meta or DEFAULT_META.copy()
        stake_frac = meta.get(
            "stake_long_frac" if side == "long" else "stake_short_frac",
            STAKE_LONG_FRAC_DEFAULT if side == "long" else STAKE_SHORT_FRAC_DEFAULT,
        )
        self.current_trade = log_trade_entry(
            signal_time=self.entry_time,
            fill_time=self.entry_time,
            symbol=symbol,
            side=side,
            requested_price=self.entry_price,
            fill_price=self.entry_price,
            qty=abs(amount),
            stake_frac=stake_frac,
            leverage=meta.get("recommended_leverage", LEVERAGE_DEFAULT),
            stop_loss=meta.get("stop_loss_frac", STOP_LOSS_FRAC_DEFAULT),
            take_profit=meta.get("take_profit_frac", TAKE_PROFIT_FRAC_DEFAULT),
            max_hold_hours=meta.get("max_hold_hours", MAX_HOLD_HOURS_DEFAULT),
            regime=meta.get("regime", "trend"),
            tactical_pred=self._last_tactical_pred,
            strategic_params=meta.copy(),
            equity_before=self.broker.get_equity(),
        )

        log_info(
            f"ADOPTED existing {side.upper()} qty={abs(amount):.6f} "
            f"entry={self.entry_price:.2f} | managing exchange position "
            f"(max_hold counts from adoption; existing SL/TP brackets reused)"
        )
        return True

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

        self._install_shutdown_signal_handlers()

        model_refresh_every = self.model_refresh_every
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            try:
                iteration += 1
                reload_runtime_overrides()
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

    def _install_shutdown_signal_handlers(self) -> None:
        """Close positions on SIGTERM/SIGHUP so the exchange is never left orphaned.

        ``tmux kill-session`` delivers SIGHUP and systemd stop delivers SIGTERM;
        without a handler the process dies mid-loop and leaves an open position
        that the next boot must re-adopt. Routing both signals through the same
        graceful close path as Ctrl+C keeps that invariant.
        """
        def _shutdown_handler(signum, _frame):
            log_info(f"Received signal {signum}, closing all positions")
            try:
                self._close_all_positions("manual_shutdown")
            except Exception as e:
                log_error(f"Signal shutdown close failed: {e}")
            finally:
                raise SystemExit(0)

        try:
            signal.signal(signal.SIGTERM, _shutdown_handler)
            signal.signal(signal.SIGHUP, _shutdown_handler)
        except (ValueError, OSError) as e:
            log_warning(f"Could not install shutdown signal handlers: {e}")

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

        # Sync trailing SL/TP to exchange when position is still open
        if self.position != 0:
            self._sync_trailing_orders(current_price)

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
        # Absolute volume ratio (Oct 5 post-mortem: relative vol_ratio masks volume vacuums)
        abs_vol_ratio = last_row.get("abs_vol_ratio", 1.0)
        abs_vol_state = classify_abs_vol_state(
            abs_vol_ratio, float(_cfg_flag("ABS_VOL_RATIO_CRITICAL", 0.10, self.config))
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

        # Phase 2b: absolute volume gate blocks ENTRIES when market activity
        # is too low (e.g., weekend, overnight, volume vacuum).
        # abs_vol_ratio = actual_volume / rolling_median_volume
        # Low abs_vol means unreliable price discovery → higher slippage risk.
        abs_vol_blocked = False
        if (signal in ("long", "short")
                and _cfg_flag("GATE_ABS_VOL_ENABLED", True, self.config)):
            abs_vol_critical = float(_cfg_flag("ABS_VOL_RATIO_CRITICAL", 0.10, self.config))
            if abs_vol_ratio < abs_vol_critical:
                abs_vol_blocked = True
                signal = "hold"
                log_warning(
                    f"GATE abs_vol: abs_vol_ratio={abs_vol_ratio:.3f} "
                    f"(<{abs_vol_critical:.3f}) blocked {pred:+.6f} entry"
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
        halt_regime = self.current_meta.get("regime") in ("chop", "halt")
        if halt_regime or signal == "hold":
            if self.current_meta.get("regime") == "chop" and signal in ("long", "short"):
                log_warning(
                    f"GATE regime: chop market blocked {signal} entry "
                    f"(pred={pred:+.6f} thr={thr:.6f})"
                )
            reason = (
                "abs_vol_gate" if abs_vol_blocked
                else "vol_gate" if vol_blocked
                else "liq_gate" if liq_blocked
                else "halt_regime" if self.current_meta.get("regime") == "halt"
                else "chop_gate" if self.current_meta.get("regime") == "chop"
                else "below_thr" if not np.isnan(pred) and abs(pred) <= thr
                else "nan_pred"
            )
            thr_detail = ""
            if thr_mode == "adapt":
                q95 = int(_cfg_flag("ADAPTIVE_QUANTILE", 0.95, self.config) * 100)
                lookback = _cfg_flag("ADAPTIVE_LOOKBACK", 200, self.config)
                thr_detail = f" q95={q95} lookback={lookback} preds={len(self.pred_history)}"
            log_info(
                f"DECISION iter={self._iteration} pred={pred:+.6f} "
                f"thr={thr:.6f}({thr_mode}{thr_detail}) regime={regime} vol={vol_ratio:.2f}({vol_state}) "
                f"abs_vol={abs_vol_ratio:.3f}({abs_vol_state}) "
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

        # Spread-aware entry gate (Improvement 3): block entries when spread
        # is wide — wide spreads mean higher entry costs for the same expected move.
        if signal in ("long", "short") and self.position == 0:
            spread_bps = 0.0
            if hasattr(self.broker, "get_market_snapshot"):
                try:
                    bid, ask, mid = self.broker.get_market_snapshot()
                    if mid > 0:
                        spread_bps = (ask - bid) / mid * 10000
                except Exception as e:
                    log_debug(f"Spread gate: failed to get market snapshot: {e}")
            spread_threshold = float(_cfg_flag("SPREAD_ENTRY_GATE_BPS", 20, self.config))
            if spread_bps > spread_threshold:
                signal = "hold"
                log_warning(
                    f"GATE spread: {spread_bps:.1f}bps > {spread_threshold:.1f}bps "
                    f"blocked {pred:+.6f} entry"
                )

        if signal in ("long", "short"):
            log_debug(
                f"DECISION iter={self._iteration} pred={pred:+.6f} "
                f"thr={thr:.6f}({thr_mode}) regime={regime} vol={vol_ratio:.2f}({vol_state}) "
                f"abs_vol={abs_vol_ratio:.3f}({abs_vol_state}) "
                f"pos={self.position:+.6f} price={current_price:.2f} "
                f"atr14={self._last_atr14:.4f} cooldown_left={cooldown_left:.1f}m "
                f"signal={signal}"
            )

        # Entry / same-direction scaling / opposite-direction partial exit
        if self.position == 0 and signal in ("long", "short"):
            self._enter_position(signal, current_price, now, pred=pred, threshold=thr)
        elif signal in ("long", "short") and self.position != 0:
            same_dir = (
                (signal == "long" and self.position > 0)
                or (signal == "short" and self.position < 0)
            )
            if same_dir:
                self._scale_in_live(signal, current_price, now, pred=pred, threshold=thr)
            else:
                self._handle_opposite_live(signal, current_price, now)

    def _log_state_summary(self, iteration: int, sleep_seconds: int) -> None:
        price = 0.0
        try:
            price = self.broker.get_last_price()
        except Exception as e:
            log_debug(f"_log_state_summary: get_last_price failed: {e}")
        equity = 0.0
        try:
            equity = self.broker.get_equity()
        except Exception as e:
            log_debug(f"_log_state_summary: get_equity failed: {e}")
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

    # ── Confidence-Weighted Position Sizing ──────────────────────────────
    def _confidence_scaled_stake(self, base_stake: float, pred: float, threshold: float) -> float:
        """Scale stake fraction by prediction confidence relative to threshold.

        Uses sqrt scaling (diminishing returns) with a hard cap to avoid
        over-exposure on outlier predictions.
        """
        if threshold <= 0 or pred == 0:
            return base_stake
        confidence = abs(pred) / threshold
        power = float(_cfg_flag("CONFIDENCE_STAKE_POWER", 0.5, self.config))
        cap = float(_cfg_flag("CONFIDENCE_STAKE_CAP", 3.0, self.config))
        factor = min(confidence ** power, cap)
        return base_stake * factor

    def _liquidity_entry_gate(self, side: str) -> Optional[str]:
        """Return block reason if orderbook too thin for a new entry, else None.

        Uses the recorder's ``health_report()`` for hard-block checks
        (inverted book, dead tape) that override fail-open semantics, then
        falls back to per-metric gates on the raw snapshot.
        """
        mon = self.liquidity_monitor
        if mon is None:
            return None

        # ── Hard-blocks from health report (override fail-open) ──────────
        health = mon.health_report()
        if health["status"] == "no_data":
            return None if _cfg_flag("LIQUIDITY_FAIL_OPEN_ON_STALE", True, self.config) else "stale_book"
        if not health["safe_to_trade"]:
            if "inverted_book" in health["issues"]:
                return "inverted_book"
            if "dead_tape" in health["issues"]:
                return "dead_tape"

        # ── Per-metric gates (fail-open on stale) ────────────────────────
        snap = health["snap"]
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
        
        # Adaptive or static trade intensity threshold
        intensity = snap["trade_intensity_usd_s"]
        if _cfg_flag("ADAPTIVE_LIQ_ENABLED", True, self.config):
            # Add to history
            self.trade_intensity_history.append(intensity)
            lookback = int(_cfg_flag("ADAPTIVE_LIQ_LOOKBACK", 200, self.config))
            if len(self.trade_intensity_history) > lookback:
                self.trade_intensity_history = self.trade_intensity_history[-lookback:]
            min_intensity = adaptive_liquidity_threshold(
                pd.Series(self.trade_intensity_history),
                lookback=lookback,
                quantile=float(_cfg_flag("ADAPTIVE_LIQ_QUANTILE", 0.25, self.config)),
                min_threshold=float(_cfg_flag("ADAPTIVE_LIQ_MIN_USD", 10000.0, self.config)),
                max_threshold=float(_cfg_flag("ADAPTIVE_LIQ_MAX_USD", 100000.0, self.config)),
            )
        else:
            min_intensity = float(_cfg_flag("LIQ_MIN_TRADE_INTENSITY_USD", 25000.0, self.config))
        
        if intensity < min_intensity:
            return "intensity"
        return None

    def _liquidity_exit_gate(self, current_time: datetime) -> Optional[str]:
        """Return 'liq_exit' when the book thins dangerously while holding.

        Logs which specific metric triggered the exit for debugging.
        """
        mon = self.liquidity_monitor
        if mon is None:
            return None
        snap = mon.snapshot()
        if snap is None or not snap["fresh"]:
            return None
        intensity = snap.get("trade_intensity_usd_s", float('inf'))
        spread_bps = snap.get("spread_bps", 0)
        depth = snap.get("depth_total_usd", 0)
        vanish = snap.get("vanish_pct", 0)
        tape_trades = snap.get("trade_count_ps", 0) * 60  # default window_sec = 60s
        
        # Adaptive or static trade intensity threshold for exit
        if _cfg_flag("ADAPTIVE_LIQ_ENABLED", True, self.config):
            # Add to history
            self.trade_intensity_history.append(intensity)
            lookback = int(_cfg_flag("ADAPTIVE_LIQ_LOOKBACK", 200, self.config))
            if len(self.trade_intensity_history) > lookback:
                self.trade_intensity_history = self.trade_intensity_history[-lookback:]
            min_intensity = adaptive_liquidity_threshold(
                pd.Series(self.trade_intensity_history),
                lookback=lookback,
                quantile=float(_cfg_flag("ADAPTIVE_LIQ_QUANTILE", 0.25, self.config)),
                min_threshold=float(_cfg_flag("ADAPTIVE_LIQ_MIN_USD", 10000.0, self.config)),
                max_threshold=float(_cfg_flag("ADAPTIVE_LIQ_MAX_USD", 100000.0, self.config)),
            )
        else:
            min_intensity = float(_cfg_flag("LIQ_EXIT_MIN_TRADE_INTENSITY_USD", 25000.0, self.config))
        
        # Always log liquidity state when in position (for debugging)
        if self.position != 0 and hasattr(self, '_liq_log_counter'):
            self._liq_log_counter += 1
        else:
            self._liq_log_counter = 1
        
        if self.position != 0 and self._liq_log_counter % 10 == 0:  # every 10 iterations (~10 min)
            log_info(
                f"[LIQ STATE] pos={self.position} intensity={intensity:.0f}/{min_intensity:.0f} "
                f"spread={spread_bps:.1f}/{100.0} depth={depth:.0f}/{500000.0} "
                f"vanish={vanish:.1f}% tape={tape_trades:.0f} "
                f"adaptive={'on' if _cfg_flag('ADAPTIVE_LIQ_ENABLED', True, self.config) else 'off'}"
            )
        
        if intensity < min_intensity:
            # Cooldown: prevent repeated LIQ_EXIT triggers in short succession
            cooldown_min = float(_cfg_flag("ADAPTIVE_LIQ_COOLDOWN_MIN", 15, self.config))
            if self._last_liq_exit_time is not None:
                elapsed_min = (current_time - self._last_liq_exit_time).total_seconds() / 60.0
                if elapsed_min < cooldown_min:
                    log_debug(
                        f"Skipping LIQ_EXIT: {cooldown_min:.0f}min cooldown "
                        f"(elapsed={elapsed_min:.1f}min)"
                    )
                    return None
            self._last_liq_exit_time = current_time
            
            log_warning(
                f"LIQ_EXIT triggered: trade_intensity={intensity:.0f} "
                f"< threshold={min_intensity:.0f} USD/s "
                f"(spread={spread_bps:.1f}bps, "
                f"depth={depth:.0f}USD, "
                f"vanish={vanish:.1f}%, "
                f"tape_trades={tape_trades:.0f}/min)"
            )
            return "liq_exit"
        max_spread = float(_cfg_flag("LIQ_EXIT_SPREAD_BPS", 100.0, self.config))
        if spread_bps > max_spread:
            log_warning(
                f"LIQ_EXIT triggered: spread={spread_bps:.1f}bps "
                f"> threshold={max_spread:.1f}bps "
                f"(intensity={intensity:.0f}USD/s, "
                f"depth={depth:.0f}USD)"
            )
            return "liq_exit"
        min_depth = float(_cfg_flag("LIQ_EXIT_MIN_DEPTH_USD", 500000.0, self.config))
        if depth < min_depth:
            log_warning(
                f"LIQ_EXIT triggered: depth={depth:.0f}USD "
                f"< threshold={min_depth:.0f}USD "
                f"(intensity={intensity:.0f}USD/s, "
                f"spread={spread_bps:.1f}bps)"
            )
            return "liq_exit"
        return None


    def _check_exits(self, current_price: float, current_time: datetime) -> Optional[str]:
        """Check if position should be exited. Returns exit reason or None.

        Priority: time-based exit -> SL/TP -> trailing stop -> liquidity emergency.
        Price-based exits fire first; liquidity gate acts as emergency override
        only when market structure is collapsing (Oct 5 post-mortem fix).
        """
        if self.position == 0:
            return None

        # Time-based exit
        if self.entry_time is not None:
            hours_held = (current_time - self.entry_time).total_seconds() / 3600
            if hours_held >= self.current_meta["max_hold_hours"]:
                return "max_hold"

        # Stop loss / Take profit with time-weighted tightening
        # After 50% of max_hold_hours, tighten SL to half distance to limit prolonged losses
        # Use a separate variable to avoid mutating the original SL fraction used by trailing stop
        sl_frac = self.current_meta.get("stop_loss_frac", STOP_LOSS_FRAC_DEFAULT)
        sl_frac_tightened = sl_frac
        if self.entry_time is not None and self.position != 0:
            hours_held = (current_time - self.entry_time).total_seconds() / 3600
            max_hold = self.current_meta.get("max_hold_hours", MAX_HOLD_HOURS_DEFAULT)
            if hours_held >= max_hold * 0.5:
                sl_frac_tightened = sl_frac * 0.5  # Tighten SL to half

        # Calculate SL/TP prices based on position direction
        if self.position > 0:  # Long
            sl_price = self.entry_price * (1 - sl_frac_tightened)
            tp_price = self.entry_price * (1 + self.current_meta["take_profit_frac"])
        else:  # Short
            sl_price = self.entry_price * (1 + sl_frac_tightened)
            tp_price = self.entry_price * (1 - self.current_meta["take_profit_frac"])

        # DEBUG: Log SL check details
        if self.position != 0:
            if self.position > 0:  # Long: trigger when price <= SL
                sl_trigger = current_price <= sl_price
            else:  # Short: trigger when price >= SL
                sl_trigger = current_price >= sl_price
            log_debug(
                f"[SL CHECK] pos={self.position:+.6f} entry={self.entry_price:.2f} "
                f"sl_frac={sl_frac:.6f} sl_frac_tightened={sl_frac_tightened:.6f} "
                f"hours_held={hours_held:.2f} max_hold*0.5={max_hold * 0.5:.2f} "
                f"sl_price={sl_price:.2f} current={current_price:.2f} "
                f"trigger={sl_trigger}"
            )

        if self.position > 0:  # Long
            if current_price <= sl_price:
                return "sl"
            if current_price >= tp_price:
                return "tp"
        else:  # Short
            if current_price >= sl_price:
                return "sl"
            if current_price <= tp_price:
                return "tp"

        # Trailing stop (Phase 2): raise SL to breakeven, then trail at ATR multiple.
        # Mirrors MockBroker._check_exit_conditions in simulate.py. Live only
        # sees the last price (no intra-bar high/low), so trail_extreme is
        # updated from current_price each iteration.
        if _cfg_flag("TRAILING_STOP_ENABLED", True, self.config) and self.position != 0:
            atr14 = self._last_atr14
            if atr14 and atr14 > 0:
                # Use original (untightened) SL fraction for trailing stop calculations
                # This ensures breakeven and trailing levels are based on initial risk
                sl_frac = self.current_meta.get("stop_loss_frac", STOP_LOSS_FRAC_DEFAULT)
                atr_mult = float(_cfg_flag("TRAILING_ATR_MULT", 1.2, self.config))
                be_mult = float(_cfg_flag("TRAILING_BREAKEVEN_MULT", 1.0, self.config))
                if self.trail_extreme <= 0:
                    self.trail_extreme = self.entry_price
                if self.position > 0:
                    self.trail_extreme = max(self.trail_extreme, current_price)

                    # DEBUG: Log trailing SL check details
                    profit_dist = current_price - self.entry_price
                    breakeven_price = self.entry_price * (1 + be_mult * sl_frac)
                    base_sl = self.entry_price * (1 - sl_frac)
                    trail_sl = self.trail_extreme - atr_mult * atr14
                    eff_sl = max(base_sl, trail_sl)
                    log_debug(
                        f"[TRAIL CHECK] pos=long trail_extreme={self.trail_extreme:.2f} "
                        f"atr14={atr14:.2f} profit_dist={profit_dist:.2f} "
                        f"breakeven={breakeven_price:.2f} base_sl={base_sl:.2f} "
                        f"trail_sl={trail_sl:.2f} eff_sl={eff_sl:.2f} "
                        f"current={current_price:.2f} "
                        f"profit>=1R={profit_dist >= sl_frac * self.entry_price} "
                        f"trail_active={self.trail_extreme >= breakeven_price} "
                        f"below_eff={current_price <= eff_sl} "
                        f"eff>base={eff_sl > base_sl}"
                    )
                    # Profit-lock: once profit >= 1x SL distance, lock SL at breakeven
                    profit_dist = current_price - self.entry_price
                    if profit_dist >= sl_frac * self.entry_price:
                        # Lock SL at entry price — no more loss possible
                        if current_price <= self.entry_price:
                            return "trailing_sl"
                        # Trail from breakeven
                        trail_sl = self.entry_price - atr_mult * atr14
                        if current_price <= trail_sl:
                            return "trailing_sl"
                    else:
                        breakeven_price = self.entry_price * (1 + be_mult * sl_frac)
                        if self.trail_extreme >= breakeven_price:
                            base_sl = self.entry_price * (1 - sl_frac)
                            trail_sl = self.trail_extreme - atr_mult * atr14
                            eff_sl = max(base_sl, trail_sl)
                            if current_price <= eff_sl and eff_sl > base_sl:
                                return "trailing_sl"
                else:
                    self.trail_extreme = min(self.trail_extreme, current_price)
                    # Profit-lock: once profit >= 1x SL distance, lock SL at breakeven
                    profit_dist = self.entry_price - current_price

                    # DEBUG: Log trailing SL check details for short
                    breakeven_price = self.entry_price * (1 - be_mult * sl_frac)
                    base_sl = self.entry_price * (1 + sl_frac)
                    trail_sl = self.trail_extreme + atr_mult * atr14
                    eff_sl = min(base_sl, trail_sl)
                    log_debug(
                        f"[TRAIL CHECK] pos=short trail_extreme={self.trail_extreme:.2f} "
                        f"atr14={atr14:.2f} profit_dist={profit_dist:.2f} "
                        f"breakeven={breakeven_price:.2f} base_sl={base_sl:.2f} "
                        f"trail_sl={trail_sl:.2f} eff_sl={eff_sl:.2f} "
                        f"current={current_price:.2f} "
                        f"profit>=1R={profit_dist >= sl_frac * self.entry_price} "
                        f"trail_active={self.trail_extreme <= breakeven_price} "
                        f"above_eff={current_price >= eff_sl} "
                        f"eff<base={eff_sl < base_sl}"
                    )

                    # Profit-lock: once profit >= 1x SL distance, lock SL at breakeven
                    profit_dist = self.entry_price - current_price
                    if profit_dist >= sl_frac * self.entry_price:
                        if current_price >= self.entry_price:
                            return "trailing_sl"
                        trail_sl = self.entry_price + atr_mult * atr14
                        if current_price >= trail_sl:
                            return "trailing_sl"
                    else:
                        breakeven_price = self.entry_price * (1 - be_mult * sl_frac)
                        if self.trail_extreme <= breakeven_price:
                            base_sl = self.entry_price * (1 + sl_frac)
                            trail_sl = self.trail_extreme + atr_mult * atr14
                            eff_sl = min(base_sl, trail_sl)
                            if current_price >= eff_sl and eff_sl < base_sl:
                                return "trailing_sl"

        # Liquidity emergency override: fires LAST, only when book structure
        # is collapsing. This ensures SL/TP/trailing stop get the first chance
        # to exit at reasonable prices. Fail-open: stale or missing book data
        # never forces an exit.
        if _cfg_flag("GATE_LIQUIDITY_EXIT", True, self.config):
            liq_reason = self._liquidity_exit_gate(current_time)
            if liq_reason is not None:
                # Grace period: suppress liq_exit shortly after entry to avoid
                # "entry shock" - the book may be thin at entry but recover.
                grace_min = float(_cfg_flag("LIQ_EXIT_GRACE_PERIOD_MIN", 10, self.config))
                if self.entry_time is not None:
                    elapsed_min = (current_time - self.entry_time).total_seconds() / 60.0
                    if elapsed_min < grace_min:
                        log_debug(
                            f"Skipping liq_exit: {grace_min:.0f}min grace period "
                            f"(elapsed={elapsed_min:.1f}min)"
                        )
                        liq_reason = None
                if liq_reason is not None:
                    return liq_reason

        return None

    def _sync_trailing_orders(self, current_price: float) -> None:
        """Sync trailing SL/TP levels to the exchange when moved meaningfully.

        Calculates the strategy's trailing SL and TP, then calls the broker's
        update methods only if the new price differs from the last synced
        exchange price by at least TRAILING_SL_UPDATE_THRESHOLD (hybrid
        approach — minimises API calls while keeping protection tight).
        """
        if self.position == 0:
            return

        atr14 = self._last_atr14
        if not atr14 or atr14 <= 0:
            return

        sl_frac = self.current_meta.get("stop_loss_frac", STOP_LOSS_FRAC_DEFAULT)
        tp_frac = self.current_meta.get("take_profit_frac", TAKE_PROFIT_FRAC_DEFAULT)
        atr_mult = float(_cfg_flag("TRAILING_ATR_MULT", 1.2, self.config))
        be_mult = float(_cfg_flag("TRAILING_BREAKEVEN_MULT", 1.0, self.config))
        
        # Adaptive or static trailing SL update threshold
        atr_pct = atr14 / current_price if current_price > 0 else 0
        if _cfg_flag("ADAPTIVE_TRAIL_UPDATE_ENABLED", True, self.config):
            # Add ATR to history
            self.atr_history.append(atr14)
            lookback = int(_cfg_flag("ADAPTIVE_TRAIL_UPDATE_LOOKBACK", 50, self.config))
            if len(self.atr_history) > lookback:
                self.atr_history = self.atr_history[-lookback:]
            # adaptive_trailing_threshold returns threshold as price fraction
            # Now accepts price parameter to convert ATR to fraction
            threshold = adaptive_trailing_threshold(
                pd.Series(self.atr_history),
                current_price,
                lookback=lookback,
                quantile=float(_cfg_flag("ADAPTIVE_TRAIL_UPDATE_QUANTILE", 0.5, self.config)),
                mult=float(_cfg_flag("ADAPTIVE_TRAIL_UPDATE_MULT", 0.1, self.config)),
                min_threshold=float(_cfg_flag("ADAPTIVE_TRAIL_UPDATE_MIN", 0.001, self.config)),
                max_threshold=float(_cfg_flag("ADAPTIVE_TRAIL_UPDATE_MAX", 0.01, self.config)),
            )
            log_debug(f"[ADAPTIVE TRAIL THRESHOLD] ATR={atr14:.2f} ({atr_pct:.4%}) -> threshold={threshold:.4%} (lookback={lookback})")
        else:
            threshold = float(_cfg_flag("TRAILING_SL_UPDATE_THRESHOLD", 0.002, self.config))

        if self.trail_extreme <= 0:
            self.trail_extreme = self.entry_price

        # Check for R-multiple partial exits (profit taking)
        self._check_partial_exits(current_price)

        # Calculate trailing SL — always compute trailing level, but only use it
        # if it improves over the initial bracket SL. This avoids the "breakeven
        # threshold" problem where SL stays at initial level until 1R profit.
        if self.position > 0:  # Long
            self.trail_extreme = max(self.trail_extreme, current_price)
            base_sl = self.entry_price * (1 - sl_frac)          # Initial bracket SL
            trail_sl = self.trail_extreme - atr_mult * atr14    # Trailing SL from peak
            new_sl = max(base_sl, trail_sl)                     # Use the better (higher) SL

            # TP: Use initial bracket TP as wide safety net (not trailed)
            # Profit taking is handled by partial exits at 1R/2R/3R
            new_tp = self.entry_price * (1 + tp_frac)

        else:  # Short
            self.trail_extreme = min(self.trail_extreme, current_price)
            base_sl = self.entry_price * (1 + sl_frac)          # Initial bracket SL
            trail_sl = self.trail_extreme + atr_mult * atr14    # Trailing SL from trough
            new_sl = min(base_sl, trail_sl)                     # Use the better (lower) SL

            # TP: Use initial bracket TP as wide safety net
            new_tp = self.entry_price * (1 - tp_frac)

        # Log significant loss detection before syncing
        if self._last_synced_sl_price is not None:
            loss_pct = abs(current_price - self._last_synced_sl_price) / self._last_synced_sl_price
            if loss_pct > 0.03:  # 3% threshold for significant loss warning
                logger.warning(
                    f"[SIGNIFICANT LOSS DETECTED] Symbol: {self.broker.symbol}, "
                    f"Entry: {self.entry_price}, Current: {current_price}, "
                    f"Last Synced SL: {self._last_synced_sl_price}, "
                    f"Loss from SL: {loss_pct:.4%}, "
                    f"Position: {self.position}, Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )

        # Compute initial bracket SL for comparison (trailing only improves on this)
        base_sl = self.entry_price * (1 - sl_frac) if self.position > 0 else self.entry_price * (1 + sl_frac)

        # Only attempt SL update if trailing has actually improved the SL beyond initial bracket level
        # For long: new_sl > base_sl (trailing moved SL up)
        # For short: new_sl < base_sl (trailing moved SL down)
        sl_improved = (self.position > 0 and new_sl > base_sl) or (self.position < 0 and new_sl < base_sl)

        # Sync SL to exchange if trailing improved it and moved meaningfully from last synced
        if sl_improved and (self._last_synced_sl_price is None or abs(new_sl - self._last_synced_sl_price) / self._last_synced_sl_price >= threshold):
            now = time.time()
            if now - self._last_sl_update_ts >= 60:  # 1-min cooldown between updates
                success = self.broker.update_sl_order(self.broker.symbol, new_sl, threshold)
                if success:
                    self._last_synced_sl_price = new_sl
                    self._last_sl_update_ts = now
                    logger.info(
                        f"[SL MOVED] Symbol: {self.broker.symbol}, "
                        f"New SL: {new_sl:.6f}, "
                        f"Previous SL: {self._last_synced_sl_price or 'N/A'}, "
                        f"Change: {abs(new_sl - (self._last_synced_sl_price or new_sl)) / new_sl:.4%}, "
                        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                else:
                    # On testnet, False often means "threshold not met" — not an error
                    if is_testnet_env():
                        logger.debug(
                            f"[SL SIMULATED - NO CHANGE] Symbol: {self.broker.symbol}, "
                            f"Target SL: {new_sl:.6f}, Last Synced: {self._last_synced_sl_price}, "
                            f"Threshold not met or directional guard failed (testnet)"
                        )
                    else:
                        logger.error(
                            f"[SL ORDER FAILED TO BE PLACED] Symbol: {self.broker.symbol}, "
                            f"Target SL: {new_sl:.6f}, "
                            f"Entry: {self.entry_price}, Current Price: {current_price}, "
                            f"Position: {self.position}, ATR: {atr14:.6f}, "
                            f"Trail Extreme: {self.trail_extreme:.6f}, "
                            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
            else:
                logger.debug(
                    f"[SL UPDATE SKIPPED - COOLDOWN] Symbol: {self.broker.symbol}, "
                    f"New SL: {new_sl:.6f}, "
                    f"Last Update: {now - self._last_sl_update_ts:.1f}s ago, "
                    f"Cooldown: 60s"
                )
        elif not sl_improved:
            logger.debug(
                f"[SL TRAILING NOT ACTIVE] Symbol: {self.broker.symbol}, "
                f"Trailing SL: {new_sl:.6f} not better than initial SL: {base_sl:.6f}, "
                f"Trail Extreme: {self.trail_extreme:.6f}, ATR: {atr14:.6f}"
            )
        else:
            logger.debug(
                f"[SL NO CHANGE NEEDED] Symbol: {self.broker.symbol}, "
                f"New SL: {new_sl:.6f}, Last Synced: {self._last_synced_sl_price:.6f}, "
                f"Change: {abs(new_sl - self._last_synced_sl_price) / self._last_synced_sl_price:.4%}, "
                f"Threshold: {threshold:.4%}"
            )

        # Sync TP to exchange (initial bracket TP only - profit taking via partial exits)
        # Only sync once on entry; TP is a wide safety net, not trailed
        if self._last_synced_tp_price is None:
            base_tp = self.entry_price * (1 + tp_frac) if self.position > 0 else self.entry_price * (1 - tp_frac)
            new_tp = base_tp
            success = self.broker.update_tp_order(self.broker.symbol, new_tp, threshold)
            if success:
                self._last_synced_tp_price = new_tp
                logger.info(
                    f"[TP SET] Symbol: {self.broker.symbol}, "
                    f"Initial TP: {new_tp:.6f}, "
                    f"Entry: {self.entry_price:.2f}, "
                    f"Distance: {abs(new_tp - self.entry_price) / self.entry_price:.2%}, "
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                if is_testnet_env():
                    logger.debug(
                        f"[TP SIMULATED] Symbol: {self.broker.symbol}, "
                        f"Initial TP: {new_tp:.6f} (testnet simulation)"
                    )
                else:
                    logger.error(
                        f"[TP ORDER FAILED] Symbol: {self.broker.symbol}, "
                        f"Target TP: {new_tp:.6f}, "
                        f"Entry: {self.entry_price:.2f}, Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )

    def _check_partial_exits(self, current_price: float) -> bool:
        """
        Check and execute partial exits at R-multiple levels.
        
        Profit-taking schedule (configurable):
        - 1R: Close R_PARTIAL_FRACTIONS[0] (default 33%)
        - 2R: Close R_PARTIAL_FRACTIONS[1] (default 33%)  
        - 3R: Close R_PARTIAL_FRACTIONS[2] (default 34%)
        - Remaining: Let trailing SL protect
        
        Returns True if a partial exit was executed.
        """
        if not _cfg_flag("R_PARTIAL_EXITS_ENABLED", True, self.config):
            return False
            
        max_levels = int(_cfg_flag("R_PARTIAL_LEVELS", 3, self.config))
        fractions = _cfg_flag("R_PARTIAL_FRACTIONS", [0.33, 0.33, 0.34], self.config)
        
        if self.position == 0 or self._partial_exits_hit >= max_levels:
            return False

        sl_frac = self.current_meta.get("stop_loss_frac", STOP_LOSS_FRAC_DEFAULT)
        initial_risk = self.entry_price * sl_frac
        current_gain = (current_price - self.entry_price) if self.position > 0 else (self.entry_price - current_price)

        # Check if we've hit the next R-level
        target_gain = self._next_partial_r_level * initial_risk
        if current_gain >= target_gain:
            # Execute partial exit
            fraction = fractions[self._partial_exits_hit] if self._partial_exits_hit < len(fractions) else fractions[-1]
            fill_price = self.broker.close_position_fraction(fraction=fraction)
            
            if fill_price is not None:
                self._partial_exits_hit += 1
                self._next_partial_r_level += 1
                
                remaining = 1.0 - sum(fractions[:self._partial_exits_hit])
                logger.info(
                    f"[PARTIAL EXIT {self._partial_exits_hit}/{max_levels} @ {self._partial_exits_hit}R] "
                    f"Symbol: {self.broker.symbol}, "
                    f"Side: {'LONG' if self.position > 0 else 'SHORT'}, "
                    f"Fraction: {fraction:.0%}, Fill: {fill_price:.2f}, "
                    f"Entry: {self.entry_price:.2f}, Gain: {current_gain/initial_risk:.2f}R, "
                    f"Remaining position: {remaining:.0%}"
                )
                return True
        
        return False

    def _execute_signal(self, signal: str, current_price: float, current_time: datetime, pred: float = 0.0, threshold: float = 0.0) -> None:
        """Execute trading signal."""
        # Exit if reversal
        if signal == "long" and self.position < 0:
            self._exit_position("reversal", current_price, current_time)
        elif signal == "short" and self.position > 0:
            self._exit_position("reversal", current_price, current_time)

        # Enter if flat
        if self.position == 0 and signal in ("long", "short"):
            self._enter_position(signal, current_price, current_time, pred=pred, threshold=threshold)

    def _enter_position(self, side: str, price: float, timestamp: datetime, pred: float = 0.0, threshold: float = 0.0) -> None:
        """Open new position with optional confidence-weighted stake scaling."""
        base_stake_frac = (
            self.current_meta["stake_long_frac"]
            if side == "long"
            else self.current_meta["stake_short_frac"]
        )
        leverage = self.current_meta["recommended_leverage"]

        # Confidence-weighted scaling (Improvement 1)
        if threshold > 0 and pred != 0:
            stake_frac = self._confidence_scaled_stake(base_stake_frac, pred, threshold)
        else:
            stake_frac = base_stake_frac

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

        # Initialize trailing sync state from bracket result
        data = getattr(result, "data", {}) or {}
        self._last_synced_sl_price = data.get("sl_price")
        self._last_synced_tp_price = data.get("tp_price")

        # Fallback: calculate from entry_price and meta if not provided by broker
        sl_frac = self.current_meta.get("stop_loss_frac", STOP_LOSS_FRAC_DEFAULT)
        tp_frac = self.current_meta.get("take_profit_frac", TAKE_PROFIT_FRAC_DEFAULT)
        entry_price = self.entry_price  # Will be set from actual_pos below, but use price for now
        if self._last_synced_sl_price is None:
            if side == "long":
                self._last_synced_sl_price = round(price * (1 - sl_frac), 2)
            else:
                self._last_synced_sl_price = round(price * (1 + sl_frac), 2)
        if self._last_synced_tp_price is None:
            if side == "long":
                self._last_synced_tp_price = round(price * (1 + tp_frac), 2)
            else:
                self._last_synced_tp_price = round(price * (1 - tp_frac), 2)

        # Also sync to broker for testnet simulation
        if hasattr(self.broker, "_last_synced_sl_price"):
            self.broker._last_synced_sl_price = self._last_synced_sl_price
        if hasattr(self.broker, "_last_synced_tp_price"):
            self.broker._last_synced_tp_price = self._last_synced_tp_price

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
        self._partial_exits_hit = 0
        self._next_partial_r_level = 1

        # Market snapshot at signal time
        bid, ask, mid = 0.0, 0.0, 0.0
        if hasattr(self.broker, "get_market_snapshot"):
            bid, ask, mid = self.broker.get_market_snapshot()

        # Granular costs from broker result
        data = getattr(result, "data", {}) or {}
        commission = data.get("commission", 0.0)
        slippage = data.get("slippage", 0.0)

        # Create trade record with full timing/market/cost data
        self.current_trade = log_trade_entry(
            signal_time=timestamp,
            order_submit_time=timestamp,
            exchange_ack_time=timestamp,
            fill_time=timestamp,
            symbol="BTCUSDT",
            side=side,
            bid=bid,
            ask=ask,
            mid=mid,
            requested_price=self.entry_price,
            fill_price=self.entry_price,
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

    def _scale_in_live(self, signal: str, price: float, timestamp: datetime, pred: float = 0.0, threshold: float = 0.0) -> None:
        """Phase 3: scale into an open position on a same-direction signal.

        Adds SCALE_STAKE_FRAC-sized stake (optionally confidence-scaled) via
        broker.scale_in, re-reconciles from the broker's actual fill, and re-
        places the TP/SL bracket on the enlarged position.
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

        # Scale-in quantity with optional confidence weighting
        base_qty = self._initial_qty or abs(self.current_trade.get("qty", 0.0))
        add_qty = base_qty * float(_cfg_flag("SCALE_STAKE_FRAC", 0.5, self.config))
        if threshold > 0 and pred != 0 and add_qty > 0:
            conf_factor = self._confidence_scaled_stake(1.0, pred, threshold)
            add_qty = add_qty * conf_factor
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

        # Preserve trailing progress before bracket replacement
        # Calculate current trailing SL/TP to compare with new initial bracket
        atr14 = self._last_atr14
        sl_frac = self.current_meta.get("stop_loss_frac", STOP_LOSS_FRAC_DEFAULT)
        tp_frac = self.current_meta.get("take_profit_frac", TAKE_PROFIT_FRAC_DEFAULT)
        atr_mult = float(_cfg_flag("TRAILING_ATR_MULT", 1.2, self.config))
        
        preserved_sl = None
        preserved_tp = None
        if atr14 and atr14 > 0 and self.trail_extreme > 0:
            if self.position > 0:  # Long
                trail_sl = self.trail_extreme - atr_mult * atr14
                initial_risk = self.entry_price * sl_frac
                current_gain = self.trail_extreme - self.entry_price
                if current_gain >= 2 * initial_risk:
                    tp_trail = self.entry_price + current_gain * 0.5
                    trail_tp = max(self.entry_price * (1 + tp_frac), tp_trail)
                else:
                    trail_tp = self.entry_price * (1 + tp_frac)
            else:  # Short
                trail_sl = self.trail_extreme + atr_mult * atr14
                initial_risk = self.entry_price * sl_frac
                current_gain = self.entry_price - self.trail_extreme
                if current_gain >= 2 * initial_risk:
                    tp_trail = self.entry_price - current_gain * 0.5
                    trail_tp = min(self.entry_price * (1 - tp_frac), tp_trail)
                else:
                    trail_tp = self.entry_price * (1 - tp_frac)
            preserved_sl = trail_sl
            preserved_tp = trail_tp

        # Re-place TP/SL bracket on the enlarged position
        try:
            self.broker.replace_bracket_order(
                self.broker.symbol,
                self._initial_qty,
                broker_side,
                tp_frac=tp_frac,
                sl_frac=sl_frac,
            )
            # Sync trailing state after bracket replacement
            # Use the better of (new initial bracket, preserved trailing) to avoid losing progress
            pos = self.broker.get_position()
            if pos and pos.entry_price:
                entry_price = pos.entry_price
                if self.position > 0:
                    initial_tp = round(entry_price * (1 + tp_frac), 2)
                    initial_sl = round(entry_price * (1 - sl_frac), 2)
                    # Use better (higher) SL and (higher) TP
                    self._last_synced_tp_price = max(initial_tp, round(preserved_tp, 2)) if preserved_tp else initial_tp
                    self._last_synced_sl_price = max(initial_sl, round(preserved_sl, 2)) if preserved_sl else initial_sl
                else:
                    initial_tp = round(entry_price * (1 - tp_frac), 2)
                    initial_sl = round(entry_price * (1 + sl_frac), 2)
                    # Use better (lower) SL and (lower) TP
                    self._last_synced_tp_price = min(initial_tp, round(preserved_tp, 2)) if preserved_tp else initial_tp
                    self._last_synced_sl_price = min(initial_sl, round(preserved_sl, 2)) if preserved_sl else initial_sl
                # Also sync to broker
                if hasattr(self.broker, "_last_synced_sl_price"):
                    self.broker._last_synced_sl_price = self._last_synced_sl_price
                if hasattr(self.broker, "_last_synced_tp_price"):
                    self.broker._last_synced_tp_price = self._last_synced_tp_price
                log_info(
                    f"[SCALE-IN BRACKET] Symbol: {self.broker.symbol}, "
                    f"Entry: {entry_price:.2f}, Initial SL: {initial_sl:.2f}, "
                    f"Trailing SL: {preserved_sl:.2f if preserved_sl else 'N/A'}, "
                    f"Synced SL: {self._last_synced_sl_price:.2f}, "
                    f"Initial TP: {initial_tp:.2f}, Trailing TP: {preserved_tp:.2f if preserved_tp else 'N/A'}, "
                    f"Synced TP: {self._last_synced_tp_price:.2f}"
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
        close_result = self.broker.close_position()
        actual_fill = close_result.fill_price if close_result else (price if close_result is None else 0.0)

        if self.current_trade:
            entry_price = self.entry_price or actual_fill
            notional = abs(self.current_trade.get("qty", 0.0)) or 1.0
            raw_pnl = (
                (actual_fill - entry_price) * notional
                if self.position > 0
                else (entry_price - actual_fill) * notional
            )
            equity_before = self.current_trade.get("equity_before") or 1.0

            # Market snapshot at exit time
            bid, ask, mid = 0.0, 0.0, 0.0
            if hasattr(self.broker, "get_market_snapshot"):
                bid, ask, mid = self.broker.get_market_snapshot()

            # Granular costs from broker result
            spread = close_result.spread if close_result else 0.0
            slippage = close_result.slippage if close_result else 0.0
            commission = close_result.commission if close_result else 0.0
            financing = close_result.financing if close_result else 0.0

            # Average expected trade profit: use probability-weighted expectation
            # based on model prediction magnitude and historical win rate for the regime
            take_profit_frac = self.current_meta.get("take_profit_frac", 0.02)
            stop_loss_frac = self.current_meta.get("stop_loss_frac", 0.02)
            regime = self.current_meta.get("regime", "trend")
            
            # Heuristic win rate by regime (could be enhanced with historical tracking)
            regime_win_rates = {
                "trend": 0.45,
                "mean_reversion": 0.55,
                "chop": 0.40,
                "unknown": 0.42,
            }
            win_rate = regime_win_rates.get(regime, 0.42)
            
            # Model prediction confidence scales the win rate
            pred_confidence = min(abs(self._last_tactical_pred or 0.0) / 0.02, 1.0)  # normalize to ~2%
            win_rate_adj = win_rate * (0.8 + 0.4 * pred_confidence)  # 80%-120% of base
            
            # Probability-weighted expected profit per trade
            avg_win = take_profit_frac
            avg_loss = stop_loss_frac
            expected_profit_frac = win_rate_adj * avg_win - (1 - win_rate_adj) * avg_loss
            
            # Floor at 0 (don't log negative expectation for risk management)
            expected_profit_frac = max(expected_profit_frac, 0.0)
            avg_expected_profit = equity_before * expected_profit_frac

            log_trade_exit(
                trade=self.current_trade,
                exit_price=actual_fill,
                exit_reason=reason,
                pnl=raw_pnl,
                pnl_pct=raw_pnl / equity_before if equity_before else 0.0,
                equity_after=self.broker.get_equity(),
                spread=spread,
                slippage=slippage,
                commission=commission,
                financing=financing,
                avg_expected_trade_profit=avg_expected_profit,
            )

            # Log cost and edge summaries to CSV + logger
            trade_id = self.current_trade.get("fill_time", "")[:19].replace("-", "").replace(":", "").replace("T", "") or "0"
            effective_cost = spread + slippage + commission + financing
            log_cost_summary(
                trade_id=trade_id,
                side=side,
                spread_cost=spread,
                commission=commission,
                slippage=slippage,
                financing=financing,
                total_cost=effective_cost,
            )
            notional = abs(self.current_trade.get("qty", 0.0) or 1.0) * self.entry_price
            log_edge_summary(
                trade_id=trade_id,
                pred=self._last_tactical_pred,
                notional=notional,
                expected_profit=avg_expected_profit,
                total_cost=effective_cost,
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
        self._partial_exits_hit = 0
        self._next_partial_r_level = 1

        # Reset hybrid sync state
        self._last_synced_sl_price = None
        self._last_synced_tp_price = None
        self._last_sl_update_ts = 0.0


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