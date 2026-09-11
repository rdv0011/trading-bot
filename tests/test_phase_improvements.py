"""Tests for Phase 1-3 performance / profitability improvements:

  1. Phase 1: feature-frame TTL cache (strategy._cached_features)
  2. Phase 2: adaptive signal threshold (sim + strategy)
  3. Phase 2: trailing stop / extreme-volatility gate (sim + strategy)
  4. Phase 3: cooldown, scaling, partial exit, reversal full-close (sim + strategy)

simulate.py / strategy.py live at the repo root and use bare sibling imports.
Heavy imports stay inside fixtures/test bodies so this file never perturbs
sibling tests in the same pytest process (test_simplified_simulate.py imports
simulate at module level with its own mock_config). Feature flags are passed
as concrete SimpleNamespace values, never MagicMock (unset attrs would auto-
create truthy MagicMocks and silently enable features).
"""

import sys
import types
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

repo_root = Path(__file__).resolve().parent.parent


# ── Synthetic data builders (no heavy imports) ─────────────────────────
def _make_ohlcv(periods=300, freq="5min", seed=42):
    idx = pd.date_range("2024-01-01", periods=periods, freq=freq)
    rng = np.random.default_rng(seed)
    close = 40000 + np.cumsum(rng.normal(0, 50, periods))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(1, 10, periods),
        },
        index=idx,
    )


def _make_featured_df(periods=300, seed=42):
    """Featured frame with atr14 / vol_12 / vol_48 / regime for feature tests."""
    df = _make_ohlcv(periods, seed=seed)
    df["ret1"] = df["close"].pct_change(1)
    df["tr"] = df["high"] - df["low"]
    df["atr14"] = df["tr"].rolling(14).mean()
    df["vol_12"] = df["ret1"].rolling(12).std()
    df["vol_48"] = df["ret1"].rolling(48).std()
    df["regime"] = "trend"
    return df.dropna()


def _make_flat_df(periods=300, freq="5min"):
    """Flat-price frame: no TP/SL/ATR interference for scaling/cooldown tests."""
    idx = pd.date_range("2024-01-01", periods=periods, freq=freq)
    close = np.full(periods, 40000.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 20.0,
            "low": close - 20.0,
            "close": close,
            "volume": 5.0,
            "ret1": 0.0,
            "atr14": 0.0,
            "vol_12": 0.001,
            "vol_48": 0.001,
            "regime": "trend",
        },
        index=idx,
    ).round(5)


# ── Feature configs: concrete values only (NO MagicMock) ───────────────
# strategy/data/model/broker bind `from config import ...` at first import;
# when strategy is imported fresh inside patch.dict({"config": cfg}), every
# imported name must exist on cfg or the import dies. Keep this surface complete.
def _base_cfg(**overrides):
    """Complete config surface; concrete values only so fresh imports bind real numbers."""
    cfg = types.SimpleNamespace(
        SYMBOL="BTCUSDT",
        TACTICAL_TF="15m",
        STRATEGIC_TF="1h",
        HISTORY_DAYS=50,
        TRAIN_FRACTION=0.8,
        LABEL_HORIZON=4,
        FEATURE_LAGS=[1, 2, 3, 5, 10, 20, 50],
        EMA_SPANS=[5, 10, 20, 50, 100],
        ATR_PERIOD=14,
        MODEL_DIR="models",
        STRATEGIC_TARGET_COLS=[
            "recommended_leverage", "max_exposure_frac",
            "stake_long_frac", "stake_short_frac",
            "stop_loss_frac", "take_profit_frac", "max_hold_hours",
        ],
        REGIME_LEVERAGE={"trend": 5.0, "high_vol": 2.0, "chop": 1.0},
        REGIME_STAKE_LONG={"trend": 0.2, "high_vol": 0.1, "chop": 0.1},
        REGIME_STAKE_SHORT={"trend": 0.1, "high_vol": 0.05, "chop": 0.05},
        REGIME_STOP_LOSS={"trend": 0.015, "high_vol": 0.03, "chop": 0.02},
        TAKE_PROFIT_MULT=2.0,
        REGIME_MAX_HOLD={"trend": 8.0, "high_vol": 2.0, "chop": 4.0},
        STAKE_LONG_FRAC_DEFAULT=0.10,
        STAKE_SHORT_FRAC_DEFAULT=0.05,
        STOP_LOSS_FRAC_DEFAULT=0.02,
        TAKE_PROFIT_FRAC_DEFAULT=0.04,
        MAX_HOLD_HOURS_DEFAULT=4.0,
        LEVERAGE_DEFAULT=1.0,
        INITIAL_EQUITY=1.0,
        FEE=0.0004,
        SLIPPAGE=0.0003,
        WALKFORWARD_RETRAIN_EVERY=100,
        MODEL_REFRESH_EVERY=50,
        TACTICAL_MODEL_PARAMS={"iterations": 10, "depth": 4, "learning_rate": 0.03},
        STRATEGIC_MODEL_PARAMS={"iterations": 10, "depth": 4, "learning_rate": 0.03},
        ABSOLUTE_THRESHOLD=0.006,
        FEATURE_CACHE_TTL=60,
        ADAPTIVE_THRESHOLD_ENABLED=True,
        ADAPTIVE_LOOKBACK=200,
        ADAPTIVE_QUANTILE=0.95,
        ADAPTIVE_MIN_THRESHOLD=0.002,
        ADAPTIVE_MAX_THRESHOLD=0.02,
        TRAILING_STOP_ENABLED=True,
        TRAILING_ATR_MULT=1.5,
        TRAILING_BREAKEVEN_MULT=1.0,
        GATE_EXTREME_VOL=True,
        EXTREME_VOL_RATIO=1.8,
        SCALING_ENABLED=True,
        MAX_SCALE_COUNT=2,
        SCALE_CONFIRM_BARS=2,
        SCALE_STAKE_FRAC=0.5,
        PARTIAL_EXIT_ENABLED=True,
        PARTIAL_EXIT_FRACTION=0.33,
        REVERSAL_FULL_CLOSE_STREAK=2,
        TRADE_COOLDOWN_ENABLED=True,
        TRADE_COOLDOWN_MINUTES=30,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _enabled_cfg(**overrides):
    """All Phase 1-3 features ENABLED with production-like parameters."""
    cfg = _base_cfg()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _legacy_cfg(**overrides):
    """All Phase 1-3 features DISABLED (mirrors the mock_config in the legacy tests)."""
    cfg = _base_cfg(
        ADAPTIVE_THRESHOLD_ENABLED=False,
        TRAILING_STOP_ENABLED=False,
        GATE_EXTREME_VOL=False,
        SCALING_ENABLED=False,
        PARTIAL_EXIT_ENABLED=False,
        TRADE_COOLDOWN_ENABLED=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def sim_module():
    """Import simulate lazily (module-level import would rebind _config and
    break test_simplified_simulate.py which runs later in this process)."""
    import simulate as sim_mod
    return sim_mod


# ── Phase 2: Trailing stop (sim) ───────────────────────────────────────
def test_sim_trailing_stop_exits_on_retrace(sim_module, monkeypatch):
    cfg = _enabled_cfg()
    monkeypatch.setattr(sim_module, "_config", cfg)
    df = _make_featured_df()
    broker = sim_module.MockBroker(df=df)
    broker.position = 1.0
    broker.entry_price = 40000.0
    broker.entry_time = df.index[99]
    broker.current_meta["stop_loss_frac"] = 0.02
    broker.current_meta["take_profit_frac"] = 0.04
    broker.current_meta["max_hold_hours"] = 4.0
    # Price already rallied above breakeven (entry * 1.02) before this check
    broker.trail_extreme = 41200.0
    atr14 = float(df.iloc[100]["atr14"])
    assert atr14 > 0
    trail_sl = 41200.0 - 1.5 * atr14
    assert trail_sl > 40000.0 * 0.98  # effector above base SL

    # Retrace below the trailing SL -> exit
    reason = broker._check_exit_conditions(100, trail_sl - 1.0)
    assert reason == "trailing_sl"


def test_sim_trailing_stop_inactive_below_breakeven(sim_module, monkeypatch):
    cfg = _enabled_cfg()
    monkeypatch.setattr(sim_module, "_config", cfg)
    df = _make_featured_df()
    broker = sim_module.MockBroker(df=df)
    broker.position = 1.0
    broker.entry_price = 40000.0
    broker.entry_time = df.index[99]
    broker.current_meta["stop_loss_frac"] = 0.02
    broker.current_meta["take_profit_frac"] = 0.04
    broker.current_meta["max_hold_hours"] = 4.0
    # Never reached breakeven: trailing must NOT fire
    broker.trail_extreme = 40500.0  # below breakeven 40800
    assert broker._check_exit_conditions(100, 40600.0) is None


def test_sim_trailing_stop_disabled_returns_none(sim_module, monkeypatch):
    cfg = _legacy_cfg()
    monkeypatch.setattr(sim_module, "_config", cfg)
    df = _make_featured_df()
    broker = sim_module.MockBroker(df=df)
    broker.position = 1.0
    broker.entry_price = 40000.0
    broker.entry_time = df.index[99]
    broker.current_meta["stop_loss_frac"] = 0.02
    broker.current_meta["take_profit_frac"] = 0.04
    broker.current_meta["max_hold_hours"] = 4.0
    broker.trail_extreme = 41200.0
    # Even in a retrace scenario the flag-off path must not produce trailing_sl
    assert broker._check_exit_conditions(100, 40500.0) is None


# ── Phase 2: Adaptive threshold (data helper) ───────────────────────────
def test_adaptive_threshold_warmup_falls_back(sim_module):
    from data import adaptive_threshold
    # <20 valid predictions -> fallback threshold
    preds = pd.Series([0.01] * 10 + [np.nan] * 5)
    assert adaptive_threshold(preds, fallback=0.006) == 0.006


def test_adaptive_threshold_percentile_clamped(sim_module):
    from data import adaptive_threshold
    preds = pd.Series([0.01] * 200)
    # 95th percentile of |0.01| == 0.01, within [0.002, 0.02]
    assert adaptive_threshold(preds, lookback=200, quantile=0.95) == pytest.approx(0.01)
    # Below min clamps up
    assert adaptive_threshold(pd.Series([0.0001] * 100), min_threshold=0.002, max_threshold=0.02) == 0.002
    # Above max clamps down
    assert adaptive_threshold(pd.Series([0.5] * 100), min_threshold=0.002, max_threshold=0.02) == 0.02


def test_classify_vol_state_normal_and_extreme(sim_module):
    from data import classify_vol_state
    assert classify_vol_state(1.0) == "normal"
    assert classify_vol_state(1.8) == "extreme"
    assert classify_vol_state(2.5) == "extreme"
    assert classify_vol_state(float("nan")) == "normal"
    assert classify_vol_state(float("inf")) == "normal"


# ── Phase 2: Adaptive threshold via run_simulation ──────────────────────
def test_run_adaptive_threshold_blocks_weak_signals(sim_module, monkeypatch):
    df = _make_featured_df()
    preds = pd.Series([0.01] * len(df), index=df.index, name="tactical_prediction")
    metas = [{}] * len(df)

    cfg_adaptive = _enabled_cfg(GATE_EXTREME_VOL=False, TRADE_COOLDOWN_ENABLED=False)
    monkeypatch.setattr(sim_module, "_config", cfg_adaptive)
    _, metrics_adaptive, _ = sim_module.run_simulation(df, preds, metas, config=cfg_adaptive)

    cfg_static = _legacy_cfg(GATE_EXTREME_VOL=False)
    monkeypatch.setattr(sim_module, "_config", cfg_static)
    _, metrics_static, _ = sim_module.run_simulation(df, preds, metas, config=cfg_static)

    assert metrics_static["num_trades"] > 0
    assert metrics_adaptive["num_trades"] < metrics_static["num_trades"]


def test_run_static_threshold_trades(sim_module, monkeypatch):
    # Same constant preds but adaptive disabled: static 0.006 threshold -> trades.
    cfg = _legacy_cfg(GATE_EXTREME_VOL=False)
    monkeypatch.setattr(sim_module, "_config", cfg)
    df = _make_featured_df()
    preds = pd.Series([0.01] * len(df), index=df.index, name="tactical_prediction")
    metas = [{}] * len(df)
    _, metrics, _ = sim_module.run_simulation(df, preds, metas, config=cfg)
    assert metrics["num_trades"] > 0


# ── Phase 2: Extreme-volatility gate via run_simulation ─────────────────
def test_run_extreme_vol_gate_blocks_entries(sim_module, monkeypatch):
    # vol_12/vol_48 ratio forced to 2.0 (>= 1.8) everywhere -> all entries blocked.
    df = _make_featured_df()
    df["vol_12"] = 0.02
    df["vol_48"] = 0.01
    preds = pd.Series([0.01] * len(df), index=df.index, name="tactical_prediction")
    metas = [{}] * len(df)

    cfg_on = _legacy_cfg(GATE_EXTREME_VOL=True, EXTREME_VOL_RATIO=1.8)
    monkeypatch.setattr(sim_module, "_config", cfg_on)
    _, metrics_on, _ = sim_module.run_simulation(df, preds, metas, config=cfg_on)
    assert metrics_on["num_trades"] == 0

    cfg_off = _legacy_cfg(GATE_EXTREME_VOL=False)
    monkeypatch.setattr(sim_module, "_config", cfg_off)
    _, metrics_off, _ = sim_module.run_simulation(df, preds, metas, config=cfg_off)
    assert metrics_off["num_trades"] > 0


# ── Phase 3: Scaling (sim) ──────────────────────────────────────────────
def test_sim_scale_in_grows_and_caps(sim_module, monkeypatch):
    cfg = _enabled_cfg(
        SCALING_ENABLED=True, MAX_SCALE_COUNT=2, SCALE_CONFIRM_BARS=2,
        SCALE_STAKE_FRAC=0.5, TRADE_COOLDOWN_ENABLED=False,
    )
    monkeypatch.setattr(sim_module, "_config", cfg)
    df = _make_flat_df()
    broker = sim_module.MockBroker(df=df)

    # Entry
    step0 = broker.step(0, "long")
    assert broker.position > 0
    base_qty = abs(broker.position)

    # Same-direction signals build a streak; after SCALE_CONFIRM_BARS the
    # position grows by SCALE_STAKE_FRAC of the base qty (weighted-avg entry).
    step1 = broker.step(1, "long")  # streak=1, not yet confirmed
    assert abs(broker.position) == pytest.approx(base_qty)
    broker.step(2, "long")          # streak=2 -> scale #1
    assert abs(broker.position) == pytest.approx(base_qty * 1.5)
    assert broker.scale_count == 1

    broker.step(3, "long")          # streak=1
    broker.step(4, "long")          # streak=2 -> scale #2 (max reached)
    # Compounding like live: base_qty rebases to the grower position each scale,
    # and the scaled qty becomes the new scale base (parity with _scale_in_live).
    assert abs(broker.position) == pytest.approx(base_qty * 2.25)
    assert broker.scale_count == 2

    broker.step(5, "long")          # streak=1
    broker.step(6, "long")          # streak=2 but scale_count already == MAX
    assert abs(broker.position) == pytest.approx(base_qty * 2.25)
    assert broker.scale_count == 2


def test_sim_scale_in_disabled_never_scales(sim_module, monkeypatch):
    cfg = _legacy_cfg()
    monkeypatch.setattr(sim_module, "_config", cfg)
    df = _make_flat_df()
    broker = sim_module.MockBroker(df=df)
    broker.step(0, "long")
    base_qty = abs(broker.position)
    for i in range(1, 6):
        broker.step(i, "long")
    assert abs(broker.position) == pytest.approx(base_qty)
    assert broker.scale_count == 0


# ── Phase 3: Partial exit / reversal (sim) ──────────────────────────────
def test_sim_partial_exit_on_first_reversal(sim_module, monkeypatch):
    cfg = _enabled_cfg(
        PARTIAL_EXIT_ENABLED=True, PARTIAL_EXIT_FRACTION=0.33,
        REVERSAL_FULL_CLOSE_STREAK=2, TRADE_COOLDOWN_ENABLED=False,
    )
    monkeypatch.setattr(sim_module, "_config", cfg)
    df = _make_flat_df()
    broker = sim_module.MockBroker(df=df)
    broker.step(0, "long")
    base_qty = abs(broker.position)

    broker.step(1, "short")  # first reversal -> partial close 33%
    assert broker.position > 0                      # still long
    assert abs(broker.position) == pytest.approx(base_qty * 0.67)
    assert broker.reversal_streak == 1
    assert len(broker.trades) == 0                  # not fully closed yet
    assert abs(broker.current_trade["qty"]) == pytest.approx(base_qty * 0.67)


def test_sim_full_close_flip_on_persistent_reversal(sim_module, monkeypatch):
    cfg = _enabled_cfg(
        PARTIAL_EXIT_ENABLED=True, PARTIAL_EXIT_FRACTION=0.33,
        REVERSAL_FULL_CLOSE_STREAK=2, TRADE_COOLDOWN_ENABLED=False,
    )
    monkeypatch.setattr(sim_module, "_config", cfg)
    df = _make_flat_df()
    broker = sim_module.MockBroker(df=df)
    broker.step(0, "long")

    broker.step(1, "short")  # partial
    broker.step(2, "short")  # persistent reversal -> full close + flip
    assert broker.position < 0                      # flipped to short
    assert broker.reversal_streak == 0
    assert len(broker.trades) == 1                  # long fully closed & archived
    assert broker.current_trade is not None         # short position opened
    assert abs(broker.current_trade["qty"]) > 0


def test_sim_legacy_reversal_closes_and_flips_immediately(sim_module, monkeypatch):
    # PARTIAL_EXIT_ENABLED=False -> legacy: immediate full close + flip.
    cfg = _legacy_cfg()
    monkeypatch.setattr(sim_module, "_config", cfg)
    df = _make_flat_df()
    broker = sim_module.MockBroker(df=df)
    broker.step(0, "long")
    broker.step(1, "short")
    assert broker.position < 0
    assert broker.reversal_streak == 0
    assert len(broker.trades) == 1


# ── Phase 3: Cooldown (sim) ─────────────────────────────────────────────
def test_sim_cooldown_blocks_flat_reenry_then_allows(sim_module, monkeypatch):
    cfg = _enabled_cfg(TRADE_COOLDOWN_ENABLED=True, TRADE_COOLDOWN_MINUTES=30)
    monkeypatch.setattr(sim_module, "_config", cfg)
    df = _make_flat_df()
    broker = sim_module.MockBroker(df=df)

    broker.step(0, "long")       # entry at t0 -> last_entry_time = t0
    broker._execute_exit(1, "sl", float(df.iloc[1]["close"]))
    assert broker.position == 0
    assert broker.last_entry_time == df.index[0]

    # 5-min candles: 2 bars later = 10 min < 30 min cooldown -> entry blocked
    result = broker.step(2, "long")
    assert broker.position == 0
    assert broker.current_trade is None
    assert result["position"] == 0

    # Jump 31 minutes into the future -> cooldown expired -> entry allowed
    broker.last_entry_time = df.index[0] - timedelta(minutes=31)
    broker.step(3, "long")
    assert broker.position > 0


def test_sim_cooldown_disabled_allows_immediate_reenry(sim_module, monkeypatch):
    cfg = _legacy_cfg()
    monkeypatch.setattr(sim_module, "_config", cfg)
    df = _make_flat_df()
    broker = sim_module.MockBroker(df=df)
    broker.step(0, "long")
    broker._execute_exit(1, "sl", float(df.iloc[1]["close"]))
    broker.step(2, "long")
    assert broker.position > 0


# ── Phase 1: Feature-frame TTL cache (strategy) ─────────────────────────
@pytest.fixture
def strategy_factory():
    # Import once per process: this pulls in catboost/scipy/numpy.fft and must
    # stay cached. Importing inside patch.dict(sys.modules,...) would delete
    # the whole chain from the cache on exit (numpy extension modules cannot
    # be re-imported on Python 3.14: "cannot load module more than once").
    sys.path.insert(0, str(repo_root))
    import strategy as strategy_mod

    def _make(config, **kwargs):
        broker = MagicMock()
        broker.symbol = "BTCUSDT"
        broker.get_last_price.return_value = 65000.0
        broker.close_position.return_value = 64990.0
        broker.get_equity.return_value = 1000.0

        strat = strategy_mod.DualMLStrategy(
            broker=broker,
            config=config,
            tactical_model=MagicMock(),
            strategic_model=MagicMock(),
            feature_cols=["close"],
            **kwargs,
        )
        return strategy_mod, strat, broker
    return _make


def test_strategy_feature_cache_reuses_frame(strategy_factory, monkeypatch):
    cfg = _enabled_cfg()
    strategy_mod, strat, _ = strategy_factory(cfg)

    feats_engine = MagicMock()
    feats_engine.return_value = _make_featured_df()
    monkeypatch.setattr(strategy_mod, "make_features_df", feats_engine)

    df_raw = _make_ohlcv(60)
    f1 = strat._cached_features(df_raw, "15m")
    f2 = strat._cached_features(df_raw, "15m")
    assert feats_engine.call_count == 1      # second call served from cache
    assert f1 is f2

    # A changed last candle invalidates the cache
    df_raw2 = df_raw.iloc[:-1].copy()
    df_raw2.loc[df_raw2.index[-1]] = df_raw2.iloc[-1]
    _ = strat._cached_features(df_raw2, "15m")
    assert feats_engine.call_count == 2


def test_strategy_feature_cache_expires_after_ttl(strategy_factory, monkeypatch):
    cfg = _enabled_cfg()
    strategy_mod, strat, _ = strategy_factory(cfg)

    feats_engine = MagicMock()
    feats_engine.return_value = _make_featured_df()
    monkeypatch.setattr(strategy_mod, "make_features_df", feats_engine)

    df_raw = _make_ohlcv(60)
    mock_time = MagicMock()
    mock_time.time.side_effect = [1000.0, 1000.0, 1000.0 + 61.0]
    monkeypatch.setattr(strategy_mod, "time", mock_time)

    _ = strat._cached_features(df_raw, "15m")
    _ = strat._cached_features(df_raw, "15m")   # same last_ts, within TTL -> cached
    assert feats_engine.call_count == 1
    _ = strat._cached_features(df_raw, "15m")   # TTL expired -> rebuild
    assert feats_engine.call_count == 2


# ── Phase 2: Adaptive threshold (strategy) ──────────────────────────────
def test_strategy_current_threshold_adaptive_vs_static(strategy_factory):
    cfg_adaptive = _enabled_cfg()
    _, strat_adapt, _ = strategy_factory(cfg_adaptive)
    strat_adapt.pred_history = [0.01] * 50
    assert strat_adapt._current_threshold() == pytest.approx(0.01)

    cfg_static = _legacy_cfg()
    _, strat_static, _ = strategy_factory(cfg_static)
    strat_static.pred_history = [0.01] * 50
    assert strat_static._current_threshold() == pytest.approx(strat_static.threshold)


# ── Phase 2: Trailing stop (strategy) ───────────────────────────────────
def test_strategy_trailing_stop_exits_on_retrace(strategy_factory):
    cfg = _enabled_cfg()
    _, strat, _ = strategy_factory(cfg)
    strat.position = 1.0
    strat.entry_price = 40000.0
    strat.entry_time = datetime(2024, 1, 1, 0, 0)
    strat.current_meta["stop_loss_frac"] = 0.02
    strat.current_meta["take_profit_frac"] = 0.04
    strat.current_meta["max_hold_hours"] = 4.0
    strat._last_atr14 = 160.0
    strat.trail_extreme = 41200.0  # rallied above breakeven 40800
    now = datetime(2024, 1, 1, 0, 30)
    # trail_sl = 41200 - 1.5*160 = 40960 > base_sl 39200, price below it -> exit
    assert strat._check_exits(40950.0, now) == "trailing_sl"
    # price still above trail_sl -> hold
    assert strat._check_exits(41000.0, now) is None


def test_strategy_trailing_stop_disabled(strategy_factory):
    cfg = _legacy_cfg()
    _, strat, _ = strategy_factory(cfg)
    strat.position = 1.0
    strat.entry_price = 40000.0
    strat.entry_time = datetime(2024, 1, 1, 0, 0)
    strat.current_meta["stop_loss_frac"] = 0.02
    strat.current_meta["take_profit_frac"] = 0.04
    strat.current_meta["max_hold_hours"] = 4.0
    strat._last_atr14 = 160.0
    strat.trail_extreme = 41200.0
    assert strat._check_exits(40950.0, datetime(2024, 1, 1, 0, 30)) is None


# ── Phase 3: Opposite-direction handling (strategy) ─────────────────────
def _open_trade(qty=1.0, **overrides):
    """Realistic trade dict mirroring log_trade_entry() output, so the
    log_trade_exit() debug formatter (which formats side/entry_price/pnl
    etc. directly) never hits None."""
    return {
        "timestamp": "2024-01-01T00:00:00",
        "symbol": "BTCUSDT",
        "side": "long",
        "entry_price": 40000.0,
        "exit_price": "",
        "qty": qty,
        "stake_frac": 0.1,
        "leverage": 1.0,
        "stop_loss": 0.02,
        "take_profit": 0.04,
        "max_hold_hours": 4.0,
        "regime": "trend",
        "exit_reason": "",
        "pnl": "",
        "pnl_pct": "",
        "equity_before": 1000.0,
        "equity_after": "",
        "fee_paid": "",
        "slippage_paid": "",
        "tactical_pred": 0.01,
        "strategic_params": "{}",
        **overrides,
    }


def test_strategy_opposite_live_disabled_is_noop(strategy_factory):
    cfg = _legacy_cfg()
    _, strat, broker = strategy_factory(cfg)
    strat.position = 1.0
    strat.same_dir_streak = 3
    strat.reversal_streak = 3
    strat._handle_opposite_live("short", 40000.0, datetime(2024, 1, 1))
    assert strat.position == 1.0                       # never flips live
    broker.close_position_fraction.assert_not_called()
    broker.close_position.assert_not_called()
    broker.open_position.assert_not_called()
    assert strat.same_dir_streak == 0
    assert strat.reversal_streak == 0


def test_strategy_opposite_live_partial_then_full_flip(strategy_factory):
    cfg = _enabled_cfg(
        PARTIAL_EXIT_ENABLED=True, PARTIAL_EXIT_FRACTION=0.33,
        REVERSAL_FULL_CLOSE_STREAK=2,
    )
    _, strat, broker = strategy_factory(cfg)
    strat.position = 1.0
    strat._initial_qty = 1.0
    strat.current_trade = _open_trade()
    broker.close_position_fraction.return_value = 39900.0
    broker.close_position.return_value = 39900.0
    broker.get_position.return_value = MagicMock(amount=0.67, entry_price=39900.0)
    broker.open_position.return_value = MagicMock(success=True, amount=0.67, entry_price=39900.0)

    # First reversal -> partial close (still long)
    strat._handle_opposite_live("short", 40000.0, datetime(2024, 1, 1, 0, 0))
    broker.close_position_fraction.assert_called_once_with("BTCUSDT", 0.33)
    assert strat.position == 1.0
    assert strat.reversal_streak == 1
    assert strat.current_trade["qty"] == pytest.approx(0.67)

    # Second reversal -> full close + flip to short
    strat._handle_opposite_live("short", 40000.0, datetime(2024, 1, 1, 0, 5))
    broker.close_position.assert_called_once()
    broker.open_position.assert_called_once()
    assert strat.position == -1.0
    assert strat.reversal_streak == 0


# ── Phase 3: Scale-in (strategy) ────────────────────────────────────────
def test_strategy_scale_in_live_confirms_then_scales(strategy_factory):
    cfg = _enabled_cfg(
        SCALING_ENABLED=True, MAX_SCALE_COUNT=2, SCALE_CONFIRM_BARS=2,
        SCALE_STAKE_FRAC=0.5,
    )
    _, strat, broker = strategy_factory(cfg)
    strat.position = 1.0
    strat._initial_qty = 1.0
    strat.current_trade = {"qty": 1.0}
    broker.scale_in.return_value = MagicMock(amount=1.5, error=None)
    broker.get_position.side_effect = [
        MagicMock(amount=1.5, entry_price=40000.0),
        MagicMock(amount=2.25, entry_price=40000.0),
    ]
    broker.replace_bracket_order.return_value = None
    now = datetime(2024, 1, 1, 0, 0)

    # streak=1 < SCALE_CONFIRM_BARS -> no scale yet
    strat._scale_in_live("long", 40000.0, now)
    broker.scale_in.assert_not_called()
    assert strat.scale_count == 0

    # streak=2 -> scale #1 (0.5 * 1.0 base)
    strat._scale_in_live("long", 40000.0, now)
    assert broker.scale_in.call_count == 1
    assert broker.replace_bracket_order.call_count == 1
    assert strat.scale_count == 1
    assert strat._initial_qty == pytest.approx(1.5)

    # streak=2 again -> scale #2 (0.5 * 1.5 = 0.75, new total 2.25)
    strat.same_dir_streak = 0
    strat._scale_in_live("long", 40000.0, datetime(2024, 1, 1, 0, 10))  # streak 1
    strat._scale_in_live("long", 40000.0, datetime(2024, 1, 1, 0, 15))  # streak 2 -> scale
    assert broker.scale_in.call_count == 2
    assert strat.scale_count == 2
    assert strat._initial_qty == pytest.approx(2.25)

    # MAX_SCALE_COUNT reached -> no further scaling
    strat.same_dir_streak = 0
    strat._scale_in_live("long", 40000.0, datetime(2024, 1, 1, 0, 20))
    strat.same_dir_streak = 0
    strat._scale_in_live("long", 40000.0, datetime(2024, 1, 1, 0, 25))
    assert broker.scale_in.call_count == 2


def test_strategy_scale_in_live_disabled(strategy_factory):
    cfg = _legacy_cfg()
    _, strat, broker = strategy_factory(cfg)
    strat.position = 1.0
    strat._initial_qty = 1.0
    strat.current_trade = {"qty": 1.0}
    strat._scale_in_live("long", 40000.0, datetime(2024, 1, 1))
    broker.scale_in.assert_not_called()
    assert strat.scale_count == 0