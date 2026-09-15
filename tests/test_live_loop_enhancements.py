"""Tests for live-loop enhancements (run on ARM server conda env):
  1. KeyboardInterrupt closes all open positions
  2. Size-based log rotation bounds per-day log size
  3. Atomic model save (no partial file visible to a parallel reader)
  4. Periodic model hot-swap when a newer model lands on disk

strategy.py / logger.py / model.py live at the repo root and use bare
sibling imports. Heavy imports stay inside fixtures/test bodies so this file
never perturbs sibling tests in the same pytest process.
"""

import sys
import os
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

repo_root = Path(__file__).resolve().parent.parent
simplified_dir = repo_root

mock_config = MagicMock()
mock_config.MODEL_DIR = "models"
mock_config.WALKFORWARD_RETRAIN_EVERY = 100
mock_config.MODEL_REFRESH_EVERY = 5
mock_config.ABSOLUTE_THRESHOLD = 0.006
mock_config.TACTICAL_TF = "15m"
mock_config.STRATEGIC_TF = "1h"
mock_config.SYMBOL = "BTCUSDT"
mock_config.STAKE_LONG_FRAC_DEFAULT = 0.10
mock_config.STAKE_SHORT_FRAC_DEFAULT = 0.05
mock_config.STOP_LOSS_FRAC_DEFAULT = 0.02
mock_config.TAKE_PROFIT_FRAC_DEFAULT = 0.04
mock_config.MAX_HOLD_HOURS_DEFAULT = 4.0
mock_config.LEVERAGE_DEFAULT = 1.0
# Phase 1-3 flags: explicitly disabled for legacy tests. MagicMock auto-creates
# truthy attributes for unset names (truthy bools / MagicMock numerics crash
# int()/float() in strategy._current_threshold etc.), so every new flag must be
# set to a concrete value here.
mock_config.FEATURE_CACHE_TTL = 60
mock_config.ADAPTIVE_THRESHOLD_ENABLED = False
mock_config.ADAPTIVE_LOOKBACK = 200
mock_config.ADAPTIVE_QUANTILE = 0.95
mock_config.ADAPTIVE_MIN_THRESHOLD = 0.002
mock_config.ADAPTIVE_MAX_THRESHOLD = 0.02
mock_config.TRAILING_STOP_ENABLED = False
mock_config.TRAILING_ATR_MULT = 1.5
mock_config.TRAILING_BREAKEVEN_MULT = 1.0
mock_config.GATE_EXTREME_VOL = False
mock_config.EXTREME_VOL_RATIO = 1.8
mock_config.SCALING_ENABLED = False
mock_config.MAX_SCALE_COUNT = 2
mock_config.SCALE_CONFIRM_BARS = 2
mock_config.SCALE_STAKE_FRAC = 0.5
mock_config.PARTIAL_EXIT_ENABLED = False
mock_config.PARTIAL_EXIT_FRACTION = 0.33
mock_config.REVERSAL_FULL_CLOSE_STREAK = 2
mock_config.TRADE_COOLDOWN_ENABLED = False
mock_config.TRADE_COOLDOWN_MINUTES = 30


@pytest.fixture
def simp_path(monkeypatch):
    monkeypatch.syspath_prepend(str(simplified_dir))
    return simplified_dir


@pytest.fixture
def strategy(simp_path):
    with patch.dict(sys.modules, {"config": mock_config}):
        import strategy as strategy_mod

    broker = MagicMock()
    broker.symbol = "BTCUSDT"
    broker.get_last_price.return_value = 65000.0
    broker.close_position.return_value = 64990.0
    broker.get_equity.return_value = 1000.0
    broker.cancel_open_orders.return_value = None

    tactical = MagicMock()
    tactical.params = {"iterations": 10}
    tactical.metadata = {"feature_cols": ["close"], "saved_at": "2026-01-01T00:00:00"}
    tactical.model = MagicMock()

    strategic = MagicMock()
    strategic.metadata = {"feature_cols": [], "saved_at": "2026-01-01T00:00:00"}
    strategic.model = MagicMock()

    return strategy_mod.DualMLStrategy(
        broker=broker,
        tactical_model=tactical,
        strategic_model=strategic,
        feature_cols=["close"],
        config=mock_config,
    )


# ── Requirement 1: KeyboardInterrupt closes open positions ─────────────
def test_shutdown_closes_open_position(strategy):
    strategy.position = 1.0
    strategy.entry_price = 65000.0
    strategy.current_trade = {
        "timestamp": "2026-01-01T00:00:00", "side": "long",
        "entry_price": 65000.0, "exit_price": "",
        "qty": 0.5, "equity_before": 1000.0,
    }

    def fake_live_iteration():
        raise KeyboardInterrupt()

    with patch.object(strategy.__class__, "_live_iteration", side_effect=fake_live_iteration):
        strategy.run_live_loop(sleep_seconds=0.001, max_iterations=1)

    strategy.broker.close_position.assert_called()
    strategy.broker.cancel_open_orders.assert_called()


def test_shutdown_with_no_position_skips_close(strategy):
    strategy.position = 0.0

    def fake_live_iteration():
        raise KeyboardInterrupt()

    with patch.object(strategy.__class__, "_live_iteration", side_effect=fake_live_iteration):
        strategy.run_live_loop(sleep_seconds=0.001, max_iterations=1)

    strategy.broker.close_position.assert_not_called()
    strategy.broker.cancel_open_orders.assert_called()


def test_shutdown_during_sleep_closes_open_position(strategy):
    strategy.position = 1.0
    strategy.entry_price = 65000.0
    strategy.current_trade = {
        "timestamp": "2026-01-01T00:00:00", "side": "long",
        "entry_price": 65000.0, "exit_price": "",
        "qty": 0.5, "equity_before": 1000.0,
    }

    iterations = {"count": 0}

    def fake_live_iteration():
        iterations["count"] += 1

    def fake_sleep(_seconds):
        raise KeyboardInterrupt()

    with patch.object(strategy.__class__, "_live_iteration", side_effect=fake_live_iteration), \
         patch("time.sleep", side_effect=fake_sleep):
        strategy.run_live_loop(sleep_seconds=0.001, max_iterations=5)

    assert iterations["count"] == 1, "loop must break after interrupt during sleep"
    strategy.broker.close_position.assert_called()
    strategy.broker.cancel_open_orders.assert_called()


# ── Requirement 2: Size-based log rotation ─────────────────────────────
def test_log_rotates_past_max_bytes(tmp_path, simp_path):
    from logger import DailyDatedLogHandler
    handler = DailyDatedLogHandler(log_dir=tmp_path, max_bytes=100, backup_count=3)
    logger_instance = logging.getLogger("test_rotate")
    logger_instance.handlers = [handler]
    logger_instance.setLevel(logging.DEBUG)

    for _ in range(200):
        logger_instance.debug("a" * 50)

    backup_files = list(tmp_path.glob("trading_*.log.*"))
    assert len(backup_files) >= 1, "expected at least one rotated backup"
    indexes = [int(p.name.rsplit('.', 1)[-1]) for p in backup_files]
    assert max(indexes) <= 3, "backup count must not exceed configured cap"
    handler.close()


def test_log_prune_removes_old_rotated_backups(tmp_path, simp_path):
    from logger import DailyDatedLogHandler
    stale = tmp_path / "trading_2019-01-01.log.2"
    stale.write_text("stale")
    handler = DailyDatedLogHandler(log_dir=tmp_path, max_bytes=10_000_000, backup_count=3)
    handler._prune_old_files()
    assert not stale.exists()
    handler.close()


def test_trade_csv_suppressed_inside_decorated_fn():
    import logger as logger_mod
    from logger import suppress_trade_csv, log_trade

    @suppress_trade_csv
    def run():
        logger_mod.set_trade_csv_enabled(True)
        log_trade({"timestamp": "2024-01-01T00:00:00", "symbol": "BTCUSDT",
                   "side": "long", "entry_price": 40000.0, "exit_price": 39900.0,
                   "qty": 1.0, "pnl": -100.0, "pnl_pct": -0.01, "exit_reason": "test",
                   "regime": "trend", "stake_frac": 0.1, "leverage": 1.0,
                   "stop_loss": 0.02, "take_profit": 0.04, "max_hold_hours": 4.0,
                   "equity_before": 1000.0, "equity_after": 900.0, "fee_paid": 0.0,
                   "slippage_paid": 0.0, "tactical_pred": 0.01, "strategic_params": "{}"})
        return logger_mod._TRADE_CSV_ENABLED

    assert run() is False  # suppressed inside, restored after
    assert logger_mod._TRADE_CSV_ENABLED is True  # state restored on exit
    assert not list(logger_mod.LOG_DIR.glob("trades_*.csv")) or all(
        "2024-01-01" not in p.read_text() for p in logger_mod.LOG_DIR.glob("trades_*.csv")
    )


# ── Requirement 3: Atomic model save ───────────────────────────────────
def test_atomic_save_writes_no_partial(tmp_path, simp_path):
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor
    from model import CatBoostModel

    df = pd.DataFrame({"close": np.linspace(1.0, 2.0, 50)})
    df["x"] = df["close"] * 2
    m = CatBoostModel(model_type="tactical", model_dir=str(tmp_path))
    base = CatBoostRegressor(iterations=4, depth=2, learning_rate=0.05, verbose=False)
    base.fit(df[["x"]], df["close"])
    m.model = base
    m.metadata = {"feature_cols": ["x"], "n_features": 1, "multi_output": False}

    m.save()
    meta = json.loads((tmp_path / "model_tactical_meta.json").read_text())
    assert "saved_at" in meta
    assert (tmp_path / "model_tactical.cbm").exists()
    assert not list(tmp_path.glob("*.tmp")), "no leftover temp files after save"


# ── Requirement 3b: hot-swap picks up newer model ──────────────────────
def test_refresh_models_hot_swaps_newer_tactical(strategy, tmp_path, simp_path):
    (tmp_path / "model_tactical_meta.json").write_text(
        json.dumps({"feature_cols": ["close", "new_feat"], "saved_at": "2026-02-01T00:00:00"})
    )
    (tmp_path / "model_tactical.cbm").write_bytes(b"dummy")
    (tmp_path / "model_strategic_meta.json").write_text(
        json.dumps({"feature_cols": [], "saved_at": "2026-01-01T00:00:00"})
    )
    (tmp_path / "model_strategic.cbm").write_bytes(b"dummy")

    strategy.model_dir = str(tmp_path)
    strategy._last_saved_at = {"tactical": "2026-01-01T00:00:00", "strategic": "2026-01-01T00:00:00"}

    fresh_tact = MagicMock()
    fresh_tact.metadata = {"feature_cols": ["close", "new_feat"], "saved_at": "2026-02-01T00:00:00"}
    fresh_tact.model = MagicMock()
    fresh_tact.params = {"iterations": 10}

    with patch("model.CatBoostModel") as mock_cls:
        mock_cls.return_value = fresh_tact
        strategy._refresh_models()

    assert strategy.tactical_model is fresh_tact
    assert strategy.feature_cols == ["close", "new_feat"]
    assert strategy._last_saved_at["tactical"] == "2026-02-01T00:00:00"


def test_refresh_models_no_change_when_same_timestamp(strategy, tmp_path, simp_path):
    (tmp_path / "model_tactical_meta.json").write_text(
        json.dumps({"feature_cols": ["close"], "saved_at": "2026-01-01T00:00:00"})
    )
    (tmp_path / "model_strategic_meta.json").write_text(
        json.dumps({"feature_cols": [], "saved_at": "2026-01-01T00:00:00"})
    )
    strategy.model_dir = str(tmp_path)
    strategy._last_saved_at = {"tactical": "2026-01-01T00:00:00", "strategic": "2026-01-01T00:00:00"}

    old_model = strategy.tactical_model
    with patch("model.CatBoostModel") as mock_cls:
        strategy._refresh_models()
        mock_cls.assert_not_called()
    assert strategy.tactical_model is old_model