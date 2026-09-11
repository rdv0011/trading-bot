"""Tests for startup position adoption (run on ARM server conda env):
  1. Adopts an existing short on the exchange into strategy state
  2. Adopts an existing long
  3. No open position -> starts flat
  4. ADOPT_EXISTING_POSITION=False + open position -> refuses to start
  5. Broker query failure / junk payload -> fail-open, starts flat

Mirrors the mock conventions of test_live_loop_enhancements.py.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
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
mock_config.ADOPT_EXISTING_POSITION = True


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
    broker.get_equity.return_value = 1000.0
    broker.close_position.return_value = 64990.0
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


def test_adopts_existing_short(strategy):
    strategy.broker.get_position.return_value = SimpleNamespace(
        amount=-0.053, entry_price=77851.15
    )

    ok = strategy.sync_exchange_position()

    assert ok is True
    assert strategy.position == -1.0
    assert strategy.entry_price == 77851.15
    assert strategy.entry_time is not None
    assert strategy._initial_qty == 0.053
    assert strategy.trail_extreme == 77851.15
    assert strategy.last_entry_time == strategy.entry_time
    assert strategy.current_trade is not None
    assert strategy.current_trade["side"] == "short"
    assert strategy.current_trade["qty"] == 0.053
    assert strategy.current_trade["equity_before"] == 1000.0


def test_adopts_existing_long(strategy):
    strategy.broker.get_position.return_value = SimpleNamespace(
        amount=0.5, entry_price=64000.0
    )

    ok = strategy.sync_exchange_position()

    assert ok is True
    assert strategy.position == 1.0
    assert strategy.entry_price == 64000.0
    assert strategy._initial_qty == 0.5
    assert strategy.current_trade["side"] == "long"


def test_no_position_starts_flat(strategy):
    strategy.broker.get_position.return_value = None

    ok = strategy.sync_exchange_position()

    assert ok is True
    assert strategy.position == 0.0
    assert strategy.entry_price == 0.0
    assert strategy.current_trade is None


def test_flag_off_refuses_when_position_exists(strategy):
    strategy.broker.get_position.return_value = SimpleNamespace(
        amount=-0.053, entry_price=77851.15
    )
    mock_config.ADOPT_EXISTING_POSITION = False
    try:
        ok = strategy.sync_exchange_position()
    finally:
        mock_config.ADOPT_EXISTING_POSITION = True

    assert ok is False
    assert strategy.position == 0.0
    assert strategy.current_trade is None


def test_broker_error_fails_open(strategy):
    strategy.broker.get_position.side_effect = RuntimeError("network down")

    ok = strategy.sync_exchange_position()

    assert ok is True
    assert strategy.position == 0.0
    assert strategy.current_trade is None


def test_junk_position_payload_fails_open(strategy):
    strategy.broker.get_position.return_value = MagicMock()

    ok = strategy.sync_exchange_position()

    assert ok is True
    assert strategy.position == 0.0
    assert strategy.current_trade is None