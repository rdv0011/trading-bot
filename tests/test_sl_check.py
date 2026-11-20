"""Tests for SL check logic in _check_exits.

These tests verify that the stop loss check correctly triggers
when price falls below the stop loss level, including edge cases
like the time-weighted SL tightening and trailing stop interaction.
"""

import sys
from datetime import datetime, timedelta
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
mock_config.TRAILING_STOP_ENABLED = True
mock_config.TRAILING_ATR_MULT = 1.2
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
mock_config.TRAILING_SL_UPDATE_THRESHOLD = 0.002
mock_config.TRAILING_BREAKEVEN_MULT = 1.0
mock_config.GATE_ABS_VOL_ENABLED = False
mock_config.ABS_VOL_RATIO_CRITICAL = 0.10
mock_config.GATE_LIQUIDITY_EXIT = False


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
    broker.get_last_price.return_value = 50000.0

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


def setup_long_position(strategy, entry_price=50000.0, qty=0.1, hours_held=0.0, trail_extreme=None):
    """Helper to set up a long position for testing."""
    now = datetime.now()
    entry_time = now - timedelta(hours=hours_held)

    strategy.position = 1.0
    strategy.entry_price = entry_price
    strategy.entry_time = now - timedelta(hours=hours_held)
    strategy._initial_qty = qty
    strategy.current_meta = {
        "stop_loss_frac": 0.02,      # 2% SL
        "take_profit_frac": 0.04,    # 4% TP
        "max_hold_hours": 4.0,
        "regime": "trend",
        "_init": True,
    }
    strategy._last_atr14 = 1000.0
    if trail_extreme is not None:
        strategy.trail_extreme = trail_extreme
    else:
        strategy.trail_extreme = entry_price

    # Mock broker to return current price
    broker = strategy.broker
    broker.get_last_price.return_value = entry_price

    return now


def setup_short_position(strategy, entry_price=50000.0, qty=0.1, hours_held=0.0, trail_extreme=None):
    """Helper to set up a short position for testing."""
    now = datetime.now()
    entry_time = now - timedelta(hours=hours_held)

    strategy.position = -1.0
    strategy.entry_price = entry_price
    strategy.entry_time = now - timedelta(hours=hours_held)
    strategy._initial_qty = qty
    strategy.current_meta = {
        "stop_loss_frac": 0.02,
        "take_profit_frac": 0.04,
        "max_hold_hours": 4.0,
        "regime": "trend",
        "_init": True,
    }
    strategy._last_atr14 = 1000.0
    if trail_extreme is not None:
        strategy.trail_extreme = trail_extreme
    else:
        strategy.trail_extreme = entry_price

    broker = strategy.broker
    broker.get_last_price.return_value = entry_price

    return now


class TestSLCheckBasic:
    """Basic SL check tests."""

    def test_long_sl_triggered_below_initial_sl(self, strategy):
        """Test that SL triggers when price falls below initial SL."""
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1)
        strategy.broker.get_last_price.return_value = 48900.0  # Below 2% SL (49000)

        exit_reason = strategy._check_exits(48900.0, datetime.now())

        assert exit_reason == "sl", f"Expected 'sl', got {exit_reason}"

    def test_short_sl_triggered_above_initial_sl(self, strategy):
        """Test that SL triggers when price rises above initial SL for short."""
        now = setup_short_position(strategy, entry_price=50000.0, qty=0.1)
        strategy.broker.get_last_price.return_value = 51100.0  # Above 2% SL (51000)

        exit_reason = strategy._check_exits(51100.0, datetime.now())

        assert exit_reason == "sl", f"Expected 'sl', got {exit_reason}"

    def test_long_tp_triggered_above_initial_tp(self, strategy):
        """Test that TP triggers when price rises above initial TP."""
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1)
        strategy.broker.get_last_price.return_value = 52100.0  # Above 4% TP (52000)

        exit_reason = strategy._check_exits(52100.0, datetime.now())

        assert exit_reason == "tp", f"Expected 'tp', got {exit_reason}"

    def test_no_exit_when_price_between_sl_and_tp(self, strategy):
        """Test no exit when price is between SL and TP."""
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1)
        strategy.broker.get_last_price.return_value = 50500.0  # Between SL (49000) and TP (52000)

        exit_reason = strategy._check_exits(50500.0, datetime.now())

        assert exit_reason is None, f"Expected None, got {exit_reason}"


class TestTimeWeightedSLTightening:
    """Tests for time-weighted SL tightening."""

    def test_sl_tightens_after_50_percent_max_hold(self, strategy):
        """Test SL tightens to half after 50% of max_hold_hours."""
        # 4 hour max hold, so tightening at 2 hours
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1, hours_held=2.5)

        # Initial SL: 50000 * 0.98 = 49000
        # Tightened SL: 50000 * 0.99 = 49500
        strategy.broker.get_last_price.return_value = 49300.0  # Below tightened SL, above original

        exit_reason = strategy._check_exits(49300.0, datetime.now())

        assert exit_reason == "sl", f"Expected 'sl' with tightened SL, got {exit_reason}"

    def test_original_sl_used_before_50_percent(self, strategy):
        """Test original SL is used before 50% of max_hold_hours."""
        # 4 hour max hold, so tightening at 2 hours - we're at 1.5 hours
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1, hours_held=1.5)
        strategy.broker.get_last_price.return_value = 49300.0  # Below original SL (49000)? No, 49300 > 49000

        # At 49300, should NOT trigger (above original 49000, but below tightened 49500)
        exit_reason = strategy._check_exits(49300.0, datetime.now())

        assert exit_reason is None, f"Expected None (above original SL), got {exit_reason}"

    def test_sl_tightening_does_not_affect_trailing_calculations(self, strategy):
        """Verify that trailing SL uses original SL fraction, not tightened."""
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1, hours_held=2.5)
        strategy.trail_extreme = 51000.0  # Price moved up

        # At 2.5 hours, SL is tightened to 1% (0.01)
        # But trailing should use original 2% for base_sl calculation
        # base_sl = 50000 * 0.98 = 49000
        # trail_sl = 51000 - 1.2 * atr14 (need to check atr14)

        # With atr14=1000, trail_sl = 51000 - 1200 = 49800
        # eff_sl = max(49000, 49800) = 49800
        strategy._last_atr14 = 1000.0

        # Current price at 50500 - should NOT trigger trailing SL
        strategy.broker.get_last_price.return_value = 50500.0

        exit_reason = strategy._check_exits(50500.0, datetime.now())

        # Should not trigger because 50500 > 49800 (eff_sl)
        assert exit_reason is None, f"Expected None, got {exit_reason}"


class TestTrailingStopActivation:
    """Tests for trailing stop activation logic."""

    def test_trailing_sl_not_active_before_1R_profit(self, strategy):
        """Trailing SL should not activate before 1R profit."""
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1)
        strategy._last_atr14 = 1000.0

        # Price moved to 50500 (profit 500, less than 1R = 1000)
        strategy.broker.get_last_price.return_value = 50500.0

        exit_reason = strategy._check_exits(50500.0, datetime.now())

        # Should not trigger trailing SL (profit < 1R)
        assert exit_reason is None, f"Expected None (no trailing before 1R), got {exit_reason}"

    def test_trailing_sl_activates_after_1R_profit(self, strategy):
        """Trailing SL activates after 1R profit is reached."""
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1)
        strategy.trail_extreme = 51000.0  # 1000 profit = 1R (SL=2% of 50000 = 1000)
        strategy._last_atr14 = 1000.0

        # Price drops to 50000 (at breakeven)
        strategy.broker.get_last_price.return_value = 50000.0

        exit_reason = strategy._check_exits(50000.0, datetime.now())

        # Should trigger trailing SL at breakeven
        assert exit_reason == "trailing_sl", f"Expected 'trailing_sl', got {exit_reason}"

    def test_trailing_sl_tracks_from_breakeven(self, strategy):
        """After 1R profit, trailing SL should trail from breakeven."""
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1)
        strategy.trail_extreme = 51500.0  # 1500 profit > 1R
        strategy._last_atr14 = 1000.0

        # Price drops to 50200 - below breakeven but above trail from breakeven
        # trail_sl from breakeven = 50000 - 1.2*1000 = 48800
        # At 50200, should NOT trigger
        strategy.broker.get_last_price.return_value = 50200.0

        exit_reason = strategy._check_exits(50200.0, datetime.now())

        assert exit_reason is None, f"Expected None (above trail from breakeven), got {exit_reason}"

    def test_trailing_sl_triggers_after_breakeven_trail(self, strategy):
        """Trailing SL triggers when price drops below trail from breakeven."""
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1)
        strategy.trail_extreme = 51500.0
        strategy._last_atr14 = 1000.0

        # trail_sl from breakeven = 50000 - 1.2*1000 = 48800
        # Drop below 48800
        strategy.broker.get_last_price.return_value = 48700.0

        exit_reason = strategy._check_exits(48700.0, datetime.now())

        assert exit_reason == "trailing_sl", f"Expected 'trailing_sl', got {exit_reason}"


class TestEdgeCases:
    """Edge case tests."""

    def test_no_position_returns_none(self, strategy):
        """No position should return None immediately."""
        strategy.position = 0.0

        exit_reason = strategy._check_exits(50000.0, datetime.now())

        assert exit_reason is None

    def test_max_hold_exit(self, strategy):
        """Test max hold hours exit."""
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1, hours_held=5.0)
        strategy.broker.get_last_price.return_value = 50000.0

        exit_reason = strategy._check_exits(50000.0, datetime.now())

        assert exit_reason == "max_hold"

    def test_sl_price_exactly_at_boundary(self, strategy):
        """Test SL triggers exactly at boundary."""
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1)
        strategy.broker.get_last_price.return_value = 49000.0  # Exactly at 2% SL

        exit_reason = strategy._check_exits(49000.0, datetime.now())

        assert exit_reason == "sl"

    def test_short_sl_exactly_at_boundary(self, strategy):
        """Test short SL triggers exactly at boundary."""
        now = setup_short_position(strategy, entry_price=50000.0, qty=0.1)
        strategy.broker.get_last_price.return_value = 51000.0  # Exactly at 2% SL

        exit_reason = strategy._check_exits(51000.0, datetime.now())

        assert exit_reason == "sl"


class TestSlFractionMutation:
    """Tests for the sl_frac mutation fix."""

    def test_sl_fraction_not_mutated_for_trailing(self, strategy):
        """Verify sl_frac is not mutated when checking trailing stop."""
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1, hours_held=2.5)
        strategy._last_atr14 = 1000.0
        strategy.trail_extreme = 51000.0

        # Save original SL fraction
        original_sl_frac = strategy.current_meta.get("stop_loss_frac", 0.02)

        strategy.broker.get_last_price.return_value = 50500.0
        exit_reason = strategy._check_exits(50500.0, datetime.now())

        # Check that the meta still has original value
        assert strategy.current_meta.get("stop_loss_frac") == 0.02, \
            "SL fraction in meta should not be mutated"

    def test_tightened_sl_frac_only_used_for_initial_check(self, strategy):
        """Verify tightened SL fraction is only used for initial SL/TP check."""
        now = setup_long_position(strategy, entry_price=50000.0, qty=0.1, hours_held=2.5)

        # Price at 49300 - below tightened SL (49500) but above original (49000)
        # Should trigger SL because tightened is used for initial check
        strategy.broker.get_last_price.return_value = 49300.0

        exit_reason = strategy._check_exits(49300.0, datetime.now())

        assert exit_reason == "sl", "Should use tightened SL for initial check"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])