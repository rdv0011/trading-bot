from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from riskguard import RiskGuard


def test_rg01_first_call_initialises_start_of_day_equity():
    rg = RiskGuard()
    result = rg.update(1000.0)
    assert result is True
    assert rg._start_of_day_equity == 1000.0


def test_rg02_below_daily_loss_threshold_allowed():
    rg = RiskGuard(max_daily_loss_frac=0.05)
    rg.update(1000.0)
    result = rg.update(960.0)
    assert result is True
    assert rg.is_halted is False


def test_rg03_exactly_at_daily_loss_threshold_halted():
    rg = RiskGuard(max_daily_loss_frac=0.05)
    rg.update(1000.0)
    result = rg.update(950.0)
    assert result is False
    assert rg.is_halted is True


def test_rg04_exceeds_daily_loss_threshold_halted():
    rg = RiskGuard(max_daily_loss_frac=0.05)
    rg.update(1000.0)
    result = rg.update(900.0)
    assert result is False
    assert rg.is_halted is True


def test_rg05_drawdown_below_threshold_allowed():
    rg = RiskGuard(max_drawdown_frac=0.15, max_daily_loss_frac=1.0)
    rg.update(1000.0)
    rg.update(1100.0)
    result = rg.update(940.0)
    assert result is True
    assert rg.is_halted is False


def test_rg06_drawdown_at_threshold_halted():
    rg = RiskGuard(max_drawdown_frac=0.15)
    rg.update(1000.0)
    rg.update(1100.0)
    result = rg.update(935.0)
    assert result is False
    assert rg.is_halted is True


def test_rg07_once_halted_stays_halted_on_recovery():
    rg = RiskGuard(max_daily_loss_frac=0.05)
    rg.update(1000.0)
    rg.update(950.0)
    assert rg.is_halted is True
    result = rg.update(1100.0)
    assert result is False
    assert rg.is_halted is True


def test_rg08_new_day_resets_halt_state():
    rg = RiskGuard(max_daily_loss_frac=0.05)
    today = date(2024, 1, 1)
    tomorrow = date(2024, 1, 2)

    with patch("riskguard.date") as mock_date:
        mock_date.today.return_value = today
        rg.update(1000.0)
        rg.update(900.0)
        assert rg.is_halted is True

        # New day should reset the halt
        mock_date.today.return_value = tomorrow
        rg.update(800.0)

    assert rg._start_of_day_equity == 800.0
    assert rg.is_halted is False


def test_rg09_clamp_leverage_below_cap_unchanged():
    rg = RiskGuard(max_leverage=10.0)
    assert rg.clamp_leverage(5.0) == 5.0


def test_rg10_clamp_leverage_above_cap_clamped():
    rg = RiskGuard(max_leverage=10.0)
    assert rg.clamp_leverage(15.0) == 10.0


def test_rg11_clamp_leverage_exactly_at_cap_unchanged():
    rg = RiskGuard(max_leverage=10.0)
    assert rg.clamp_leverage(10.0) == 10.0


def test_rg12_peak_equity_tracks_correctly():
    rg = RiskGuard()
    rg.update(1000.0)
    rg.update(1200.0)
    rg.update(1100.0)
    assert rg._peak_equity == 1200.0


def test_rg13_reset_clears_halt_and_resets_equity_trackers():
    rg = RiskGuard(max_daily_loss_frac=0.05)
    rg.update(1000.0)
    rg.update(900.0)
    assert rg.is_halted is True

    rg.reset(500.0)
    assert rg.is_halted is False
    assert rg._start_of_day_equity == 500.0
    assert rg._peak_equity == 500.0


def test_rg14_reset_allows_trading_to_resume():
    rg = RiskGuard(max_daily_loss_frac=0.05)
    rg.update(1000.0)
    rg.update(900.0)
    assert rg.is_halted is True

    rg.reset(500.0)
    result = rg.update(480.0)
    assert result is True
    assert rg.is_halted is False
