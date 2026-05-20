from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

from binancebasebroker import PositionResult, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_HOLD
from positionmanager import (
    PositionManager,
    StrategicDecision,
    PARTIAL_CLOSE_FRAC,
    CONSECUTIVE_SIGNALS_REQUIRED,
)


def _make_broker(live_position=None):
    broker = MagicMock()
    broker.get_position.return_value = live_position
    broker.get_cash.return_value = 1000.0
    broker.open_position_with_bracket.return_value = MagicMock(
        success=True, data={"entry_price": 40000.0}
    )
    return broker


def _make_pm(broker):
    return PositionManager(broker, "BTC", "USDT", logger=lambda _: None)


def _make_strategic(allow=True, regime="trend", max_hold_hours=4.0):
    return StrategicDecision(
        allow_trading=allow,
        market_regime=regime,
        volatility_state="normal",
        recommended_leverage=5.0,
        max_exposure_frac=0.5,
        stake_long_frac=0.1,
        stake_short_frac=0.05,
        stop_loss_frac=0.02,
        take_profit_frac=0.04,
        max_hold_hours=max_hold_hours,
    )


def _make_tactical(signal):
    t = MagicMock()
    t.signal = signal
    return t


def test_pm01_startup_no_live_position_starts_flat():
    broker = _make_broker(live_position=None)
    pm = _make_pm(broker)
    assert pm.has_position is False


def test_pm02_startup_with_live_long_position_reconciles():
    live = PositionResult(amount=0.5, entry_price=40000.0)
    broker = _make_broker(live_position=live)
    pm = _make_pm(broker)
    assert pm.has_position is True
    assert pm.position_side == SIGNAL_LONG


def test_pm03_startup_with_live_short_position_reconciles():
    live = PositionResult(amount=-0.5, entry_price=40000.0)
    broker = _make_broker(live_position=live)
    pm = _make_pm(broker)
    assert pm.has_position is True
    assert pm.position_side == SIGNAL_SHORT


def test_pm04_on_signal_hold_no_position_no_broker_call():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_HOLD), _make_strategic(), 40000.0)
    broker.open_position_with_bracket.assert_not_called()


def test_pm05_on_signal_long_opens_long():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    broker.open_position_with_bracket.assert_called_once()
    args = broker.open_position_with_bracket.call_args
    assert args[0][1] == SIGNAL_LONG


def test_pm06_on_signal_short_opens_short():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_SHORT), _make_strategic(), 40000.0)
    broker.open_position_with_bracket.assert_called_once()
    args = broker.open_position_with_bracket.call_args
    assert args[0][1] == SIGNAL_SHORT


def test_pm07_veto_with_open_position_closes():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    broker.close_position.reset_mock()

    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(allow=False), 40000.0)
    broker.close_position.assert_called_once()
    assert pm.has_position is False


def test_pm08_veto_with_no_position_no_close_call():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_HOLD), _make_strategic(allow=False), 40000.0)
    broker.close_position.assert_not_called()


def test_pm09_chop_regime_no_new_entry():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(regime="chop"), 40000.0)
    broker.open_position_with_bracket.assert_not_called()


def test_pm10_same_direction_below_consecutive_threshold_no_scale_up():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    entry_call_count = broker.open_position_with_bracket.call_count
    assert entry_call_count == 1

    broker.open_position_with_bracket.reset_mock()
    pm.on_signal(_make_tactical(SIGNAL_HOLD), _make_strategic(), 40000.0)
    broker.open_position_with_bracket.assert_not_called()


def test_pm11_same_direction_two_consecutive_triggers_scale_up():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    broker.open_position_with_bracket.reset_mock()

    for _ in range(CONSECUTIVE_SIGNALS_REQUIRED):
        pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)

    assert broker.open_position_with_bracket.call_count >= 1


def test_pm12_opposite_direction_triggers_partial_close():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    broker.close_position.reset_mock()

    pm.on_signal(_make_tactical(SIGNAL_SHORT), _make_strategic(), 40000.0)
    broker.close_position.assert_called_once()


def test_pm13_max_hold_time_exceeded_triggers_full_close():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(max_hold_hours=1.0), 40000.0)
    broker.close_position.reset_mock()

    stale_time = datetime.now() - timedelta(hours=2)
    pm._state.entry_time = stale_time

    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(max_hold_hours=1.0), 40000.0)
    broker.close_position.assert_called_once()
    assert pm.has_position is False


def test_pm14_emergency_close_with_state_closes_position():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    broker.close_position.reset_mock()

    pm.emergency_close()
    broker.close_position.assert_called_once()
    assert pm.has_position is False


def test_pm15_emergency_close_with_no_state_no_broker_call():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.emergency_close()
    broker.close_position.assert_not_called()


def test_pm16_emergency_close_live_with_live_position_closes():
    broker = _make_broker()
    pm = _make_pm(broker)
    broker.get_position.return_value = PositionResult(amount=0.5, entry_price=40000.0)

    pm.emergency_close_live()

    broker.close_position.assert_called_once()
    assert pm.has_position is False


def test_pm17_emergency_close_live_no_live_position_no_close():
    broker = _make_broker()
    pm = _make_pm(broker)
    broker.get_position.return_value = None

    pm.emergency_close_live()
    broker.close_position.assert_not_called()


def test_pm18_emergency_close_live_cancel_raises_still_closes():
    broker = _make_broker()
    pm = _make_pm(broker)
    broker.cancel_open_orders.side_effect = RuntimeError("network error")
    broker.get_position.return_value = PositionResult(amount=0.5, entry_price=40000.0)

    pm.emergency_close_live()
    broker.close_position.assert_called_once()


def test_pm19_broker_entry_failure_state_remains_none():
    broker = _make_broker()
    broker.open_position_with_bracket.return_value = MagicMock(success=False, error="rejected")
    pm = _make_pm(broker)

    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    assert pm.has_position is False


def test_pm20_get_cash_called_with_quote_symbol():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    broker.get_cash.assert_called_with("USDT")
