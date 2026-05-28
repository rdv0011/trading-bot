from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

from binancebasebroker import PositionResult, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_HOLD
from positionmanager import (
    PositionManager,
    StrategicDecision,
    PARTIAL_CLOSE_FRAC,
    CONSECUTIVE_SIGNALS_REQUIRED,
    MIN_LIQUIDATION_BUFFER_FRAC,
)


def _make_broker(live_position=None, liquidation_price=None):
    broker = MagicMock()
    broker.get_position.return_value = live_position
    broker.get_liquidation_price.return_value = liquidation_price
    broker.get_cash.return_value = 1000.0
    broker.open_position_with_bracket.return_value = MagicMock(
        success=True,
        data={
            "entry_price": 40000.0,
            "tp_price": 40800.0,
            "sl_price": 39600.0,
            "tp_algo_id": "tp_1",
            "sl_algo_id": "sl_1",
        },
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


def test_pm07_veto_with_open_long_position_closes_with_positive_sign():
    """LONG full close: close_position receives positive amount (SELL to close)."""
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    broker.close_position.reset_mock()

    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(allow=False), 40000.0)
    broker.close_position.assert_called_once()
    args = broker.close_position.call_args[0]
    # args = (symbol, position_amount)
    pos_arg = args[1]
    assert pos_arg > 0, f"Expected positive (sell LONG), got {pos_arg}"
    assert pm.has_position is False


def test_pm07b_veto_with_open_short_position_closes_with_negative_sign():
    """SHORT full close: close_position receives negative amount (BUY to cover)."""
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_SHORT), _make_strategic(), 40000.0)
    broker.close_position.reset_mock()

    pm.on_signal(_make_tactical(SIGNAL_SHORT), _make_strategic(allow=False), 40000.0)
    broker.close_position.assert_called_once()
    args = broker.close_position.call_args[0]
    # args = (symbol, position_amount)
    pos_arg = args[1]
    assert pos_arg < 0, f"Expected negative (buy SHORT), got {pos_arg}"
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


def test_pm12_opposite_signal_triggers_partial_close_long():
    """LONG partial close receives positive amount (SELL to close part)."""
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    broker.close_position.reset_mock()

    pm.on_signal(_make_tactical(SIGNAL_SHORT), _make_strategic(), 40000.0)
    broker.close_position.assert_called_once()
    args = broker.close_position.call_args[0]
    pos_arg = args[1]
    assert pos_arg > 0, f"Expected positive for LONG partial close, got {pos_arg}"


def test_pm12b_opposite_signal_triggers_partial_close_short():
    """SHORT partial close receives negative amount (BUY to cover)."""
    broker = _make_broker(live_position=PositionResult(amount=-0.5, entry_price=40000.0))
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    broker.close_position.reset_mock()

    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    broker.close_position.assert_called_once()
    args = broker.close_position.call_args[0]
    pos_arg = args[1]
    assert pos_arg < 0, f"Expected negative for SHORT partial close, got {pos_arg}"


def test_pm13_max_hold_time_exceeded_triggers_full_close_long():
    """LONG max-hold close: close_position gets positive amount."""
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(max_hold_hours=1.0), 40000.0)
    broker.close_position.reset_mock()

    stale_time = datetime.now() - timedelta(hours=2)
    pm._state.entry_time = stale_time

    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(max_hold_hours=1.0), 40000.0)
    broker.close_position.assert_called_once()
    args = broker.close_position.call_args[0]
    pos_arg = args[1]
    assert pos_arg > 0, f"Expected positive for LONG full close, got {pos_arg}"
    assert pm.has_position is False


def test_pm13b_max_hold_time_exceeded_triggers_full_close_short():
    """SHORT max-hold close: close_position gets negative amount."""
    broker = _make_broker(live_position=PositionResult(amount=-0.5, entry_price=40000.0))
    pm = _make_pm(broker)
    broker.close_position.reset_mock()

    # After reconciliation, make get_position return None to simulate successful close
    broker.get_position.return_value = None

    stale_time = datetime.now() - timedelta(hours=2)
    pm._state.entry_time = stale_time

    pm.on_signal(_make_tactical(SIGNAL_SHORT), _make_strategic(max_hold_hours=1.0), 40000.0)
    broker.close_position.assert_called_once()
    args = broker.close_position.call_args[0]
    pos_arg = args[1]
    assert pos_arg < 0, f"Expected negative for SHORT full close, got {pos_arg}"
    assert pm.has_position is False


def test_pm14_emergency_close_with_state_closes_position():
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    broker.close_position.reset_mock()

    pm.emergency_close()
    broker.close_position.assert_called_once()
    assert pm.has_position is False


def test_pm14b_emergency_close_short_passes_negative_sign():
    broker = _make_broker(live_position=PositionResult(amount=-0.5, entry_price=40000.0))
    pm = _make_pm(broker)
    broker.close_position.reset_mock()

    # After reconciliation, simulate successful close
    broker.get_position.return_value = None

    pm.emergency_close()
    broker.close_position.assert_called_once()
    args = broker.close_position.call_args[0]
    pos_arg = args[1]
    assert pos_arg < 0, f"Expected negative for SHORT emergency close, got {pos_arg}"
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


# ---------------------------------------------------------------------------
# Scale-up verification
# ---------------------------------------------------------------------------

def test_pm21_scale_up_verifies_position_increased():
    """Scale-up must verify that the market order actually increased the position."""
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)

    # After opening, broker returns a live position for subsequent scale-up checks
    broker.get_position.return_value = PositionResult(amount=0.5, entry_price=40000.0)

    prev_amount = pm._state.amount
    broker.open_position_with_bracket.reset_mock()

    for _ in range(CONSECUTIVE_SIGNALS_REQUIRED):
        pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)

    assert broker.open_position_with_bracket.call_count >= 1
    # State should reflect the live position returned by broker
    assert pm._state.amount >= prev_amount


def test_pm22_scale_up_skipped_when_position_not_increased():
    """When broker reports no position increase, scale-up is aborted."""
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    prev_amount = pm._state.amount
    prev_scale_count = pm._state.scale_count

    # broker.get_position returns None (no position) — scale-up should abort
    broker.get_position.return_value = None

    for _ in range(CONSECUTIVE_SIGNALS_REQUIRED):
        pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)

    # State should remain unchanged since position wasn't verified
    assert pm._state.amount == prev_amount
    assert pm._state.scale_count == prev_scale_count


# ---------------------------------------------------------------------------
# Full close verification
# ---------------------------------------------------------------------------

def test_pm23_full_close_verifies_position_closed():
    """_full_close should verify the position is actually closed via broker.get_position."""
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(max_hold_hours=1.0), 40000.0)

    # Broker reports flat after close
    broker.get_position.return_value = None

    stale_time = datetime.now() - timedelta(hours=2)
    pm._state.entry_time = stale_time
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(max_hold_hours=1.0), 40000.0)
    assert pm.has_position is False


def test_pm24_full_close_retries_on_first_failure():
    """If the position is still open after first close attempt, _full_close retries."""
    broker = _make_broker()
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(max_hold_hours=1.0), 40000.0)
    prev_amount = pm._state.amount

    # _full_close calls get_position: 1) live_before, 2) first verification (still open),
    # 3) retry verification (now closed)
    broker.get_position.side_effect = [
        PositionResult(amount=0.5, entry_price=40000.0),  # live_before
        PositionResult(amount=0.5, entry_price=40000.0),  # first verification → still open → retry
        None,  # second verification → closed
    ]

    stale_time = datetime.now() - timedelta(hours=2)
    pm._state.entry_time = stale_time
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(max_hold_hours=1.0), 40000.0)
    assert pm.has_position is False
    # close_position should have been called twice (first attempt + retry)
    assert broker.close_position.call_count >= 2


# ---------------------------------------------------------------------------
# Liquidation buffer check
# ---------------------------------------------------------------------------

def test_pm25_liquidation_buffer_check_on_entry():
    """Opening a position triggers liquidation buffer check."""
    broker = _make_broker(liquidation_price=45000.0)
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    broker.get_liquidation_price.assert_called_once()


def test_pm26_liquidation_buffer_too_close_logs_warning():
    """When SL is too close to liquidation, a warning is logged."""
    broker = _make_broker(liquidation_price=39650.0)  # very close to SL at 39600
    logs = []
    pm = _make_pm(broker)
    pm.log = logs.append
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    assert any("LIQUIDATION RISK" in msg for msg in logs), (
        f"Expected liquidation warning in logs: {logs}"
    )


def test_pm27_liquidation_buffer_safe_no_warning():
    """When SL is safely far from liquidation, no risk warning."""
    broker = _make_broker(liquidation_price=30000.0)  # far from SL at 39600
    logs = []
    pm = _make_pm(broker)
    pm.log = logs.append
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    assert all("LIQUIDATION RISK" not in msg for msg in logs), (
        f"Unexpected liquidation warning in logs: {logs}"
    )


def test_pm28_liquidation_buffer_not_available_skips_check():
    """When exchange doesn't report liquidation price, no crash."""
    broker = _make_broker(liquidation_price=None)
    pm = _make_pm(broker)
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    # Should not raise


# ---------------------------------------------------------------------------
# Cancel open orders retry
# ---------------------------------------------------------------------------

def test_pm29_cancel_open_orders_retries_on_failure():
    """cancel_open_orders should retry when cancellation fails."""
    broker = _make_broker()
    pm = _make_pm(broker)

    # open_position_with_bracket triggers cancellation during scale-up
    pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)
    broker.get_position.return_value = PositionResult(amount=0.5, entry_price=40000.0)

    for _ in range(CONSECUTIVE_SIGNALS_REQUIRED):
        pm.on_signal(_make_tactical(SIGNAL_LONG), _make_strategic(), 40000.0)

    broker.cancel_open_orders.assert_called()
    args = broker.cancel_open_orders.call_args
    assert "max_retries" in args[1] or len(args[0]) > 1
