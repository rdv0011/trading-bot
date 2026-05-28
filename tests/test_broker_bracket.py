from unittest.mock import MagicMock, patch, call

import pytest

from binancebasebroker import (
    BinanceBaseBroker,
    BracketOrderResult,
    MarketOrderResult,
    PositionResult,
    SIGNAL_LONG,
    SIGNAL_SHORT,
)


class StubBroker(BinanceBaseBroker):
    def setup_client(self):
        pass

    def get_cash(self, *a):
        return 1000.0

    def get_position(self, *a):
        return None

    def get_last_price(self, *a):
        return 40000.0

    def _create_market_order(self, symbol, side, quantity):
        return None

    def _create_bracket_order(self, *a, **kw):
        return None

    def cancel_open_orders(self, *a, **kw):
        pass

    def close_position(self, *a):
        pass


@pytest.fixture
def broker():
    return StubBroker(config={})


def _good_market_order(price=40000.0):
    return MarketOrderResult(order_id="ord_1", entry_price=price)


def _good_bracket_order():
    return BracketOrderResult(tp_order_id="tp_1", sl_order_id="sl_1")


def test_bb01_invalid_signal_returns_failure(broker):
    result = broker.open_position_with_bracket("BTCUSDT", "invalid", 0.01)
    assert result.success is False
    assert "Invalid signal" in result.error


def test_bb02_market_order_returns_none_returns_failure(broker):
    broker._create_market_order = MagicMock(return_value=None)
    result = broker.open_position_with_bracket("BTCUSDT", SIGNAL_LONG, 0.01)
    assert result.success is False
    assert "None" in result.error


@patch("binancebasebroker.time.sleep")
def test_bb03_entry_price_immediate_correct_tp_sl(mock_sleep, broker):
    broker._create_market_order = MagicMock(return_value=_good_market_order(40000.0))
    broker._create_bracket_order = MagicMock(return_value=_good_bracket_order())

    result = broker.open_position_with_bracket("BTCUSDT", SIGNAL_LONG, 0.01, tp_frac=0.02, sl_frac=0.01)

    assert result.success is True
    assert result.data["tp_price"] == round(40000.0 * 1.02, 2)
    assert result.data["sl_price"] == round(40000.0 * 0.99, 2)
    mock_sleep.assert_not_called()


@patch("binancebasebroker.time.sleep")
def test_bb04_entry_price_none_confirmed_on_second_retry(mock_sleep, broker):
    broker._create_market_order = MagicMock(
        return_value=MarketOrderResult(order_id="ord_1", entry_price=None)
    )
    broker._create_bracket_order = MagicMock(return_value=_good_bracket_order())

    call_count = 0

    def get_position_side_effect(symbol):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return PositionResult(amount=0.01, entry_price=40000.0)
        return None

    broker.get_position = get_position_side_effect

    result = broker.open_position_with_bracket("BTCUSDT", SIGNAL_LONG, 0.01)

    assert result.success is True
    assert call_count == 2


@patch("binancebasebroker.time.sleep")
def test_bb05_all_retries_fail_position_closed(mock_sleep, broker):
    broker._create_market_order = MagicMock(
        return_value=MarketOrderResult(order_id="ord_1", entry_price=None)
    )
    broker.get_position = MagicMock(return_value=None)
    broker.close_position = MagicMock()

    result = broker.open_position_with_bracket("BTCUSDT", SIGNAL_LONG, 0.01)

    assert result.success is False
    assert "timeout" in result.error.lower()
    broker.close_position.assert_called_once()


@patch("binancebasebroker.time.sleep")
def test_bb06_bracket_order_fails_position_closed(mock_sleep, broker):
    broker._create_market_order = MagicMock(return_value=_good_market_order())
    broker._create_bracket_order = MagicMock(return_value=None)
    broker.get_position = MagicMock(
        return_value=PositionResult(amount=0.01, entry_price=40000.0)
    )
    broker.close_position = MagicMock()

    result = broker.open_position_with_bracket("BTCUSDT", SIGNAL_LONG, 0.01)

    assert result.success is False
    broker.close_position.assert_called_once()


@patch("binancebasebroker.time.sleep")
def test_bb07_short_signal_tp_below_entry_sl_above(mock_sleep, broker):
    entry = 40000.0
    broker._create_market_order = MagicMock(return_value=_good_market_order(entry))
    broker._create_bracket_order = MagicMock(return_value=_good_bracket_order())

    result = broker.open_position_with_bracket("BTCUSDT", SIGNAL_SHORT, 0.01, tp_frac=0.02, sl_frac=0.01)

    assert result.success is True
    assert result.data["tp_price"] < entry
    assert result.data["sl_price"] > entry


@patch("binancebasebroker.time.sleep")
def test_bb08_long_signal_tp_above_entry_sl_below(mock_sleep, broker):
    entry = 40000.0
    broker._create_market_order = MagicMock(return_value=_good_market_order(entry))
    broker._create_bracket_order = MagicMock(return_value=_good_bracket_order())

    result = broker.open_position_with_bracket("BTCUSDT", SIGNAL_LONG, 0.01, tp_frac=0.02, sl_frac=0.01)

    assert result.success is True
    assert result.data["tp_price"] > entry
    assert result.data["sl_price"] < entry


@patch("binancebasebroker.time.sleep")
def test_bb09_exception_in_market_order_returns_failure(mock_sleep, broker):
    broker._create_market_order = MagicMock(side_effect=RuntimeError("network error"))

    result = broker.open_position_with_bracket("BTCUSDT", SIGNAL_LONG, 0.01)

    assert result.success is False
    assert len(result.error) > 0


@patch("binancebasebroker.time.sleep")
def test_bb10_entry_timeout_long_closes_with_positive_qty(mock_sleep, broker):
    """LONG entry price timeout: close_position receives positive amount (SELL to close)."""
    broker._create_market_order = MagicMock(
        return_value=MarketOrderResult(order_id="ord_1", entry_price=None)
    )
    broker.get_position = MagicMock(return_value=None)
    broker.close_position = MagicMock()

    result = broker.open_position_with_bracket("BTCUSDT", SIGNAL_LONG, 0.01)

    assert result.success is False
    broker.close_position.assert_called_once()
    args = broker.close_position.call_args[0]
    assert args[1] > 0, f"Expected positive for LONG entry timeout close, got {args[1]}"


@patch("binancebasebroker.time.sleep")
def test_bb11_entry_timeout_short_closes_with_negative_qty(mock_sleep, broker):
    """SHORT entry price timeout: close_position receives negative amount (BUY to cover)."""
    broker._create_market_order = MagicMock(
        return_value=MarketOrderResult(order_id="ord_2", entry_price=None)
    )
    broker.get_position = MagicMock(return_value=None)
    broker.close_position = MagicMock()

    result = broker.open_position_with_bracket("BTCUSDT", SIGNAL_SHORT, 0.01)

    assert result.success is False
    broker.close_position.assert_called_once()
    args = broker.close_position.call_args[0]
    assert args[1] < 0, f"Expected negative for SHORT entry timeout close, got {args[1]}"
