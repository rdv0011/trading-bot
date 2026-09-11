"""Regression tests for the live precision bug (-1111) and the degenerate
frozen-model guard discovered during live validation.

  1. broker.scale_in / broker.close_position_fraction quantize quantities to
     the symbol step size before order placement. The entry path uses
     _quantize_qty() but scale-in / partial-close passed raw floats (e.g.
     0.0285 * 0.5 = 0.01425, 5 decimals) to futures_create_order -> Binance
     APIError -1111 "Precision is over the maximum defined for this asset".
  2. strategy._tactical_model_is_degenerate() detects a frozen 1-tree model
     (early-stopped at iteration 0) which predicts ~constant values that never
     cross the threshold, and forces an immediate retrain.

broker.py / strategy.py live at the repo root and use bare sibling imports.
Heavy imports stay inside fixtures/test bodies so this file never perturbs
sibling tests in the same pytest process.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

repo_root = Path(__file__).resolve().parent.parent

# Concrete config values (never MagicMock defaults) because strategy's
# __init__ reads ABSOLUTE_THRESHOLD / WALKFORWARD_RETRAIN_EVERY /
# MODEL_REFRESH_EVERY via getattr, and unset MagicMock attrs turn into
# truthy MagicMocks.
mock_config = MagicMock()
mock_config.ABSOLUTE_THRESHOLD = 0.006
mock_config.WALKFORWARD_RETRAIN_EVERY = 100
mock_config.MODEL_REFRESH_EVERY = 50
mock_config.ADAPTIVE_THRESHOLD_ENABLED = False
mock_config.SCALING_ENABLED = False
mock_config.PARTIAL_EXIT_ENABLED = False
mock_config.TRAILING_STOP_ENABLED = False
mock_config.TRADE_COOLDOWN_ENABLED = False


@pytest.fixture
def broker_cls():
    sys.path.insert(0, str(repo_root))
    import broker as broker_mod
    return broker_mod


@pytest.fixture
def broker(broker_cls):
    """BinanceBroker with setup_client bypassed and client mocked."""
    with patch.object(broker_cls.BinanceBroker, "setup_client"):
        b = broker_cls.BinanceBroker(symbol="BTCUSDT")
    b.client = MagicMock()
    return b


@pytest.fixture
def strategy_mod():
    sys.path.insert(0, str(repo_root))
    with patch.dict(sys.modules, {"config": mock_config}):
        import strategy as strategy_mod
    return strategy_mod


# ── Requirement 1: quantity quantization (fixes APIError -1111) ────────
def test_quantize_qty_floors_to_step(broker):
    assert broker._quantize_qty(0.01425) == 0.014
    assert broker._quantize_qty(0.009405) == 0.009
    assert broker._quantize_qty(0.0005) == 0.0
    assert broker._quantize_qty(0.001) == 0.001
    assert broker._quantize_qty(1.234567) == 1.234


def test_scale_in_quantizes_quantity(broker):
    broker.client.futures_create_order.return_value = {"orderId": "1"}
    broker.get_position = MagicMock(return_value=MagicMock(amount=-0.042))
    result = broker.scale_in("BTCUSDT", "SELL", 0.01425)
    _, kwargs = broker.client.futures_create_order.call_args
    assert kwargs["quantity"] == 0.014  # not 0.01425
    assert result is not None


def test_scale_in_below_min_returns_none_and_no_order(broker):
    result = broker.scale_in("BTCUSDT", "SELL", 0.0004)
    assert result is None
    broker.client.futures_create_order.assert_not_called()


def test_close_position_fraction_quantizes(broker):
    broker.client.futures_create_order.return_value = {
        "executedQty": "0.009",
        "cumQuote": "708.3",
    }
    broker.get_position = MagicMock(return_value=MagicMock(amount=-0.0285))
    fill = broker.close_position_fraction("BTCUSDT", 0.33)
    # 0.0285 * 0.33 = 0.009405 -> quantized 0.009
    _, kwargs = broker.client.futures_create_order.call_args
    assert kwargs["quantity"] == 0.009
    assert kwargs.get("reduceOnly") is True
    assert fill is not None


def test_close_position_fraction_below_min_skips(broker):
    broker.get_position = MagicMock(return_value=MagicMock(amount=-0.001))
    fill = broker.close_position_fraction("BTCUSDT", 0.33)
    # 0.001 * 0.33 = 0.00033 -> 0.0 < MIN_TRADEABLE_QUANTITY -> skip
    assert fill is None
    broker.client.futures_create_order.assert_not_called()


# ── Requirement 2: degenerate frozen-model guard ───────────────────────
def test_degenerate_detected_when_one_tree(strategy_mod):
    strat = strategy_mod.DualMLStrategy(
        broker=MagicMock(),
        tactical_model=MagicMock(model=MagicMock(tree_count_=1)),
        strategic_model=MagicMock(),
        feature_cols=["close"],
        config=mock_config,
    )
    assert strat._tactical_model_is_degenerate() is True


def test_healthy_model_not_degenerate(strategy_mod):
    strat = strategy_mod.DualMLStrategy(
        broker=MagicMock(),
        tactical_model=MagicMock(model=MagicMock(tree_count_=100)),
        strategic_model=MagicMock(),
        feature_cols=["close"],
        config=mock_config,
    )
    assert strat._tactical_model_is_degenerate() is False


def test_missing_tree_count_attr_assumed_healthy(strategy_mod):
    from types import SimpleNamespace

    # SimpleNamespace has no auto-created attrs (MagicMock would fabricate a
    # truthy tree_count_ and break the "absent attribute" case).
    strat = strategy_mod.DualMLStrategy(
        broker=MagicMock(),
        tactical_model=MagicMock(model=SimpleNamespace()),
        strategic_model=MagicMock(),
        feature_cols=["close"],
        config=mock_config,
    )
    assert strat._tactical_model_is_degenerate() is False


def test_no_model_not_flag_degenerate(strategy_mod):
    strat = strategy_mod.DualMLStrategy(
        broker=MagicMock(),
        tactical_model=None,
        strategic_model=MagicMock(),
        feature_cols=["close"],
        config=mock_config,
    )
    # Model is None -> handled by the model-is-None branch, not this helper
    assert strat._tactical_model_is_degenerate() is False