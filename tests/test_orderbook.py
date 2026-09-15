"""Tests for Phase 1-2: orderbook recorder (orderbook.py) + liquidity gates.

  1. OrderBookRecorder message intake: apply_depth builds the book,
     apply_trade builds the rolling tape, snapshot() computes metrics.
  2. Strategy liquidity entry gate: blocks on stale/spread/depth/vanish/
     aggressor/intensity; fail-open semantics.
  3. Strategy liquidity exit gate: blocks on spread/depth.

orderbook.py / strategy.py live at the repo root and use bare sibling imports.
Heavy imports stay inside fixtures/test bodies so this file never perturbs
sibling tests in the same pytest process. Config values are concrete
SimpleNamespace values, never MagicMock (unset attrs would auto-create
truthy MagicMocks and silently enable features).
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

repo_root = Path(__file__).resolve().parent.parent


# ── Config with production-like liquidity thresholds ───────────────────
def _cfg(**overrides):
    cfg = types.SimpleNamespace(
        GATE_LIQUIDITY=True,
        GATE_LIQUIDITY_EXIT=True,
        LIQUIDITY_FAIL_OPEN_ON_STALE=True,
        LIQ_MAX_SPREAD_BPS=10.0,
        LIQ_MIN_DEPTH_USD=50000.0,
        LIQ_MAX_VANISH_PCT=60.0,
        LIQ_MIN_AGGRESSOR_RATIO=0.35,
        LIQ_MAX_AGGRESSOR_RATIO=0.65,
        LIQ_MIN_TRADE_INTENSITY_USD=50000.0,
        LIQ_EXIT_SPREAD_BPS=20.0,
        LIQ_EXIT_MIN_DEPTH_USD=25000.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def strategy_module():
    sys.path.insert(0, str(repo_root))
    import strategy as strategy_mod
    return strategy_mod


@pytest.fixture
def make_strat(strategy_module):
    def _make(cfg, monitor):
        broker = MagicMock()
        broker.symbol = "BTCUSDT"
        strat = strategy_module.DualMLStrategy(
            broker=broker,
            config=cfg,
            tactical_model=MagicMock(),
            strategic_model=MagicMock(),
            feature_cols=["close"],
            liquidity_monitor=monitor,
        )
        return strat
    return _make


# ── Recorder message intake ────────────────────────────────────────────
def test_recorder_apply_depth_removes_zero_qty_levels():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder()
    rec.apply_depth({"e": "depthUpdate", "b": [["100.0", "2.0"], ["99.5", "0.0"]],
                     "a": [["101.0", "3.0"], ["101.5", "0.0"]]})
    snap = rec.snapshot()
    assert snap["best_bid"] == 100.0
    assert snap["best_ask"] == 101.0
    assert 99.5 not in rec._bids
    assert 101.5 not in rec._asks


def test_recorder_snapshot_none_before_any_data():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder()
    assert rec.snapshot() is None


def test_recorder_book_metrics_from_seed():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder()
    # mid = 100.0, span = 100 * 50bps / 10000 = 0.5 -> bids >= 99.5, asks <= 100.5
    rec._seed_book({
        "bids": [["100.0", "1000.0"], ["99.5", "1000.0"], ["99.0", "1000.0"]],
        "asks": [["100.0", "500.0"], ["100.5", "500.0"], ["101.0", "500.0"]],
    })
    snap = rec.snapshot()
    assert snap["mid_price"] == 100.0
    assert snap["depth_bid_usd"] == pytest.approx(199500.0)     # 100+99.5 levels
    assert snap["depth_ask_usd"] == pytest.approx(100250.0)     # 100+100.5 levels
    assert snap["depth_total_usd"] == pytest.approx(299750.0)


def test_recorder_spread_bps():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder()
    rec._seed_book({"bids": [["99.9", "10.0"]], "asks": [["100.1", "10.0"]]})
    snap = rec.snapshot()
    assert snap["spread_bps"] == pytest.approx(20.0, rel=1e-6)  # 0.2 / 100 * 10000


def test_recorder_trade_tape_metrics():
    from orderbook import OrderBookRecorder
    import orderbook as ob_mod
    rec = OrderBookRecorder(window_sec=60.0)
    rec._seed_book({"bids": [["100.0", "1000.0"]], "asks": [["101.0", "1000.0"]]})

    now = 1_700_000_000.0
    rec._lock.acquire()
    try:
        rec._tape.append((now - 5.0, 100.0, 2.0, True))    # buyer=maker -> passive, sell-ish
        rec._tape.append((now - 2.0, 100.0, 2.0, False))   # buyer=taker -> aggressive buy
        rec._tape.append((now - 1.0, 101.0, 4.0, False))   # buyer=taker -> aggressive buy
        rec._last_msg_ts = now
    finally:
        rec._lock.release()

    with patch.object(ob_mod.time, "time", return_value=now):
        snap = rec.snapshot()
    total = 2.0 * 100.0 + 2.0 * 100.0 + 4.0 * 101.0
    assert snap["trade_intensity_usd_s"] == pytest.approx(total / 5.0)
    # aggressive buys = the two taker (m=False) trades: 200 + 404 = 604
    assert snap["aggressor_buy_ratio"] == pytest.approx(604.0 / total)
    assert snap["fresh"] is True


def test_recorder_stale_snapshot():
    from orderbook import OrderBookRecorder
    import orderbook as ob_mod
    rec = OrderBookRecorder(stale_after_s=15.0)
    rec._seed_book({"bids": [["100.0", "1000.0"]], "asks": [["101.0", "1000.0"]]})
    now = 1_700_000_000.0
    rec._last_msg_ts = now - 20.0
    with patch.object(ob_mod.time, "time", return_value=now):
        snap = rec.snapshot()
    assert snap["fresh"] is False


def test_recorder_vanish_pct():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder()
    rec._seed_book({"bids": [["100.0", "10.0"]], "asks": [["101.0", "10.0"]]})
    rec._vanish_cur = {100.0, 101.0, 102.0}
    rec._vanish_prev = {100.0, 101.0, 102.0, 103.0, 104.0}
    assert rec._vanish_pct_locked() == pytest.approx(40.0)


# ── Stream fallback (aggTrade silent -> trade stream) ──────────────────
def test_recorder_current_url_defaults_to_aggtrade():
    from orderbook import OrderBookRecorder, WS_URL, WS_URL_FALLBACK
    rec = OrderBookRecorder()
    assert rec._current_url() == WS_URL
    assert "aggTrade" in rec._current_url()
    rec._on_fallback = True
    assert rec._current_url() == WS_URL_FALLBACK
    assert "btcusdt@trade" in rec._current_url()


def test_recorder_no_switch_when_trades_fresh():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder(fallback_after_s=30.0)
    rec._stream_started_ts = 1000.0
    rec._last_trade_ts = 1000.0
    assert rec._maybe_switch_stream() is False
    assert rec._on_fallback is False


def test_recorder_switches_to_fallback_on_agg_silence():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder(fallback_after_s=30.0)
    rec._stream_started_ts = 1000.0
    rec._last_trade_ts = 1000.0  # trades stopped at t=1000
    with patch.object(rec, "_fallback_started_ts", 1000.0), \
         patch.object(__import__("orderbook"), "time") as mock_time:
        mock_time.time.return_value = 1031.0
        switched = rec._maybe_switch_stream()
    assert switched is True
    assert rec._on_fallback is True


def test_recorder_no_switch_within_silence_window():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder(fallback_after_s=30.0)
    rec._stream_started_ts = 1000.0
    rec._last_trade_ts = 1000.0
    with patch.object(__import__("orderbook"), "time") as mock_time:
        mock_time.time.return_value = 1020.0  # 20s silence < 30s threshold
        switched = rec._maybe_switch_stream()
    assert switched is False
    assert rec._on_fallback is False


def test_recorder_fallback_resets_on_trade():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder()
    rec._seed_book({"bids": [["100.0", "10.0"]], "asks": [["101.0", "10.0"]]})
    rec.apply_trade({"e": "trade", "p": "100.5", "q": "1.0", "m": False})
    assert rec._last_trade_ts > 0
    assert len(rec._tape) == 1


def test_recorder_reprobe_aggtrade_after_interval():
    from orderbook import OrderBookRecorder
    import orderbook as ob_mod
    rec = OrderBookRecorder(reprobe_after_s=3600.0)
    rec._on_fallback = True
    rec._fallback_started_ts = 1000.0
    assert rec._maybe_switch_stream() is False  # 0s on fallback
    with patch.object(ob_mod.time, "time", return_value=4600.0):
        switched = rec._maybe_switch_stream()
    assert switched is True
    assert rec._on_fallback is False


# ── Strategy entry gate ────────────────────────────────────────────────
def _fresh_snapshot(**overrides):
    snap = {
        "fresh": True,
        "spread_bps": 1.0,
        "depth_ask_usd": 200000.0,
        "depth_bid_usd": 200000.0,
        "depth_total_usd": 400000.0,
        "vanish_pct": 5.0,
        "aggressor_buy_ratio": 0.5,
        "trade_intensity_usd_s": 200000.0,
    }
    snap.update(overrides)
    return snap


class _FakeMonitor:
    def __init__(self, snap):
        self._snap = snap

    def snapshot(self):
        return self._snap


def test_entry_gate_pass_when_book_healthy(strategy_module, make_strat):
    strat = make_strat(_cfg(), _FakeMonitor(_fresh_snapshot()))
    assert strat._liquidity_entry_gate("long") is None


def test_entry_gate_blocks_on_spread(strategy_module, make_strat):
    strat = make_strat(_cfg(), _FakeMonitor(_fresh_snapshot(spread_bps=15.0)))
    assert strat._liquidity_entry_gate("long") == "spread"


def test_entry_gate_side_specific_depth(strategy_module, make_strat):
    cfg = _cfg(LIQ_MIN_DEPTH_USD=100000.0)
    strat = make_strat(cfg, _FakeMonitor(_fresh_snapshot(depth_ask_usd=50000.0,
                                                         depth_bid_usd=150000.0)))
    assert strat._liquidity_entry_gate("long") == "depth"   # thin ask side
    assert strat._liquidity_entry_gate("short") is None     # bid side is deep


def test_entry_gate_aggressor_direction(strategy_module, make_strat):
    strat = make_strat(_cfg(), _FakeMonitor(_fresh_snapshot(aggressor_buy_ratio=0.90)))
    assert strat._liquidity_entry_gate("short") == "aggressor"  # buy-side dominated
    assert strat._liquidity_entry_gate("long") is None

    cfg = _cfg(LIQ_MIN_AGGRESSOR_RATIO=0.40, LIQ_MAX_AGGRESSOR_RATIO=0.60)
    strat2 = make_strat(cfg, _FakeMonitor(_fresh_snapshot(aggressor_buy_ratio=0.10)))
    assert strat2._liquidity_entry_gate("long") == "aggressor"  # sell-side dominated


def test_entry_gate_blocks_on_vanish_and_intensity(strategy_module, make_strat):
    cfg = _cfg(LIQ_MAX_VANISH_PCT=50.0, LIQ_MIN_TRADE_INTENSITY_USD=300000.0)
    strat = make_strat(cfg, _FakeMonitor(_fresh_snapshot(vanish_pct=70.0)))
    assert strat._liquidity_entry_gate("long") == "vanish"
    strat2 = make_strat(cfg, _FakeMonitor(_fresh_snapshot(trade_intensity_usd_s=1000.0)))
    assert strat2._liquidity_entry_gate("long") == "intensity"


def test_entry_gate_fail_open_on_stale_and_missing(strategy_module, make_strat):
    # Stale book + fail-open -> gate does not block
    stale = _fresh_snapshot(fresh=False)
    strat = make_strat(_cfg(), _FakeMonitor(stale))
    assert strat._liquidity_entry_gate("long") is None

    # stale book + fail-closed -> blocks
    strat2 = make_strat(_cfg(LIQUIDITY_FAIL_OPEN_ON_STALE=False), _FakeMonitor(stale))
    assert strat2._liquidity_entry_gate("long") == "stale_book"

    # No monitor at all -> never blocks
    strat3 = make_strat(_cfg(), None)
    assert strat3._liquidity_entry_gate("long") is None

    # No data yet (None snapshot) + fail-closed -> blocks
    strat4 = make_strat(_cfg(LIQUIDITY_FAIL_OPEN_ON_STALE=False), _FakeMonitor(None))
    assert strat4._liquidity_entry_gate("long") == "stale_book"


# ── Strategy exit gate ─────────────────────────────────────────────────
def test_exit_gate_pass_when_book_healthy(strategy_module, make_strat):
    strat = make_strat(_cfg(), _FakeMonitor(_fresh_snapshot()))
    assert strat._liquidity_exit_gate() is None


def test_exit_gate_blocks_on_spread(strategy_module, make_strat):
    cfg = _cfg(LIQ_EXIT_SPREAD_BPS=20.0)
    strat = make_strat(cfg, _FakeMonitor(_fresh_snapshot(spread_bps=30.0)))
    assert strat._liquidity_exit_gate() == "liq_exit"


def test_exit_gate_blocks_on_thin_depth(strategy_module, make_strat):
    cfg = _cfg(LIQ_EXIT_MIN_DEPTH_USD=25000.0)
    strat = make_strat(cfg, _FakeMonitor(_fresh_snapshot(depth_total_usd=10000.0)))
    assert strat._liquidity_exit_gate() == "liq_exit"


def test_exit_gate_never_fires_on_stale_or_missing(strategy_module, make_strat):
    strat = make_strat(_cfg(), _FakeMonitor(_fresh_snapshot(fresh=False)))
    assert strat._liquidity_exit_gate() is None
    strat2 = make_strat(_cfg(), _FakeMonitor(None))
    assert strat2._liquidity_exit_gate() is None
    strat3 = make_strat(_cfg(), None)
    assert strat3._liquidity_exit_gate() is None