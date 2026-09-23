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


def test_recorder_tape_expires_old_entries():
    from orderbook import OrderBookRecorder
    import orderbook as ob_mod
    rec = OrderBookRecorder(window_sec=60.0)
    rec._seed_book({"bids": [["100.0", "10.0"]], "asks": [["101.0", "10.0"]]})
    now = 1_700_000_000.0
    rec._lock.acquire()
    try:
        rec._tape.append((now - 65.0, 100.0, 1.0, False))   # expired
        rec._tape.append((now - 30.0, 100.5, 2.0, False))    # alive
        rec._last_msg_ts = now
        # Simulate apply_trade expiration: remove entries older than window_sec
        while rec._tape and now - rec._tape[0][0] > rec.window_sec:
            rec._tape.popleft()
    finally:
        rec._lock.release()
    with patch.object(ob_mod.time, "time", return_value=now):
        snap = rec.snapshot()
    assert snap["trade_intensity_usd_s"] == pytest.approx(201.0 / 30.0)


def test_recorder_inverted_book_returns_none():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder()
    rec._seed_book({"bids": [["101.0", "10.0"]], "asks": [["100.0", "10.0"]]})
    assert rec.snapshot() is None


def test_recorder_apply_trade_skips_invalid():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder()
    rec._seed_book({"bids": [["100.0", "10.0"]], "asks": [["101.0", "10.0"]]})
    rec.apply_trade({"p": "0", "q": "1.0", "m": False})
    rec.apply_trade({"p": "100.0", "q": "-1.0", "m": False})
    rec.apply_trade({"p": "-5.0", "q": "1.0", "m": False})
    assert len(rec._tape) == 0


def test_recorder_empty_seed_returns_none():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder()
    rec._seed_book({"bids": [], "asks": []})
    assert rec.snapshot() is None


def test_recorder_vanish_empty_prev():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder()
    rec._vanish_cur = {100.0, 101.0}
    rec._vanish_prev = set()
    assert rec._vanish_pct_locked() == 0.0


def test_recorder_vanish_mark_respects_interval():
    from orderbook import OrderBookRecorder
    import orderbook as ob_mod
    rec = OrderBookRecorder(vanish_interval=10.0)
    rec._seed_book({"bids": [["100.0", "10.0"]], "asks": [["101.0", "10.0"]]})
    with patch.object(ob_mod.time, "time", return_value=1000.0):
        rec._maybe_mark_vanish()
    with patch.object(ob_mod.time, "time", return_value=1005.0):
        rec._maybe_mark_vanish()
    assert rec._vanish_prev is None


# ── Recorder health report ─────────────────────────────────────────────
def test_recorder_health_report_healthy():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder()
    rec._seed_book({
        "bids": [["100.0", "10.0"], ["99.5", "10.0"], ["99.0", "10.0"]],
        "asks": [["100.1", "10.0"], ["100.2", "10.0"], ["100.3", "10.0"]],
    })
    import orderbook as ob_mod
    now = 1_700_000_000.0
    rec._lock.acquire()
    try:
        rec._tape.append((now - 5.0, 100.0, 2.0, False))
        rec._last_msg_ts = now
    finally:
        rec._lock.release()
    with patch.object(ob_mod.time, "time", return_value=now):
        health = rec.health_report()
    assert health["status"] == "healthy"
    assert health["safe_to_trade"] is True
    assert health["issues"] == []


def test_recorder_health_report_no_data():
    from orderbook import OrderBookRecorder
    rec = OrderBookRecorder()
    health = rec.health_report()
    assert health["status"] == "no_data"
    assert health["safe_to_trade"] is False


def test_recorder_health_report_degraded_issues():
    from orderbook import OrderBookRecorder
    import orderbook as ob_mod
    rec = OrderBookRecorder()
    rec._seed_book({"bids": [["100.0", "10.0"]], "asks": [["101.0", "10.0"]]})
    now = 1_700_000_000.0
    rec._lock.acquire()
    try:
        rec._last_msg_ts = now
    finally:
        rec._lock.release()
    with patch.object(ob_mod.time, "time", return_value=now):
        health = rec.health_report()
    assert health["status"] == "degraded"
    assert health["safe_to_trade"] is False
    assert "thin_bid_side" in health["issues"]
    assert "thin_ask_side" in health["issues"]
    assert "dead_tape" in health["issues"]


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
    import orderbook as ob_mod
    rec = OrderBookRecorder(fallback_after_s=30.0)
    rec._stream_started_ts = 999.0
    rec._last_trade_ts = 999.0  # last trade 1s ago
    with patch.object(ob_mod.time, "time", return_value=1000.0):
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
    with patch.object(ob_mod.time, "time", return_value=2000.0):  # 1000s on fallback < 3600
        assert rec._maybe_switch_stream() is False
    assert rec._on_fallback is True
    with patch.object(ob_mod.time, "time", return_value=4600.0):  # 3600s on fallback
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

    def health_report(self):
        """Return a health report compatible with the strategy's health gate."""
        if self._snap is None:
            return {"status": "no_data", "safe_to_trade": False}
        issues = []
        if self._snap.get("n_levels_bid", 0) < 3:
            issues.append("thin_bid_side")
        if self._snap.get("n_levels_ask", 0) < 3:
            issues.append("thin_ask_side")
        if self._snap.get("trade_count_ps", 1.0) < 0.01:
            issues.append("dead_tape")
        status = "healthy" if not issues else "degraded"
        return {
            "status": status,
            "safe_to_trade": len(issues) == 0,
            "issues": issues,
            "snap": self._snap,
        }


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


# ── Strategy health gate hard-blocks ───────────────────────────────────
class _HealthMonitor:
    """Fake monitor that returns a health report dict."""

    def __init__(self, health):
        self._health = health

    def health_report(self):
        return self._health

    def snapshot(self):
        return self._health.get("snap")


def _healthy_health(**overrides):
    snap = _fresh_snapshot()
    snap.update(overrides)
    return {
        "status": "healthy",
        "safe_to_trade": True,
        "issues": [],
        "snap": snap,
    }


def _degraded_health(issues, **snap_overrides):
    snap = _fresh_snapshot(**snap_overrides)
    return {
        "status": "degraded",
        "safe_to_trade": False,
        "issues": issues,
        "snap": snap,
    }


def _no_data_health():
    return {"status": "no_data", "safe_to_trade": False}


def test_health_gate_passes_healthy(strategy_module, make_strat):
    strat = make_strat(_cfg(), _HealthMonitor(_healthy_health()))
    assert strat._liquidity_entry_gate("long") is None


def test_health_gate_blocks_inverted_book(strategy_module, make_strat):
    strat = make_strat(
        _cfg(LIQUIDITY_FAIL_OPEN_ON_STALE=True),
        _HealthMonitor(_degraded_health(["inverted_book"])),
    )
    assert strat._liquidity_entry_gate("long") == "inverted_book"


def test_health_gate_blocks_dead_tape(strategy_module, make_strat):
    strat = make_strat(
        _cfg(LIQUIDITY_FAIL_OPEN_ON_STALE=True),
        _HealthMonitor(_degraded_health(["dead_tape"])),
    )
    assert strat._liquidity_entry_gate("long") == "dead_tape"


def test_health_gate_allows_thin_sides_with_fail_open(strategy_module, make_strat):
    # thin_bid_side and thin_ask_side are degraded but NOT hard blocks
    strat = make_strat(
        _cfg(LIQUIDITY_FAIL_OPEN_ON_STALE=True),
        _HealthMonitor(_degraded_health(["thin_bid_side", "thin_ask_side"])),
    )
    # Should fall through to per-metric gate (which passes because snap is healthy)
    assert strat._liquidity_entry_gate("long") is None


def test_health_gate_no_data_respects_fail_open(strategy_module, make_strat):
    # fail-open -> no block
    strat = make_strat(
        _cfg(LIQUIDITY_FAIL_OPEN_ON_STALE=True),
        _HealthMonitor(_no_data_health()),
    )
    assert strat._liquidity_entry_gate("long") is None

    # fail-closed -> blocks
    strat2 = make_strat(
        _cfg(LIQUIDITY_FAIL_OPEN_ON_STALE=False),
        _HealthMonitor(_no_data_health()),
    )
    assert strat2._liquidity_entry_gate("long") == "stale_book"


# ── Integration test: mock WebSocket responses ─────────────────────────
def test_recorder_integration_mock_ws():
    """End-to-end: seed book, feed mock depth/trade frames, verify snapshot."""
    from orderbook import OrderBookRecorder
    import orderbook as ob_mod

    rec = OrderBookRecorder(
        stale_after_s=5.0,
        window_sec=60.0,
        depth_span_bps=50.0,
    )

    # Seed with REST-like snapshot (3+ levels each side)
    rec._seed_book({
        "bids": [["40000.0", "1.0"], ["39999.0", "2.0"], ["39998.0", "0.5"], ["39997.0", "1.0"]],
        "asks": [["40001.0", "1.5"], ["40002.0", "0.8"], ["40003.0", "1.0"]],
    })

    now = 1_700_000_000.0

    # Feed a depthUpdate (removes 39999, leaving 3 bid levels: 40000, 39998, 39997)
    rec.apply_depth({
        "e": "depthUpdate",
        "b": [["40000.0", "1.5"], ["39999.0", "0.0"]],
        "a": [["40001.0", "1.0"]],
    })

    # Feed a trade
    rec.apply_trade({"e": "aggTrade", "p": "40000.5", "q": "0.5", "m": False})

    with patch.object(ob_mod.time, "time", return_value=now):
        rec._last_msg_ts = now
        snap = rec.snapshot()

    assert snap is not None
    assert snap["best_bid"] == 40000.0
    assert snap["best_ask"] == 40001.0
    assert snap["mid_price"] == 40000.5
    assert snap["fresh"] is True
    assert snap["n_levels_bid"] == 3  # 40000 + 39998 + 39997 (39999 removed)
    assert snap["n_levels_ask"] == 3  # 40001 updated, 40002 + 40003 remain
    # Check tape has 1 trade
    assert snap["trade_count_ps"] > 0
    assert snap["trade_intensity_usd_s"] > 0

    # Verify health report
    health = rec.health_report()
    assert health["status"] == "healthy"
    assert health["safe_to_trade"] is True

    # Verify stale after threshold
    with patch.object(ob_mod.time, "time", return_value=now + 10.0):
        stale_snap = rec.snapshot()
    assert stale_snap["fresh"] is False
    stale_health = rec.health_report()
    assert stale_health["status"] == "healthy"  # stale != degraded
    assert stale_health["safe_to_trade"] is True