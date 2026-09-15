"""
Orderbook recorder + liquidity metrics (Phase 1-2, Option B).

Subscribes to the Binance MAINNET futures public WebSocket streams
`btcusdt@depth@100ms` (diff orderbook) and `btcusdt@aggTrade` (aggregate
trades) -- no API keys required. Maintains a local orderbook and a rolling
trade tape; `snapshot()` returns thread-safe liquidity metrics consumed by
DualMLStrategy's liquidity gates in live mode.

Stream fallback: fstream occasionally connects `btcusdt@aggTrade` but delivers
zero trade frames (observed on the ARM server, probe-proven). When no trade
event arrives within `fallback_after_s`, the recorder reconnects to
`btcusdt@trade` -- same `p/q/m` payload schema, and proven to stream normally.
While on the fallback, it re-probes `aggTrade` every `reprobe_after_s` so the
primary stream resumes automatically when it recovers.

The live bot may trade on testnet (debug) while this recorder streams real
mainnet liquidity ("Option B" per the approved plan): the liquidity signal is
real regardless of the trading venue, so gates are validated against actual
market microstructure instead of synthetic testnet data.

Fail-open design: `snapshot()` returns None (no book yet) or a snapshot with
`fresh == False` (stale stream) instead of raising, so a recorder outage never
halts the trading loop -- the strategy gates merely log and proceed.
"""

import asyncio
import json
import threading
import time
import urllib.request
from collections import deque
from typing import Any, Deque, Dict, Optional, Set, Tuple

import websockets

from logger import log_info, log_warning, log_error

WS_URL = (
    "wss://fstream.binance.com/stream?streams="
    "btcusdt@depth@100ms/btcusdt@aggTrade"
)
WS_URL_FALLBACK = (
    "wss://fstream.binance.com/stream?streams="
    "btcusdt@depth@100ms/btcusdt@trade"
)
REST_DEPTH_URL = "https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=100"


def _fetch_snapshot_rest(timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(REST_DEPTH_URL, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        log_warning(f"OrderBook REST snapshot failed: {exc}")
        return None


class OrderBookRecorder(threading.Thread):
    """Background recorder of the mainnet BTCUSDT futures orderbook + trades."""

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        depth_span_bps: float = 50.0,
        window_sec: float = 60.0,
        stale_after_s: float = 15.0,
        top_n_vanish: int = 25,
        vanish_interval: float = 10.0,
        url: Optional[str] = None,
        fallback_url: Optional[str] = None,
        fallback_after_s: float = 30.0,
        reprobe_after_s: float = 3600.0,
    ) -> None:
        super().__init__(daemon=True, name="OrderBookRecorder")
        self.symbol = symbol
        self.depth_span_bps = depth_span_bps
        self.window_sec = window_sec
        self.stale_after_s = stale_after_s
        self.top_n_vanish = top_n_vanish
        self.vanish_interval = vanish_interval
        self._url = url or WS_URL
        self._url_fallback = fallback_url or WS_URL_FALLBACK
        self._fallback_after_s = fallback_after_s
        self._reprobe_after_s = reprobe_after_s
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}
        self._tape: Deque[Tuple[float, float, float, bool]] = deque()
        self._last_msg_ts: float = 0.0
        self._last_trade_ts: float = 0.0
        self._stream_started_ts: float = 0.0
        self._fallback_started_ts: float = 0.0
        self._on_fallback: bool = False
        self._vanish_cur: Optional[Set[float]] = None
        self._vanish_prev: Optional[Set[float]] = None
        self._vanish_mark_ts: float = 0.0
        self._reconnects: int = 0
        self._connected: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────
    def run(self) -> None:
        asyncio.run(self._async_run())

    def stop(self) -> None:
        self._stop.set()

    # ── Message intake (called from the WS loop and from tests) ─────────
    def apply_depth(self, data: Dict[str, Any]) -> None:
        """Apply a depthUpdate event: bids `b` / asks `a` arrays of [price, qty]."""
        with self._lock:
            for p, q in data.get("b", []):
                p, q = float(p), float(q)
                if q > 0:
                    self._bids[p] = q
                else:
                    self._bids.pop(p, None)
            for p, q in data.get("a", []):
                p, q = float(p), float(q)
                if q > 0:
                    self._asks[p] = q
                else:
                    self._asks.pop(p, None)

    def apply_trade(self, data: Dict[str, Any]) -> None:
        """Append a trade event (aggTrade or trade) to the rolling tape."""
        price, qty = float(data.get("p", 0.0)), float(data.get("q", 0.0))
        maker = bool(data.get("m", False))
        if price <= 0 or qty <= 0:
            return
        now = time.time()
        with self._lock:
            self._tape.append((now, price, qty, maker))
            self._last_trade_ts = now
            while self._tape and now - self._tape[0][0] > self.window_sec:
                self._tape.popleft()

    def _current_url(self) -> str:
        """Return the URL for the currently active stream."""
        return self._url_fallback if self._on_fallback else self._url

    def _maybe_switch_stream(self) -> bool:
        """Switch to the fallback stream when aggTrade goes silent; after a
        long fallback period, re-probe aggTrade to self-heal.

        Returns True when the stream selection changed (caller must reconnect).
        """
        now = time.time()
        if self._on_fallback:
            if now - self._fallback_started_ts >= self._reprobe_after_s:
                self._on_fallback = False
                log_info("OrderBookRecorder re-probing aggTrade stream")
                return True
            return False
        silent_for = now - max(self._stream_started_ts, self._last_trade_ts)
        if silent_for >= self._fallback_after_s and silent_for > 0:
            self._on_fallback = True
            self._fallback_started_ts = now
            log_warning(
                f"OrderBookRecorder aggTrade silent {silent_for:.0f}s "
                f"> {self._fallback_after_s:.0f}s, falling back to trade stream"
            )
            return True
        return False

    def _seed_book(self, snap: Dict[str, Any]) -> None:
        bids, asks = {}, {}
        for p, q in snap.get("bids", []):
            q = float(q)
            if q > 0:
                bids[float(p)] = q
        for p, q in snap.get("asks", []):
            q = float(q)
            if q > 0:
                asks[float(p)] = q
        if bids or asks:
            with self._lock:
                self._bids, self._asks = bids, asks

    # ── Metrics ─────────────────────────────────────────────────────────
    def snapshot(self) -> Optional[Dict[str, Any]]:
        """Thread-safe liquidity metrics, or None if no book data yet."""
        now = time.time()
        with self._lock:
            if not self._bids and not self._asks:
                return None
            bids = sorted(self._bids.items(), key=lambda kv: -kv[0])
            asks = sorted(self._asks.items(), key=lambda kv: kv[0])
            best_bid = bids[0][0] if bids else 0.0
            best_ask = asks[0][0] if asks else 0.0
            mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
            if mid <= 0:
                return None
            span = mid * self.depth_span_bps / 10000.0
            depth_bid = sum(p * q for p, q in bids if p >= mid - span)
            depth_ask = sum(p * q for p, q in asks if p <= mid + span)
            spread_bps = (best_ask - best_bid) / mid * 10000.0 if best_ask > best_bid else 0.0
            tape = list(self._tape)
            vanished = self._vanish_pct_locked()
            last_msg = self._last_msg_ts
        total_notional = sum(p * q for _, p, q, _ in tape)
        buy_notional = sum(p * q for _, p, q, m in tape if not m)
        if tape:
            span_sec = max(now - tape[0][0], 0.001)
            trade_intensity = total_notional / span_sec
            trade_count_ps = len(tape) / span_sec
        else:
            trade_intensity, trade_count_ps = 0.0, 0.0
        return {
            "ts": now,
            "fresh": bool(last_msg) and (now - last_msg) <= self.stale_after_s,
            "updated_ago_s": now - last_msg if last_msg else float("inf"),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_bps": spread_bps,
            "depth_bid_usd": depth_bid,
            "depth_ask_usd": depth_ask,
            "depth_total_usd": depth_bid + depth_ask,
            "vanish_pct": vanished,
            "aggressor_buy_ratio": (buy_notional / total_notional) if total_notional > 0 else 0.5,
            "trade_intensity_usd_s": trade_intensity,
            "trade_count_ps": trade_count_ps,
            "n_levels_bid": len(bids),
            "n_levels_ask": len(asks),
        }

    def _vanish_pct_locked(self) -> float:
        cur, prev = self._vanish_cur, self._vanish_prev
        if cur is None or prev is None or not prev:
            return 0.0
        return min(100.0, len(prev - cur) / len(prev) * 100.0)

    def _maybe_mark_vanish(self) -> None:
        now = time.time()
        if now - self._vanish_mark_ts < self.vanish_interval:
            return
        with self._lock:
            bids = sorted(self._bids.keys(), reverse=True)[: self.top_n_vanish]
            asks = sorted(self._asks.keys())[: self.top_n_vanish]
            cur = set(bids) | set(asks)
            if cur:
                self._vanish_prev = self._vanish_cur
                self._vanish_cur = cur
                self._vanish_mark_ts = now

    # ── Async WS loop (runs in this thread's own event loop) ───────────
    async def _async_run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                if not self._bids and not self._asks:
                    self._seed_book(_fetch_snapshot_rest())
                url = self._current_url()
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=20, close_timeout=5
                ) as ws:
                    self._connected = True
                    backoff = 1.0
                    self._stream_started_ts = time.time()
                    log_info(f"OrderBookRecorder connected: {url}")
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        data = msg.get("data") if isinstance(msg, dict) else msg
                        evt = data.get("e") if isinstance(data, dict) else None
                        if evt == "depthUpdate":
                            self.apply_depth(data)
                        elif evt in ("aggTrade", "trade"):
                            self.apply_trade(data)
                        self._last_msg_ts = time.time()
                        self._maybe_mark_vanish()
                        if self._maybe_switch_stream():
                            break
            except Exception as exc:
                if self._stop.is_set():
                    break
                self._connected = False
                self._reconnects += 1
                log_warning(
                    f"OrderBookRecorder stream error: {exc} "
                    f"(reconnect {self._reconnects} in {backoff:.0f}s)"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
        self._connected = False