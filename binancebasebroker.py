import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import pandas as pd
from binance.client import Client
from binance.enums import *
import time

MIN_TRADEABLE_QUANTITY = 0.001
TRADEABLE_QUANTITY_PRECISION = 3
SIGNAL_LONG = "long"
SIGNAL_SHORT = "short"
SIGNAL_HOLD = "hold"
MARKET_TYPE_SPOT = "spot"
MARKET_TYPE_FUTURES = "futures"

# ── Per-endpoint weight estimates (Binance Futures docs) ──────────────
# Used by the rate limiter so we don't over-reserve for cheap endpoints.
ENDPOINT_WEIGHTS: Dict[str, int] = {
    "futures_account_balance": 25,
    "futures_position_information": 10,
    "futures_symbol_ticker": 2,
    "futures_klines": 2,
    "futures_create_order": 1,
    "futures_change_leverage": 1,
    "futures_change_margin_type": 1,
    "futures_get_open_orders": 10,
    "futures_cancel_order": 1,
    # Spot endpoints
    "get_account": 25,
    "get_symbol_ticker": 2,
    "get_klines": 2,
    "create_order": 1,
    "create_oco_order": 1,
    "get_open_orders": 10,
    "cancel_order": 1,
}

# Default weight when we cannot match a specific endpoint
_DEFAULT_WEIGHT = 20

# Minimum gap (seconds) between consecutive API calls to avoid
# triggering Binance's per-second abuse detection.
_INTER_REQUEST_DELAY = 0.5


def _estimate_weight(uri: str, method: str = "GET") -> int:
    """Guess the Binance API weight for a given request URI + method."""
    for keyword, weight in ENDPOINT_WEIGHTS.items():
        if keyword in uri:
            return weight
    # POST requests tend to be heavier
    return _DEFAULT_WEIGHT + (10 if method.upper() == "POST" else 0)


@dataclass
class BracketResult:
    success: bool
    error: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionResult:
    amount: float
    entry_price: float


@dataclass
class MarketOrderResult:
    order_id: str
    entry_price: Optional[float]


@dataclass
class BracketOrderResult:
    tp_order_id: str
    sl_order_id: str


class BinanceRateLimiter:
    """Token-bucket rate limiter enforcing a max API weight per 60s window.

    Tracks the Binance ``x-mbx-used-weight-1m`` header to stay under
    the exchange's per-minute weight quota (default 1000, leaving
    headroom from the 1200 hard limit).  Callers should *acquire*
    an estimated weight cost *before* the request so the bucket is
    never overdrawn; after the response arrives they *set* the
    actual header value to keep the bucket in sync with the server.

    Also implements a circuit breaker that enters a **cooldown**
    period on ``-1003`` (rate-limit ban).  During cooldown *every*
    subsequent ``acquire()`` blocks until the ban is expected to
    expire, preventing pointless retries that would keep the IP
    banned.

    .. note::

       All internal time tracking uses ``time.time()`` (wall clock)
       so that cooldown durations computed from Binance's server-side
       timestamps are directly comparable.  Using ``time.monotonic()``
       here would cause drift when the two clocks diverge (e.g. NTP
       adjustments or suspend/resume cycles).
    """

    def __init__(self, max_weight: int = 1000, window: int = 60):
        self.max_weight = max_weight
        self.window = window
        self._used = 0
        self._window_start: float = 0.0  # set on first acquire
        self._cooldown_until: float = 0.0
        self._ban_count: int = 0
        self._lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────

    def acquire(self, weight: int) -> None:
        """Block until ``weight`` units of budget are available.

        If a rate-limit cooldown is active, blocks until it expires
        first (checked outside the lock so the lock is never held
        during a long sleep).
        """
        self._wait_cooldown()
        with self._lock:
            self._reset_if_expired()
            while self._used + weight > self.max_weight:
                remaining = self.window - (time.time() - self._window_start)
                if remaining > 0:
                    time.sleep(min(remaining, 1.0))
                self._reset_if_expired()
            self._used += weight

    def set_used(self, weight: int) -> None:
        """Sync the bucket with the server-reported weight."""
        with self._lock:
            self._reset_if_expired()
            self._used = weight
            self._window_start = time.time()

    def enter_cooldown(self, ban_until_ms: int = 0) -> None:
        """Enter cooldown after a ``-1003`` (rate-limit ban).

        Parameters
        ----------
        ban_until_ms
            Server-reported ban expiration as a millisecond epoch
            timestamp (parsed from the error message).  When zero
            the cooldown duration uses exponential backoff instead.
        """
        with self._lock:
            now = time.time()
            # Exponential backoff capped at 5 minutes when there is no
            # server-provided ban_until.
            default_seconds = min(300.0, 30.0 * (2.0 ** self._ban_count))
            self._ban_count += 1

            if ban_until_ms > 0:
                ban_remaining = (ban_until_ms / 1000.0) - now
                cooldown = max(default_seconds, ban_remaining)
            else:
                cooldown = default_seconds

            self._cooldown_until = now + cooldown

    # ── internals ─────────────────────────────────────────────────────

    def _reset_if_expired(self) -> None:
        now = time.time()
        if now - self._window_start >= self.window:
            self._used = 0
            self._window_start = now

    def _wait_cooldown(self) -> None:
        now = time.time()
        if now < self._cooldown_until:
            remaining = self._cooldown_until - now
            logging.warning(
                "🧊 Rate-limit cooldown active — sleeping %.0fs "
                "(ban #%d)",
                remaining, self._ban_count,
            )
            time.sleep(remaining)


# ── shared position-data cache ────────────────────────────────────────
# Avoids redundant futures_position_information calls by caching the
# raw exchange response for a short TTL.

@dataclass
class _PositionData:
    amount: float
    entry_price: float
    leverage: Optional[int]
    liquidation_price: Optional[float]
    cached_at: float


class _PositionCache:
    """Thread-safe short-lived cache for exchange position data."""

    def __init__(self, ttl: float = 2.0):
        self._ttl = ttl
        self._lock = threading.Lock()
        self._data: Optional[_PositionData] = None

    def get(self) -> Optional[_PositionData]:
        with self._lock:
            if self._data is None:
                return None
            if time.time() - self._data.cached_at > self._ttl:
                self._data = None
                return None
            return self._data

    def set(self, data: _PositionData) -> None:
        with self._lock:
            self._data = data

    def invalidate(self) -> None:
        with self._lock:
            self._data = None


# ═══════════════════════════════════════════════════════════════════════
# Base broker
# ═══════════════════════════════════════════════════════════════════════

class BinanceBaseBroker(ABC):
    """Base broker interface for both spot and futures."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._cached_balance = None
        self._balance_cache_time = 0
        self._balance_cache_duration = 5
        # Klines cache: key=(symbol, timeframe) -> (DataFrame, fetch_timestamp)
        self._klines_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, float]] = {}
        # Shared position cache
        self._position_cache = _PositionCache(ttl=2.0)
        self.setup_logging()
        self.setup_client()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    # ── Rate limiter ──────────────────────────────────────────────────

    def _install_rate_limiter(self) -> None:
        """Wrap ``Client._request`` with rate limiter + -1003 circuit breaker.

        Also inserts a small inter-request delay (0.5 s) between calls
        so the bot never triggers Binance's per-second abuse detection.
        """
        import re as _re
        from binance.exceptions import BinanceAPIException
        from collections import deque
        import traceback

        limiter = BinanceRateLimiter(max_weight=1000)
        original_request = self.client._request
        _last_request_time: float = 0.0
        # Rolling log of recent requests: (timestamp, method, uri)
        _request_log: deque = deque(maxlen=500)
        _request_count = 0
        _last_summary_log: float = 0.0
        _SUMMARY_INTERVAL = 60.0  # log summary once per minute

        def _parse_ban_ms(msg: str) -> int:
            m = _re.search(r"banned until (\d+)", msg)
            return int(m.group(1)) if m else 0

        def _log_request_summary(now: float) -> None:
            """Log a summary of all API calls made in the last 60 seconds."""
            cutoff = now - 60.0
            recent = [(ts, m, u) for ts, m, u in _request_log if ts >= cutoff]
            if not recent:
                return
            # Count per endpoint
            counts: dict = {}
            for _, m, u in recent:
                key = f"{m} {u.split('?')[0]}"
                counts[key] = counts.get(key, 0) + 1
            total = len(recent)
            detail = " | ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            self.logger.info(
                "📊 API call summary (last 60s): %d total — %s",
                total, detail,
            )

        def _wrapped(method, uri, signed, force_params=False, **kwargs):
            nonlocal _last_request_time, _request_count, _last_summary_log

            now = time.time()
            since_last = now - _last_request_time

            # ── log every request ─────────────────────────────────
            _request_count += 1
            _request_log.append((now, method, uri))

            # Trim endpoint for readability — strip query params, keep last 80 chars
            short_uri = uri.split("?")[0].split("/")[-1] if "?" in uri else uri.split("/")[-1]
            self.logger.info(
                "📞 API call #%d: %s %s (since_last=%.1fs)",
                _request_count, method, short_uri, since_last,
            )

            # ── periodic summary ──────────────────────────────────
            if now - _last_summary_log >= _SUMMARY_INTERVAL:
                _log_request_summary(now)
                _last_summary_log = now

            # ── inter-request delay ──────────────────────────────
            # Prevents burst of requests from triggering -1003 abuse
            # detection.  The delay is skipped after a cooldown sleep
            # (which is already much longer) so we only enforce it
            # during normal operation.
            if since_last < _INTER_REQUEST_DELAY:
                self.logger.debug(
                    "⏳ Inter-request delay: sleeping %.1fs",
                    _INTER_REQUEST_DELAY - since_last,
                )
                time.sleep(_INTER_REQUEST_DELAY - since_last)

            # ── estimate weight ──────────────────────────────────
            weight = _estimate_weight(uri, method)

            # ── acquire budget ───────────────────────────────────
            limiter.acquire(weight=weight)

            try:
                t0 = time.time()
                result = original_request(method, uri, signed, force_params, **kwargs)
                elapsed = time.time() - t0
                resp = getattr(self.client, "response", None)
                server_weight = None
                if resp is not None:
                    h = resp.headers.get("x-mbx-used-weight-1m")
                    if h is not None:
                        try:
                            server_weight = int(h)
                            limiter.set_used(server_weight)
                        except ValueError:
                            pass
                self.logger.debug(
                    "✅ API response #%d: %s %s (%.1fms, weight=%s)",
                    _request_count, method, short_uri,
                    elapsed * 1000, server_weight or "?",
                )
                return result
            except BinanceAPIException as exc:
                elapsed = time.time() - t0 if 't0' in dir() else 0
                self.logger.error(
                    "❌ API error #%d: %s %s — code=%s (%.1fms)",
                    _request_count, method, short_uri,
                    exc.code if hasattr(exc, 'code') else '?',
                    elapsed * 1000,
                )
                if "-1003" in str(exc):
                    ban_ms = _parse_ban_ms(str(exc))
                    self.logger.error(
                        "🚨 Binance -1003 ban detected — entering cooldown "
                        "(ban_until=%s, countdown=%ds)",
                        ban_ms, max(0, (ban_ms / 1000) - time.time()) if ban_ms else 0,
                    )
                    limiter.enter_cooldown(ban_until_ms=ban_ms)
                raise
            finally:
                _last_request_time = time.time()

        self.client._request = _wrapped
        self.logger.info(
            "✅ BinanceRateLimiter installed (max %d weight/min, "
            "inter-request delay %.1fs, request logging enabled)",
            limiter.max_weight, _INTER_REQUEST_DELAY,
        )

    @abstractmethod
    def setup_client(self):
        pass

    # ── Timeframe helpers ─────────────────────────────────────────────

    @staticmethod
    def _parse_timeframe_to_minutes(timeframe: str) -> int:
        """Parse timeframe string ('5m', '1h', '15m') to minutes."""
        if not timeframe:
            return 5
        unit = timeframe[-1]
        try:
            value = int(timeframe[:-1])
        except ValueError:
            return 5
        if unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 1440
        return value  # assume minutes

    def _klines_cache_ttl(self, timeframe: str) -> float:
        """TTL in seconds for a given klines timeframe."""
        minutes = self._parse_timeframe_to_minutes(timeframe)
        return max(30.0, minutes * 0.8 * 60.0)

    # ── Abstract interface ────────────────────────────────────────────

    @abstractmethod
    def get_cash(self, quote_asset_symbol: str = "USDT") -> float:
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[PositionResult]:
        pass

    @abstractmethod
    def get_last_price(self, symbol: str) -> float:
        pass

    @abstractmethod
    def _create_market_order(self, symbol: str, side: str, quantity: float) -> Optional[MarketOrderResult]:
        pass

    @abstractmethod
    def _create_bracket_order(self, *args, **kwargs):
        pass

    @abstractmethod
    def cancel_open_orders(self, symbol: str, max_retries: int, base_delay: float):
        pass

    @abstractmethod
    def close_position(self, symbol: str, position: float):
        pass

    # ── Shared public methods ─────────────────────────────────────────

    def open_position_with_bracket(
        self,
        symbol: str,
        signal: str,
        quantity: float,
        tp_frac: float = 0.02,
        sl_frac: float = 0.01
    ) -> BracketResult:
        """Open a long or short position with TP/SL bracket."""
        if signal not in [SIGNAL_LONG, SIGNAL_SHORT]:
            return BracketResult(success=False, error="Invalid signal")

        market_order_side = SIDE_BUY if signal == SIGNAL_LONG else SIDE_SELL

        try:
            # 1️⃣ Market order
            order_result = self._create_market_order(symbol, market_order_side, quantity)
            if order_result is None:
                return BracketResult(success=False, error="Market order returned None")

            entry_price = order_result.entry_price
            if entry_price is None:
                entry_price = None
                for attempt in range(5):
                    time.sleep(0.2 * (2 ** attempt))
                    position = self.get_position(symbol)
                    if position and position.entry_price and position.entry_price > 0:
                        entry_price = position.entry_price
                        break

                if entry_price is None:
                    close_qty = quantity if signal == SIGNAL_LONG else -quantity
                    self.close_position(symbol, close_qty)
                    return BracketResult(success=False, error="Fill confirmation timeout after 5 attempts")

            if not order_result.order_id:
                return BracketResult(success=False, error="Market order returned no order_id")
            if entry_price is None or entry_price <= 0:
                return BracketResult(success=False, error="Market order returned invalid entry_price")

            # 2️⃣ TP/SL prices
            if market_order_side == SIDE_BUY:
                tp_price = round(entry_price * (1 + tp_frac), 2)
                sl_price = round(entry_price * (1 - sl_frac), 2)
            else:
                tp_price = round(entry_price * (1 - tp_frac), 2)
                sl_price = round(entry_price * (1 + sl_frac), 2)

            # 3️⃣ Place bracket orders
            bracket_order_result = self._create_bracket_order(
                symbol, quantity, market_order_side, tp_price, sl_price
            )

            # 4️⃣ TP/SL failure → close position
            if not bracket_order_result:
                position = self.get_position(symbol)
                if position and position.amount:
                    self.close_position(symbol, position.amount)
                return BracketResult(success=False, error="TP/SL placement failed; position closed")

            tp_id = bracket_order_result.tp_order_id
            sl_id = bracket_order_result.sl_order_id

            return BracketResult(
                success=True,
                data={
                    "order_id": order_result.order_id,
                    "entry_price": entry_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "tp_algo_id": tp_id,
                    "sl_algo_id": sl_id,
                }
            )

        except Exception as e:
            return BracketResult(success=False, error=str(e))

    # ── Historical prices ─────────────────────────────────────────────

    def get_historical_prices(
        self,
        symbol: str,
        length: int,
        timeframe: str = "5m"
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data with TTL caching."""
        cache_key = (symbol, timeframe)
        now = time.time()

        if cache_key in self._klines_cache:
            cached_df, cached_at = self._klines_cache[cache_key]
            ttl = self._klines_cache_ttl(timeframe)
            cache_age = now - cached_at
            self.logger.debug(
                "📦 get_historical_prices(%s, %s): cache age=%.0fs ttl=%.0fs len=%d need=%d",
                symbol, timeframe, cache_age, ttl, len(cached_df), length,
            )
            if cache_age < ttl and len(cached_df) >= length:
                self.logger.debug("📦 Cache HIT for %s %s (age=%.0fs)", symbol, timeframe, cache_age)
                return cached_df
            self.logger.info(
                "📦 Cache MISS for %s %s: age=%.0fs>=%.0fs or len=%d<%d — fetching via REST",
                symbol, timeframe, cache_age, ttl, len(cached_df), length,
            )
        else:
            self.logger.info("📦 Cache COLD for %s %s — fetching via REST", symbol, timeframe)

        try:
            interval_map = {
                "1m": Client.KLINE_INTERVAL_1MINUTE,
                "3m": Client.KLINE_INTERVAL_3MINUTE,
                "5m": Client.KLINE_INTERVAL_5MINUTE,
                "15m": Client.KLINE_INTERVAL_15MINUTE,
                "1h": Client.KLINE_INTERVAL_1HOUR,
                "1d": Client.KLINE_INTERVAL_1DAY,
            }

            interval = interval_map.get(timeframe, Client.KLINE_INTERVAL_5MINUTE)

            klines = self._fetch_klines(
                symbol=symbol,
                interval=interval,
                limit=length,
            )

            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)

            self._klines_cache[cache_key] = (df, time.time())
            return df

        except Exception as e:
            self.logger.error(f"❌ Error fetching historical prices for {symbol}: {e}")
            if cache_key in self._klines_cache:
                cached_df, _ = self._klines_cache[cache_key]
                self.logger.warning(f"⚠️ Returning stale cached data for {symbol} ({timeframe})")
                return cached_df
            return None

    def _fetch_klines(self, symbol: str, interval: str, limit: int):
        """Implemented by Spot / Futures brokers."""
        raise NotImplementedError

    # ── Position cache helpers ────────────────────────────────────────

    def _cache_position_data(self, symbol: str) -> Optional[_PositionData]:
        """Fetch and cache position data from the exchange.

        Subclasses that call ``futures_position_information`` should use
        this method instead of calling the endpoint directly so that
        both ``get_position()`` and ``get_position_leverage()`` share
        a single API call.
        """
        raise NotImplementedError

    # ── Other shared methods ──────────────────────────────────────────

    def get_liquidation_price(self, symbol: str) -> Optional[float]:
        """Return the current estimated liquidation price, or None."""
        return None

    def set_leverage(self, symbol: str, leverage: int, margin_type: str = "ISOLATED") -> bool:
        """Set leverage for a symbol. No-op for spot; implemented by futures broker."""
        return True

    def get_datetime(self) -> datetime:
        return datetime.now()

    def log_message(self, msg: str):
        self.logger.info(msg)
