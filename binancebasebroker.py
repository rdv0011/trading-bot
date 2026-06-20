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

    Tracks the Binance `x-mbx-used-weight-1m` header to stay under
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
    """

    def __init__(self, max_weight: int = 1000, window: int = 60):
        self.max_weight = max_weight
        self.window = window
        self._used = 0
        self._window_start = time.monotonic()
        self._cooldown_until: float = 0.0
        self._ban_count: int = 0
        self._lock = threading.Lock()

    def _reset_if_expired(self) -> None:
        now = time.monotonic()
        if now - self._window_start >= self.window:
            self._used = 0
            self._window_start = now

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
                remaining = self.window - (time.monotonic() - self._window_start)
                if remaining > 0:
                    time.sleep(min(remaining, 1.0))
                self._reset_if_expired()
            self._used += weight

    def _wait_cooldown(self) -> None:
        now = time.monotonic()
        if now < self._cooldown_until:
            remaining = self._cooldown_until - now
            logging.warning(
                "🧊 Rate-limit cooldown active — sleeping %.0fs "
                "(ban #%d)",
                remaining, self._ban_count,
            )
            time.sleep(remaining)

    def set_used(self, weight: int) -> None:
        """Sync the bucket with the server-reported weight."""
        with self._lock:
            self._reset_if_expired()
            self._used = weight
            self._window_start = time.monotonic()

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
            now = time.monotonic()
            default_seconds = min(300.0, 30.0 * (2.0 ** self._ban_count))
            self._ban_count += 1

            if ban_until_ms > 0:
                ban_remaining = (ban_until_ms / 1000.0) - time.time()
                cooldown = max(default_seconds, ban_remaining)
            else:
                cooldown = default_seconds

            self._cooldown_until = now + cooldown


class BinanceBaseBroker(ABC):
    """
    Base broker interface for both spot and futures
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._cached_balance = None
        self._balance_cache_time = 0
        self._balance_cache_duration = 5
        # Klines cache: key=(symbol, timeframe) -> (DataFrame, fetch_timestamp)
        self._klines_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, float]] = {}
        self._cached_position_leverage: Optional[int] = None
        self._cached_leverage_time: float = 0
        self.setup_logging()
        self.setup_client()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _install_rate_limiter(self) -> None:
        """Wrap Client._request with rate limiter + -1003 circuit breaker."""
        import re as _re
        from binance.exceptions import BinanceAPIException

        limiter = BinanceRateLimiter(max_weight=1000)
        original_request = self.client._request

        def _parse_ban_ms(msg: str) -> int:
            m = _re.search(r"banned until (\d+)", msg)
            return int(m.group(1)) if m else 0

        def _wrapped(method, uri, signed, force_params=False, **kwargs):
            limiter.acquire(weight=40)
            try:
                result = original_request(method, uri, signed, force_params, **kwargs)
                resp = getattr(self.client, "response", None)
                if resp is not None:
                    h = resp.headers.get("x-mbx-used-weight-1m")
                    if h is not None:
                        try:
                            limiter.set_used(int(h))
                        except ValueError:
                            pass
                return result
            except BinanceAPIException as exc:
                if "-1003" in str(exc):
                    ban_ms = _parse_ban_ms(str(exc))
                    self.logger.error(
                        "🚨 Binance -1003 ban detected — entering cooldown "
                        "(ban_until=%s, countdown=%ds)",
                        ban_ms, max(0, (ban_ms / 1000) - time.time()) if ban_ms else 0,
                    )
                    limiter.enter_cooldown(ban_until_ms=ban_ms)
                raise

        self.client._request = _wrapped
        self.logger.info("✅ BinanceRateLimiter installed (max 1000 weight/min)")

    @abstractmethod
    def setup_client(self):
        pass

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

    # shared public method for both brokers
    def open_position_with_bracket(
        self,
        symbol: str,
        signal: str,
        quantity: float,
        tp_frac: float = 0.02,
        sl_frac: float = 0.01
    ) -> BracketResult:
        """
        Open a long or short position with TP/SL bracket.
        Unified interface for spot & futures.
        """
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
                    # Use signed quantity: positive for LONG (sell), negative for SHORT (buy to cover)
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
            bracket_order_result = self._create_bracket_order(symbol, quantity, market_order_side, tp_price, sl_price)

            # 4️⃣ TP/SL failure → close position
            if not bracket_order_result:
                position = self.get_position(symbol)
                if position.amount:
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
                    "sl_algo_id": sl_id
                }
            )

        except Exception as e:
            return BracketResult(success=False, error=str(e))
        
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
            if now - cached_at < ttl and len(cached_df) >= length:
                return cached_df

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
        """
        Implemented by Spot / Futures brokers.
        """
        raise NotImplementedError

    def get_liquidation_price(self, symbol: str) -> Optional[float]:
        """Return the current estimated liquidation price from the exchange, or None if unavailable."""
        return None

    def set_leverage(self, symbol: str, leverage: int, margin_type: str = "ISOLATED") -> bool:
        """
        Set leverage for a symbol. No-op for spot; implemented by futures broker.
        Returns True on success, False otherwise.
        """
        return True

    def get_datetime(self) -> datetime:
        return datetime.now()

    def log_message(self, msg: str):
        self.logger.info(msg)