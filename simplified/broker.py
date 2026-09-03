"""
Broker interface for Dual-ML Bitcoin Trading Bot.
Supports: BinanceBroker (live) + MockBroker (simulation).
Based on original binancebasebroker.py and binancefuturesbroker.py
"""

import threading
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import re as _re

from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException

from config import (
    SYMBOL, TACTICAL_TF, STRATEGIC_TF,
    INITIAL_EQUITY, FEE, SLIPPAGE,
)

# ── Constants ───────────────────────────────────────────────────────────
MIN_TRADEABLE_QUANTITY = 0.001
TRADEABLE_QUANTITY_PRECISION = 3
SIGNAL_LONG = "long"
SIGNAL_SHORT = "short"
SIGNAL_HOLD = "hold"

# Logging setup (from original)
LOG_DIR = Path("logs")
LOG_RETENTION_DAYS = 10
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_MAX_LINES = 10_000
LOG_FILE_LEVEL = logging.DEBUG
LOG_CONSOLE_LEVEL = logging.INFO


def _today_log_path() -> Path:
    return LOG_DIR / f"trading_{datetime.now(timezone.utc):%Y-%m-%d}.log"


class DailyDatedLogHandler(logging.Handler):
    """Daily rotating log handler with date suffix (from original)."""

    def __init__(self, log_dir: Path, level: int = LOG_FILE_LEVEL):
        super().__init__(level=level)
        self.log_dir = log_dir
        self._stream = None

    @property
    def stream(self):
        return self._stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._stream is None:
                self._open_today()
            msg = self.format(record)
            if "\n" in msg:
                msg = msg.replace("\n", " | ")
            self._stream.write(msg + "\n")
            self.flush()
        except Exception:
            self.handleError(record)

    def _open_today(self) -> None:
        self._prune_old_files()
        self._stream = open(self._today(), "a", encoding="utf-8")

    def _today(self) -> Path:
        return _today_log_path()

    def _prune_old_files(self) -> None:
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=LOG_RETENTION_DAYS)
        if not self.log_dir.exists():
            return
        for f in self.log_dir.glob("trading_????-??-??.log"):
            try:
                fdate = datetime.strptime(f.stem.split("_", 1)[1], "%Y-%m-%d").date()
                if fdate < cutoff:
                    f.unlink()
            except (IndexError, ValueError):
                continue

    def flush(self):
        if self._stream is not None:
            self._stream.flush()

    def close(self):
        if self._stream is not None:
            self._stream.close()
        super().close()


# ── Rate Limiter (from original) ───────────────────────────────────────
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
    "get_symbol_ticker": 2,
    "get_klines": 2,
}
_DEFAULT_WEIGHT = 20
_INTER_REQUEST_DELAY = 0.5


def _estimate_weight(uri: str, method: str = "GET") -> int:
    for keyword, weight in ENDPOINT_WEIGHTS.items():
        if keyword in uri:
            return weight
    return _DEFAULT_WEIGHT + (10 if method.upper() == "POST" else 0)


class BinanceRateLimiter:
    """Token-bucket rate limiter with circuit breaker (from original)."""

    def __init__(self, max_weight: int = 1000, window: int = 60):
        self.max_weight = max_weight
        self.window = window
        self._used = 0
        self._window_start: float = 0.0
        self._cooldown_until: float = 0.0
        self._ban_count: int = 0
        self._lock = threading.Lock()

    def acquire(self, weight: int) -> None:
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
        with self._lock:
            self._reset_if_expired()
            self._used = weight
            self._window_start = time.time()

    def enter_cooldown(self, ban_until_ms: int = 0) -> None:
        with self._lock:
            now = time.time()
            default_seconds = min(300.0, 30.0 * (2.0 ** self._ban_count))
            self._ban_count += 1
            if ban_until_ms > 0:
                ban_remaining = (ban_until_ms / 1000.0) - now
                cooldown = max(default_seconds, ban_remaining)
            else:
                cooldown = default_seconds
            self._cooldown_until = now + cooldown

    def _reset_if_expired(self) -> None:
        now = time.time()
        if now - self._window_start >= self.window:
            self._used = 0
            self._window_start = now

    def _wait_cooldown(self) -> None:
        now = time.time()
        if now < self._cooldown_until:
            remaining = self._cooldown_until - now
            logging.warning(f"Rate-limit cooldown: sleeping {remaining:.0f}s (ban #{self._ban_count})")
            time.sleep(remaining)


# ── Position Cache (from original) ─────────────────────────────────────
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


# ── Data Classes ────────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════
# Base Broker
# ═══════════════════════════════════════════════════════════════════════

class BaseBroker(ABC):
    """Abstract broker interface."""

    def __init__(self):
        self._cached_balance = None
        self._balance_cache_time = 0
        self._balance_cache_duration = 5
        self._klines_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, float]] = {}
        self._position_cache = _PositionCache(ttl=2.0)
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def setup_client(self):
        pass

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
    def close_position(self, symbol: str, position: float) -> Optional[float]:
        pass

    def _parse_timeframe_to_minutes(self, timeframe: str) -> int:
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
        return value

    def _klines_cache_ttl(self, timeframe: str) -> float:
        minutes = self._parse_timeframe_to_minutes(timeframe)
        return max(30.0, minutes * 0.8 * 60.0)

    def get_historical_prices(
        self,
        symbol: str,
        length: int,
        timeframe: str = "15m"
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data with caching (from original)."""
        cache_key = (symbol, timeframe)
        now = time.time()

        if cache_key in self._klines_cache:
            cached_df, cached_at = self._klines_cache[cache_key]
            ttl = self._klines_cache_ttl(timeframe)
            cache_age = now - cached_at
            if cache_age < ttl and len(cached_df) >= length:
                self.logger.debug(f"Cache HIT for {symbol} {timeframe}")
                return cached_df
            self.logger.debug(f"Cache MISS for {symbol} {timeframe}")

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

            klines = self._fetch_klines(symbol=symbol, interval=interval, limit=length)

            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)

            self._klines_cache[cache_key] = (df, time.time())
            return df
        except Exception as e:
            self.logger.error(f"Error fetching historical prices for {symbol}: {e}")
            if cache_key in self._klines_cache:
                cached_df, _ = self._klines_cache[cache_key]
                self.logger.warning(f"Returning stale cached data for {symbol}")
                return cached_df
            return None

    def _fetch_klines(self, symbol: str, interval: str, limit: int):
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════
# Binance Futures Broker (from original binancefuturesbroker.py)
# ═══════════════════════════════════════════════════════════════════════

class BinanceBroker(BaseBroker):
    """
    Binance Futures broker with rate limiter, caching, and bracket orders.
    Uses testnet by default.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        symbol: str = SYMBOL,
    ):
        super().__init__()
        self.symbol = symbol
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self.setup_client()

    def setup_client(self):
        self.client = Client(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
        )
        self._install_rate_limiter()
        self.logger.info(f"Connected to Binance Futures ({'testnet' if self.testnet else 'mainnet'})")

    def _install_rate_limiter(self):
        """Install rate limiter with circuit breaker (from original)."""
        limiter = BinanceRateLimiter(max_weight=1000)
        original_request = self.client._request
        _last_request_time: float = 0.0
        _request_count = 0

        def _parse_ban_ms(msg: str) -> int:
            m = _re.search(r"banned until (\d+)", msg)
            return int(m.group(1)) if m else 0

        def _wrapped(method, uri, signed, force_params=False, **kwargs):
            nonlocal _last_request_time, _request_count

            now = time.time()
            since_last = now - _last_request_time
            _request_count += 1

            # Inter-request delay
            if since_last < _INTER_REQUEST_DELAY:
                time.sleep(_INTER_REQUEST_DELAY - since_last)

            # Estimate weight and acquire budget
            weight = _estimate_weight(uri, method)
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
                    f"API response #{_request_count}: {method} {uri.split('?')[0][-40:]} "
                    f"({elapsed*1000:.0f}ms, weight={server_weight or '?'})"
                )
                return result
            except BinanceAPIException as exc:
                self.logger.error(f"API error: {method} {uri} — code={exc.code}")
                if "-1003" in str(exc):
                    ban_ms = _parse_ban_ms(str(exc))
                    self.logger.error(f"Rate-limit ban detected — entering cooldown")
                    limiter.enter_cooldown(ban_until_ms=ban_ms)
                raise
            finally:
                _last_request_time = time.time()

        self.client._request = _wrapped
        self.logger.info(f"Rate limiter installed (max {limiter.max_weight} weight/min)")

    # ── Price Feed ─────────────────────────────────────────────────────
    def get_last_price(self, symbol: str = None) -> float:
        sym = symbol or self.symbol
        try:
            ticker = self.client.futures_symbol_ticker(symbol=sym)
            return float(ticker["price"])
        except Exception as e:
            self.logger.error(f"Failed to get price for {sym}: {e}")
            return 0.0

    # ── Cash ───────────────────────────────────────────────────────────
    def get_cash(self, quote_asset_symbol="USDT") -> float:
        now = time.time()
        if self._cached_balance is not None and now - self._balance_cache_time < self._balance_cache_duration:
            return self._cached_balance
        try:
            balances = self.client.futures_account_balance()
            bal = next((b for b in balances if b["asset"] == quote_asset_symbol), None)
            result = float(bal["balance"]) if bal else 0.0
            self._cached_balance = result
            self._balance_cache_time = now
            return result
        except Exception as e:
            self.logger.error(f"Failed to get cash: {e}")
            return 0.0

    def get_equity(self) -> float:
        """Alias for get_cash."""
        return self.get_cash()

    # ── Position Data (with cache) ─────────────────────────────────────
    def _cache_position_data(self, symbol: str) -> Optional[_PositionData]:
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if not positions:
                self._position_cache.invalidate()
                return None
            pos = positions[0]
            amt = float(pos.get("positionAmt", 0.0))
            entry = float(pos.get("entryPrice", 0.0))
            lev = int(pos.get("leverage", 0)) or None
            liq = float(pos.get("liquidationPrice", 0.0)) or None

            data = _PositionData(
                amount=amt,
                entry_price=entry,
                leverage=lev,
                liquidation_price=liq if liq and liq > 0 else None,
                cached_at=time.time(),
            )
            self._position_cache.set(data)
            return data
        except Exception as e:
            if "-1003" in str(e):
                raise
            self.logger.error(f"Error caching position data for {symbol}: {e}")
            self._position_cache.invalidate()
            return None

    def get_position(self, symbol: str = None) -> Optional[PositionResult]:
        sym = symbol or self.symbol
        cached = self._position_cache.get()
        if cached is not None:
            amt = cached.amount if abs(cached.amount) >= MIN_TRADEABLE_QUANTITY else None
            return PositionResult(amount=amt, entry_price=cached.entry_price)

        data = self._cache_position_data(sym)
        if data is None:
            return None
        amt = data.amount if abs(data.amount) >= MIN_TRADEABLE_QUANTITY else None
        return PositionResult(amount=amt, entry_price=data.entry_price)

    def get_position_leverage(self, symbol: str = None) -> Optional[int]:
        sym = symbol or self.symbol
        cached = self._position_cache.get()
        if cached is not None:
            return cached.leverage
        data = self._cache_position_data(sym)
        return data.leverage if data else None

    def get_liquidation_price(self, symbol: str = None) -> Optional[float]:
        sym = symbol or self.symbol
        cached = self._position_cache.get()
        if cached is not None:
            return cached.liquidation_price
        data = self._cache_position_data(sym)
        return data.liquidation_price if data else None

    # ── Orders ─────────────────────────────────────────────────────────
    def _create_market_order(self, symbol: str, side: str, quantity: float) -> Optional[MarketOrderResult]:
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
            )
            return MarketOrderResult(order_id=str(order.get("orderId")), entry_price=None)
        except Exception as e:
            self.logger.error(f"Market order failed: {e}")
            return None

    def _create_bracket_order(self, symbol, amount, side, tp_price, sl_price) -> Optional[BracketOrderResult]:
        try:
            exit_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
            tp_order = self.client.futures_create_order(
                symbol=symbol,
                side=exit_side,
                type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=tp_price,
                closePosition=True,
            )
            sl_order = self.client.futures_create_order(
                symbol=symbol,
                side=exit_side,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=sl_price,
                closePosition=True,
            )
            tp_id = str(tp_order.get("algoId"))
            sl_id = str(sl_order.get("algoId"))
            return BracketOrderResult(tp_order_id=tp_id, sl_order_id=sl_id)
        except Exception as e:
            self.logger.error(f"Bracket order failed: {e}")
            return None

    def open_position_with_bracket(
        self,
        symbol: str,
        signal: str,
        quantity: float,
        tp_frac: float = 0.02,
        sl_frac: float = 0.01
    ) -> BracketResult:
        """Open position with bracket order (from original)."""
        if signal not in [SIGNAL_LONG, SIGNAL_SHORT]:
            return BracketResult(success=False, error="Invalid signal")

        market_order_side = SIDE_BUY if signal == SIGNAL_LONG else SIDE_SELL

        try:
            # 1. Market entry order
            order_result = self._create_market_order(symbol, market_order_side, quantity)
            if order_result is None:
                return BracketResult(success=False, error="Market order failed")

            # 2. Get entry price
            entry_price = order_result.entry_price
            if entry_price is None:
                for attempt in range(5):
                    time.sleep(0.2 * (2 ** attempt))
                    position = self.get_position(symbol)
                    if position and position.entry_price and position.entry_price > 0:
                        entry_price = position.entry_price
                        break
                if entry_price is None:
                    close_qty = quantity if signal == SIGNAL_LONG else -quantity
                    self.close_position(symbol, close_qty)
                    return BracketResult(success=False, error="Fill confirmation timeout")

            if not order_result.order_id:
                return BracketResult(success=False, error="No order_id returned")
            if entry_price is None or entry_price <= 0:
                return BracketResult(success=False, error="Invalid entry_price")

            # 3. Calculate TP/SL prices
            if market_order_side == SIDE_BUY:
                tp_price = round(entry_price * (1 + tp_frac), 2)
                sl_price = round(entry_price * (1 - sl_frac), 2)
            else:
                tp_price = round(entry_price * (1 - tp_frac), 2)
                sl_price = round(entry_price * (1 + sl_frac), 2)

            # 4. Place bracket orders
            bracket_order_result = self._create_bracket_order(
                symbol, quantity, market_order_side, tp_price, sl_price
            )

            # 5. TP/SL failure → close position
            if not bracket_order_result:
                position = self.get_position(symbol)
                if position and position.amount:
                    self.close_position(symbol, position.amount)
                return BracketResult(success=False, error="TP/SL placement failed; position closed")

            return BracketResult(
                success=True,
                data={
                    "order_id": order_result.order_id,
                    "entry_price": entry_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "tp_algo_id": bracket_order_result.tp_order_id,
                    "sl_algo_id": bracket_order_result.sl_order_id,
                }
            )
        except Exception as e:
            return BracketResult(success=False, error=str(e))

    def open_position(
        self,
        side: str,
        stake_frac: float,
        leverage: int,
        stop_loss_frac: float,
        take_profit_frac: float,
    ) -> BracketResult:
        """
        Strategy-friendly entry. Computes quantity from equity/stake/leverage,
        quantizes to the symbol step size, sets leverage, and delegates to
        open_position_with_bracket for the market entry + TP/SL bracket.
        """
        symbol = self.symbol
        price = self.get_last_price(symbol)
        if price <= 0:
            self.logger.error(f"Cannot enter: price fetch returned {price}")
            return BracketResult(success=False, error="price fetch failed")

        equity = self.get_cash()
        if equity <= 0:
            self.logger.error(f"Cannot enter: equity={equity}")
            return BracketResult(success=False, error="no equity")

        stake = equity * stake_frac * leverage
        qty = self._quantize_qty(stake / price)
        if qty < MIN_TRADEABLE_QUANTITY:
            self.logger.warning(
                f"Quantized qty {qty:.6f} below minimum {MIN_TRADEABLE_QUANTITY}"
            )
            return BracketResult(success=False, error="qty below minimum")

        self.set_leverage(symbol, leverage, margin_type="ISOLATED")

        signal = SIGNAL_LONG if side == "long" else SIGNAL_SHORT
        return self.open_position_with_bracket(
            symbol=symbol,
            signal=signal,
            quantity=qty,
            tp_frac=take_profit_frac,
            sl_frac=stop_loss_frac,
        )

    def _quantize_qty(self, qty: float) -> float:
        """Floor-quantize quantity to the symbol's step size."""
        step = 10 ** (-TRADEABLE_QUANTITY_PRECISION)
        return (qty // step) * step

    def cancel_open_orders(self, symbol: str = None, max_retries: int = 3, base_delay: float = 0.5):
        sym = symbol or self.symbol
        last_error = None
        for attempt in range(max_retries):
            try:
                open_orders = self.client.futures_get_open_orders(symbol=sym, conditional=True)
                if not open_orders:
                    return
                for o in open_orders:
                    algo_id = o.get("algoId")
                    if algo_id:
                        self.client.futures_cancel_order(symbol=sym, algoId=algo_id, conditional=True)
                remaining = self.client.futures_get_open_orders(symbol=sym, conditional=True)
                if not remaining:
                    return
                last_error = f"{len(remaining)} order(s) still open after cancel"
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        self.logger.error(f"Cancel open orders failed after {max_retries} retries: {last_error}")

    def close_position(self, symbol: str = None, position: float = None) -> Optional[float]:
        """Close position and return fill price (from original)."""
        sym = symbol or self.symbol
        if position is None:
            pos = self.get_position(sym)
            position = pos.amount if pos else 0.0
        if not position:
            return None

        try:
            side = SIDE_SELL if position > 0 else SIDE_BUY
            order = self.client.futures_create_order(
                symbol=sym,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=abs(position),
                reduceOnly=True,
            )
            executed_qty = order.get("executedQty", "0")
            cum_quote = order.get("cumQuote", "0")
            fill_price = None
            if executed_qty and float(executed_qty) > 0 and cum_quote and float(cum_quote) > 0:
                fill_price = float(cum_quote) / float(executed_qty)
            if fill_price is None:
                avg_price = order.get("avgPrice")
                if avg_price:
                    fill_price = float(avg_price)
            self.logger.info(
                f"Close position: {sym} qty={abs(position):.4f} fill={fill_price if fill_price else 0:.2f}"
            )
            return fill_price
        except Exception as e:
            self.logger.error(f"Close position failed for {sym}: {e}")
            return None

    def _fetch_klines(self, symbol: str, interval: str, limit: int):
        return self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)

    def set_leverage(self, symbol: str, leverage: int, margin_type: str = "ISOLATED") -> bool:
        sym = symbol or self.symbol
        # Binance requires integer leverage; strategic model emits floats (e.g. 3.33).
        leverage = int(round(leverage)) if leverage is not None else leverage
        try:
            positions = self.client.futures_position_information(symbol=sym)
            current_margin = positions[0].get("marginType", "").upper() if positions else ""
            # No open position => cannot read current margin type; a change attempt
            # is a no-op and just logs a misleading -4046 ERROR. Skip it.
            if positions and current_margin != margin_type.upper():
                for _attempt in range(2):
                    try:
                        self.client.futures_change_margin_type(symbol=sym, marginType=margin_type)
                        break
                    except Exception as e:
                        if "No need to change margin type" in str(e):
                            break
                        if "-1007" in str(e) and _attempt == 0:
                            time.sleep(1)
                            continue
                        self.logger.warning(f"Could not set margin type for {sym}: {e}")
                        break
        except Exception as e:
            self.logger.warning(f"Could not read margin type for {sym}: {e}")

        try:
            self.client.futures_change_leverage(symbol=sym, leverage=leverage)
        except Exception as e:
            if "4161" in str(e):
                self.logger.warning(f"Leverage reduction not supported with open positions")
            else:
                self.logger.error(f"Set leverage failed for {sym}: {e}")
            return False

        try:
            positions = self.client.futures_position_information(symbol=sym)
            if positions:
                confirmed = int(positions[0].get("leverage", 0))
                if confirmed != leverage:
                    self.logger.warning(f"Leverage mismatch: requested={leverage}x confirmed={confirmed}x")
                else:
                    self.logger.info(f"Leverage confirmed: {confirmed}x for {sym}")
        except Exception as e:
            self.logger.warning(f"Could not verify leverage: {e}")
        return True


# ═══════════════════════════════════════════════════════════════════════
# Mock Broker (Simulation) - from simulate.py
# ═══════════════════════════════════════════════════════════════════════

from simulate import MockBroker


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════

def create_broker(
    mode: str = "simulation",
    **kwargs
) -> BaseBroker:
    """Factory function to create broker instance."""
    if mode == "simulation":
        df = kwargs.get('df')
        if df is None:
            raise ValueError("Simulation mode requires 'df' parameter")
        return MockBroker(
            df=df,
            fee=kwargs.get('fee', FEE),
            slippage=kwargs.get('slippage', SLIPPAGE),
            initial_equity=kwargs.get('initial_equity', INITIAL_EQUITY),
        )
    elif mode == "live":
        return BinanceBroker(
            api_key=kwargs.get('api_key', ''),
            api_secret=kwargs.get('api_secret', ''),
            testnet=kwargs.get('testnet', True),
            symbol=kwargs.get('symbol', SYMBOL),
        )
    else:
        raise ValueError(f"Unknown broker mode: {mode}")