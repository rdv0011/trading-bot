from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
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

@dataclass
class BracketResult:
    success: bool
    error: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PositionResult:
    amount: float
    entri_price: float

class BinanceBaseBroker(ABC):
    """
    Base broker interface for both spot and futures
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._cached_balance = None
        self._balance_cache_time = 0
        self._balance_cache_duration = 5
        self.setup_logging()
        self.setup_client()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
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
    def _create_market_order(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        pass

    @abstractmethod
    def _create_bracket_order(self, *args, **kwargs):
        pass

    @abstractmethod
    def cancel_open_orders(self, symbol: str):
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
            create_order_id = self._create_market_order(symbol, market_order_side, quantity)
            if create_order_id is None:
                return BracketResult(success=False, error="Market order returned None")
            
            # May be there is a way to wait untill the order is fulfilled
            time.sleep(0.5)

            position = self.get_position(symbol)
            entry_price = position.entri_price if position is not None else self.get_last_price()

            if not create_order_id:
                return BracketResult(success=False, error="Market order returned no order_id")
            if entry_price is None or entry_price <= 0:
                return BracketResult(success=False, error="Market order returned invalid entry_price")

            # 2️⃣ TP/SL prices
            if signal == SIGNAL_LONG:
                tp_price = round(entry_price * (1 + tp_frac), 2)
                sl_price = round(entry_price * (1 - sl_frac), 2)
            else:
                tp_price = round(entry_price * (1 - tp_frac), 2)
                sl_price = round(entry_price * (1 + sl_frac), 2)

            # 3️⃣ Place bracket orders
            braket_order_side = SIDE_SELL if signal == SIGNAL_LONG else SIDE_BUY

            tp_id, sl_id = self._create_bracket_order(symbol, quantity, braket_order_side, tp_price, sl_price)

            # 4️⃣ TP/SL failure → close position
            if not tp_id or not sl_id:
                position = self.get_position(symbol)
                if position.amount:
                    self.close_position(symbol, position.amount)
                return BracketResult(success=False, error="TP/SL placement failed; position closed")

            return BracketResult(
                success=True,
                data={
                    "order_id": create_order_id,
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
        """
        Fetch historical OHLCV data.
        Aligned 1:1 with original futures implementation.
        """
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

            return df

        except Exception as e:
            self.logger.error(f"❌ Error fetching historical prices for {symbol}: {e}")
            return None

    def _fetch_klines(self, symbol: str, interval: str, limit: int):
        """
        Implemented by Spot / Futures brokers.
        """
        raise NotImplementedError

    def get_datetime(self) -> datetime:
        return datetime.now()

    def log_message(self, msg: str):
        self.logger.info(msg)