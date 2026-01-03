from abc import ABC, abstractmethod
from binance.client import Client
from binance.enums import *
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

MIN_TRADEABLE_QUANTITY = 0.001
TRADEABLE_QUANTITY_PRECISION = 3

@dataclass
class BracketResult:
    success: bool
    error: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketOrderResult:
    order_id: str
    entry_price: float


class BinanceBaseBroker(ABC):
    """
    Common interface for Binance Spot and Futures brokers
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
    def get_position(self, symbol: str) -> Optional[float]:
        pass

    @abstractmethod
    def get_last_price(self, symbol: str) -> float:
        pass

    @abstractmethod
    def open_position_with_bracket(
        self,
        symbol: str,
        signal: str,
        quantity: float,
        tp_frac: float,
        sl_frac: float,
    ) -> BracketResult:
        pass

    @abstractmethod
    def cancel_open_orders(self, symbol: str):
        pass

    @abstractmethod
    def close_position(self, symbol: str, position: float):
        pass

    # 🔁 Shared implementation
    def get_historical_prices(
        self,
        symbol: str,
        length: int,
        timeframe: str = "5m"
    ) -> Optional[pd.DataFrame]:

        interval_map = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "3m": Client.KLINE_INTERVAL_3MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
        }

        try:
            klines = self._klines(
                symbol=symbol,
                interval=interval_map.get(timeframe),
                limit=length,
            )

            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp","open","high","low","close","volume",
                    "close_time","qav","trades","tbav","tqav","ignore"
                ],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df[["open","high","low","close","volume"]].astype(float)
            return df

        except Exception as e:
            self.logger.error(f"❌ Historical price fetch failed: {e}")
            return None

    @abstractmethod
    def _klines(self, symbol, interval, limit):
        pass

    def get_datetime(self) -> datetime:
        return datetime.now()

    def log_message(self, msg: str):
        self.logger.info(msg)
