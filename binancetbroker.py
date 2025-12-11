from binance.client import Client
from binance.enums import *
import pandas as pd
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, TypeVar
from dataclasses import dataclass, field

T = TypeVar('T')

# Constants
MIN_TRADEABLE_QUANTITY = 0.001
TRADEABLE_QUANTITY_PRECISION = 3
BUY_SLIPPAGE = 1.005
SELL_SLIPPAGE = 0.995

@dataclass
class BracketResult:
    success: bool
    error: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketOrderResult:
    order_id: str
    entry_price: float

class BinanceBroker:
    """
    Binance SDK broker wrapper that mimics Lumibot's broker interface
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        self.setup_client()
        self._cached_balance = None
        self._balance_cache_time = 0
        self._balance_cache_duration = 5  # Cache balance for 5 seconds

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def setup_client(self):
        """Initialize Binance Futures testnet client"""
        api_key = self.config['api_key']
        api_secret = self.config['api_secret']
        self.client = Client(api_key, api_secret, testnet=True)
        try:
            info = self.client.futures_account()
            if isinstance(info, dict) and 'totalWalletBalance' in info:
                self.logger.info("✅ Connected to Binance Futures Testnet")
            else:
                self.logger.error("❌ Binance Futures Testnet connection returned unexpected data")
                raise Exception("Invalid account info returned")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Binance Futures Testnet: {e}")
            raise

    def get_cash(self, quote_asset_symbol: str = 'USDT') -> float:
        """Get available cash balance"""
        try:
            current_time = time.time()
            if (self._cached_balance is None or 
                current_time - self._balance_cache_time > self._balance_cache_duration):
                balances = self.client.futures_account_balance()
                self._cached_balance = {b['asset']: float(b['balance']) for b in balances}
                self._balance_cache_time = current_time
            return self._cached_balance.get(quote_asset_symbol, 0.0)
        except Exception as e:
            self.logger.error(f"❌ Error fetching cash balance: {e}")
            return 0.0

    def get_position(self, symbol: str) -> Optional[float]:
        """Get current Futures position"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if not positions:
                return None
            pos = positions[0]
            amt = float(pos['positionAmt'])
            if abs(amt) >= MIN_TRADEABLE_QUANTITY:
                return amt
            return None
        except Exception as e:
            self.logger.error(f"❌ Error fetching position for {symbol}: {e}")
            return None

    def get_last_price(self, symbol: str) -> float:
        """Get last price"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            self.logger.error(f"❌ Error fetching price for {symbol}: {e}")
            return 0.0

    def get_historical_prices(self, symbol: str, length: int, timeframe: str = '5m') -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data"""
        try:
            interval_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '3m': Client.KLINE_INTERVAL_3MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY
            }
            interval = interval_map.get(timeframe, Client.KLINE_INTERVAL_5MINUTE)
            klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=length)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                               'close_time','quote_asset_volume','number_of_trades',
                                               'taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp','open','high','low','close','volume']]
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)
            return df
        except Exception as e:
            self.logger.error(f"❌ Error fetching historical prices for {symbol}: {e}")
            return None
        
    def _get_entry_price_with_retry(self, symbol: str, order_id: str, retries: int = 5, delay: float = 0.5) -> float:
        """
        Try to fetch the filled price for an order, retrying if not immediately available.
        """
        for attempt in range(retries):
            try:
                order_info = self.client.futures_get_order(symbol=symbol, orderId=int(order_id))
                if order_info['status'] == 'FILLED' and float(order_info.get('avgPrice', 0)) > 0:
                    return float(order_info['avgPrice'])
            except Exception as e:
                self.logger.warning(f"Attempt {attempt+1}: Error fetching filled price for order {order_id}: {e}")
            time.sleep(delay)

        # Fallback
        self.logger.warning(f"⚠️ Could not fetch filled price for order {order_id} after {retries} retries, using last market price")
        return self.get_last_price(symbol)

    def _create_market_order(self, symbol: str, side: str, quantity: float) -> Optional[MarketOrderResult]:
        """
        Place market order and return order_id and actual entry price
        Returns: {'order_id': str, 'entry_price': float}
        """
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=SIDE_BUY if side.lower()=='buy' else SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            order_id = order.get('orderId')
            self.logger.info(f"✅ {side.upper()} market order placed: {quantity} {symbol}, order_id={order_id}")

            # Small delay to ensure order is filled
            time.sleep(0.5)

            # Fetch filled price
            entry_price = self._get_entry_price_with_retry(symbol, order_id)

            return MarketOrderResult(order_id=str(order_id), entry_price=entry_price)

        except Exception as e:
            self.logger.error(f"❌ Error placing {side} market order: {e}")
            return None

    def _create_bracket_order(self, symbol: str, signal: str, type: str, price: float) -> Optional[str]:
        try:
            order_side = SIDE_SELL if signal.lower() == "long" else SIDE_BUY
            order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type=type,
                stopPrice=price,
                closePosition=True
            )
            algo_id = order.get("algoId")
            self.logger.info(f"✅ {type} ({order_side.upper()}) placed at {price}, algoId={algo_id}")
            return str(algo_id) if algo_id else None
        except Exception as e:
            self.logger.error(f"❌ Error placing {type} ({order_side.upper()}) at {price}: {e}")
            return None

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
        Returns non-empty dict only if position and TP/SL orders successfully placed.
        Closes the position if TP or SL cannot be created.
        side: "long" or "short"
        """
        signal = signal.lower()
        if signal not in ["long", "short"]:
            return BracketResult(success=False, error="Invalid signal")

        side = "buy" if signal == "long" else "sell"

        try:
            create_order_result = self._create_market_order(symbol, side, quantity)
            if create_order_result is None:
                return BracketResult(
                    success=False,
                    error="Market order returned None"
                )
            order_id = create_order_result.order_id
            entry_price = create_order_result.entry_price

            if not order_id:
                return BracketResult(success=False, error="Market order returned no order_id")

            if entry_price is None or entry_price <= 0:
                return BracketResult(success=False, error="Market order returned invalid entry_price")

            if signal == "long":
                tp_price = round(entry_price * (1 + tp_frac), 2)
                sl_price = round(entry_price * (1 - sl_frac), 2)
            else:
                tp_price = round(entry_price * (1 - tp_frac), 2)
                sl_price = round(entry_price * (1 + sl_frac), 2)

            # 3. Place TP/SL
            tp_id = self._create_bracket_order(symbol, signal, FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET, tp_price)
            sl_id = self._create_bracket_order(symbol, signal, FUTURE_ORDER_TYPE_STOP_MARKET, sl_price)

            # TP or SL failure → close the position
            if not tp_id or not sl_id:
                position_amt = self.get_position(symbol)
                if position_amt:
                    self.close_position(symbol, position_amt)

                return BracketResult(success=False, error="TP/SL placement failed; position closed")

            return BracketResult(
                success=True,
                error="",
                data = {
                    "order_id": order_id,
                    "entry_price": entry_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "tp_algo_id": tp_id,
                    "sl_algo_id": sl_id
                }
            )

        except Exception as e:
            return BracketResult(success=False, error=str(e))

    def _cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            self.client.futures_cancel_order(symbol=symbol, algoId=order_id, conditional=True)
            self.logger.info(f"✅ Order {order_id} cancelled")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error cancelling order {order_id}: {e}")
            return False

    def cancel_open_orders(self, symbol: str):
        try:
            open_orders = self.client.futures_get_open_orders(symbol=symbol, conditional=True)
            for order in open_orders:
                order_id = order.get('algoId')
                self._cancel_order(symbol, order_id)
        except Exception as e:
            self.logger.error(f"❌ Error cancelling open orders: {e}")
    
    def close_position(self, symbol: str, position: float):
        """Close position for asset"""
        is_long = position > 0
        side = SIDE_SELL if is_long else SIDE_BUY
        _ = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=abs(position)
        )
        self.logger.info(f"✅ Closed position for {symbol} ({side} {abs(position)})")

    def get_datetime(self) -> datetime:
        return datetime.now()

    def log_message(self, message: str):
        self.logger.info(message)