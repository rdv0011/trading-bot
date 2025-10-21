import ccxt
from ccxt.base.errors import NetworkError, RequestTimeout, ExchangeError
import pandas as pd
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from typing import Dict, Any, Optional, Callable, TypeVar

# Add after constants
T = TypeVar('T')

# Constants
MIN_TRADEABLE_QUANTITY = 0.001
TRADEABLE_QUANTITY_PRECISION = 3
BUY_SLIPPAGE = 1.005
SELL_SLIPPAGE = 0.995
RETRY_PARAMETERS = {'maxRetriesOnFailure': 3, 'maxRetriesOnFailureDelay': 2000}
EXCHANGE_VERBOSE = False

class CCXTBroker:
    """
    CCXT Broker wrapper that mimics Lumibot's broker interface
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        self.setup_exchange()
        self.pending_orders = []
        self._cached_balance = None
        self._balance_cache_time = 0
        self._balance_cache_duration = 5  # Cache balance for 5 seconds
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_exchange(self):
        """Initialize CCXT exchange"""
        exchange_config = self.config.get('exchange', {})
        exchange_id = exchange_config.get('id', 'binance')
        
        exchange_class = getattr(ccxt, exchange_id)
        ccxt_exchange_config = {
            'api_key': exchange_config.get('api_key'),
            'secret': exchange_config.get('secret'),
            'enableRateLimit': True,
            'options': exchange_config.get('options', {
                'defaultType': 'future',  # Use 'future' for Binance Futures, or 'spot' for spot trading
            })
        }
        self.exchange = exchange_class(ccxt_exchange_config)
        self.exchange.set_sandbox_mode(exchange_config.get('sandbox', True))
        self.exchange.verbose = EXCHANGE_VERBOSE

        print(f"✅ Sandbox mode is {'enabled' if self.exchange.isSandboxModeEnabled else 'disabled'}\nAPI: {self.exchange.urls['api']['dapiPublic']}")
        
        # Test connection
        try:
            self.exchange.load_markets(params=RETRY_PARAMETERS)
            self.logger.info(f"✅ Connected to {exchange_id}")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to exchange: {e}")
            raise
    
    def get_cash(self, quote_asset_symbol: str = 'USDT') -> float:
        """Get available cash balance"""
        try:
            current_time = time.time()
            if (self._cached_balance is None or 
                current_time - self._balance_cache_time > self._balance_cache_duration):
                self._cached_balance = self.exchange.fetch_balance(params=RETRY_PARAMETERS)
                self._balance_cache_time = current_time
            return self._cached_balance[quote_asset_symbol]['free']
        except Exception as e:
            self.logger.error(f"❌ Error fetching cash balance for {quote_asset_symbol}: {e}")
            return 0.0

    def get_position(self, symbol: str) -> Optional[float]:
        """Get current position for symbol"""
        try:
            raw_position = self.exchange.fetch_positions([symbol], params=RETRY_PARAMETERS)
            position = next((p for p in raw_position if p['info']['symbol'] == symbol), None)
            if position is None:
                return None
            
            contracts = position.get('contracts', 0)
            contract_size = position.get('contractSize', 1)
            position_size = contracts * contract_size
            
            if position_size >= MIN_TRADEABLE_QUANTITY:
                if position.get('side') == 'short':
                    return -position_size
                return position_size
            return None
        except Exception as e:
            self.logger.error(f"❌ Error fetching position for {symbol}: {e}")
            return None
    
    def get_last_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            ticker = self.exchange.fetch_ticker(symbol, params=RETRY_PARAMETERS)
            return ticker['last']
        except Exception as e:
            self.logger.error(f"❌ Error fetching price for {symbol}: {e}")
            return 0.0
    
    def get_historical_prices(self, symbol: str, length: int, timeframe: str = '5m') -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=length, params=RETRY_PARAMETERS)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"❌ Error fetching historical prices for {symbol}: {e}")
            return None
    
    def create_market_order(self, symbol: str, side: str, amount: float) -> Optional[str]:
        """Place market order"""
        try:
            order = self.exchange.create_market_order(symbol, side, amount)
            order_id = order.get('id')
            if order_id:
                self.logger.info(f"✅ {side.upper()} market order {order_id} placed: {amount:.6f} at market price")
                return order_id
            else:
                self.logger.error(f"❌ Order placed but no ID returned")
                return None
        except Exception as e:
            self.logger.error(f"❌ Error placing {side} market order: {e}")
            return None
    
    def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[str]:
        """Place limit order"""
        try:
            order = self.exchange.create_limit_order(symbol, side, amount, price)
            self.logger.info(f"✅ {side.upper()} limit order placed: {amount:.6f} at {price:.2f}")
            order_id = order['id']
            self.pending_orders.append(order_id)
            return order_id
        except Exception as e:
            self.logger.error(f"❌ Error placing {side} limit order: {e}")
            return None
    
    def create_stop_limit_order(self, symbol: str, side: str, amount: float, 
                               stop_price: float, limit_price: float) -> Optional[str]:
        """Place stop-limit order"""
        try:
            params = {
                'stopPrice': stop_price,
                'timeInForce': 'GTC'
            }
            order = self.exchange.create_order(symbol, 'limit', side, amount, limit_price, params)
            self.logger.info(f"✅ {side.upper()} stop-limit order placed: {amount:.6f} stop at {stop_price:.2f}")
            order_id = order['id']
            self.pending_orders.append(order_id)
            return order_id
        except Exception as e:
            self.logger.error(f"❌ Error placing {side} stop-limit order: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order"""
        try:
            self.exchange.cancel_order(order_id, symbol, params=RETRY_PARAMETERS)
            self.logger.info(f"✅ Order {order_id} cancelled")
            if order_id in self.pending_orders:
                self.pending_orders.remove(order_id)
            return True
        except Exception as e:
            self.logger.error(f"❌ Error cancelling order {order_id}: {e}")
            return False
    
    def cancel_open_orders(self, symbol: str):
        """Cancel all open orders for symbol"""
        try:
            open_orders = self.exchange.fetch_open_orders(symbol, params=RETRY_PARAMETERS)
            for order in open_orders:
                self.cancel_order(symbol, order['id'])
            self.pending_orders.clear()
        except Exception as e:
            self.logger.error(f"❌ Error cancelling open orders: {e}")
    
    def get_datetime(self) -> datetime:
        """Get current datetime"""
        return datetime.now()
    
    def log_message(self, message: str):
        """Log message"""
        self.logger.info(message)