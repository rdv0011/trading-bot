import time
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
from binancetbroker import BinanceBroker, BracketResult

class BaseStrategy:
    """
    Base strategy class that mimics Lumibot's Strategy interface
    """

    def __init__(self, broker: BinanceBroker, quote_symbol: str, parameters: Dict[str, Any]):
        self.broker = broker
        self.parameters = parameters
        self.quote_asset_symbol = quote_symbol
        self.is_running = False
        self.sleep_time = parameters.get("sleeptime", 300)  # Default 5 minutes
        
    def get_cash(self) -> float:
        """Get available cash"""
        return self.broker.get_cash(self.quote_asset_symbol)
    
    def _pair_asset_symbol(self, asset_symbol: str) -> str:
        """Get Binance trading symbol"""
        return f"{asset_symbol}{self.quote_asset_symbol}"

    def get_position(self, asset_symbol: str) -> Optional[float]:
        """Get position for asset"""
        return self.broker.get_position(self._pair_asset_symbol(asset_symbol))

    def get_last_price(self, asset_symbol: str) -> float:
        """Get last price for asset"""
        return self.broker.get_last_price(self._pair_asset_symbol(asset_symbol))

    def get_historical_prices(self, asset_symbol: str, length: int, timestep: str) -> pd.DataFrame:
        """Get historical prices"""
        return self.broker.get_historical_prices(
            self._pair_asset_symbol(asset_symbol), 
            length, 
            timestep
        )
    
    def open_position_with_bracket(self, symbol: str, side: str, quantity: float, tp_frac: float=0.02, sl_frac: float=0.01) -> BracketResult:
        """Open long position with TP and SL"""
        return self.broker.open_position_with_bracket(
            self._pair_asset_symbol(symbol),
            side,
            quantity, 
            tp_frac, 
            sl_frac
        )
    
    def cancel_open_orders(self, asset_symbol: str):
        """Cancel all open orders"""
        # Assuming we're only trading one symbol
        symbol = self._pair_asset_symbol(asset_symbol)
        self.broker.cancel_open_orders(symbol)

    def close_position(self, asset_symbol: str, position: float):
        symbol = self._pair_asset_symbol(asset_symbol)
        self.broker.close_position(symbol, position)
    
    def get_datetime(self) -> datetime:
        """Get current datetime"""
        return self.broker.get_datetime()
    
    def log_message(self, message: str):
        """Log message"""
        self.broker.log_message(message)
    
    def run(self):
        """Main strategy loop"""
        self.log_message("🚀 Starting ML Trading Strategy...")
        self.is_running = True
        
        try:
            # Initialize strategy
            self.initialize()
            
            while self.is_running:
                try:
                    self.on_trading_iteration()
                    
                    # Sleep based on sleeptime parameter
                    if isinstance(self.sleep_time, str):
                        if self.sleep_time.endswith('m'):
                            sleep_seconds = int(self.sleep_time[:-1]) * 60
                        else:
                            sleep_seconds = 300  # Default 5 minutes
                    else:
                        sleep_seconds = self.sleep_time
                    
                    self.log_message(f"💤 Sleeping for {sleep_seconds} seconds...")
                    time.sleep(sleep_seconds)
                    
                except KeyboardInterrupt:
                    self.log_message("⚠️  Strategy interrupted by user")
                    break
                except Exception as e:
                    self.log_message(f"❌ Error in trading iteration: {e}")
                    import traceback
                    self.log_message(f"Traceback: {traceback.format_exc()}")
                    time.sleep(60)  # Wait before retrying
        
        except KeyboardInterrupt:
            self.log_message("⚠️  Strategy interrupted by user")
                    
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        self.log_message("🛑 Shutting down strategy...")
        self.is_running = False
        
        try:
            self.on_abrupt_closing()
            self.log_message("✅ Strategy shutdown completed")
        except Exception as e:
            self.log_message(f"❌ Error during shutdown: {e}")
    
    # Abstract methods to be implemented by strategy
    def initialize(self):
        """Initialize strategy - to be implemented by subclass"""
        pass
    
    def on_trading_iteration(self):
        """Trading iteration logic - to be implemented by subclass"""
        pass
    
    def on_abrupt_closing(self):
        """Cleanup on abrupt closing - to be implemented by subclass"""
        pass