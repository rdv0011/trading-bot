from typing import Optional
from binance.client import Client
from binance.enums import *
import time
from binancebasebroker import BinanceBaseBroker, MIN_TRADEABLE_QUANTITY, PositionResult

from binance.client import Client
from binance.enums import *

class BinanceFuturesBroker(BinanceBaseBroker):

    def setup_client(self):
        self.client = Client(
            self.config["api_key"],
            self.config["api_secret"],
            testnet=self.config.get("testnet", True)
        )
        self.logger.info("✅ Connected to Binance Futures")

    def get_cash(self, quote_asset_symbol="USDT") -> float:
        balances = self.client.futures_account_balance()
        bal = next((b for b in balances if b["asset"] == quote_asset_symbol), None)
        return float(bal["balance"]) if bal else 0.0

    def get_position(self, symbol: str) -> Optional[PositionResult]:
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if not positions:
                return None
            pos = positions[0]
            amt = float(pos.get("positionAmt", 0.0))
            amt = amt if abs(amt) >= MIN_TRADEABLE_QUANTITY else None
            price = float(pos.get("entryPrice", 0.0))
            return PositionResult(amount=amt, entri_price=price)
        except Exception as e:
            self.logger.error(f"❌ Error fetching position for {symbol}: {e}")
            return None

    def get_last_price(self, symbol: str) -> float:
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception:
            return 0.0

    def _create_market_order(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            return str(order.get("orderId"))
        except Exception as e:
            self.logger.error(f"❌ Futures market order failed: {e}")
            return None

    def _create_bracket_order(self, symbol, amount, side, tp_price, sl_price):
        try:
            tp_order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=tp_price,
                closePosition=True
            )
            sl_order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=sl_price,
                closePosition=True
            )
            tp_id = str(tp_order.get("algoId"))
            sl_id = str(sl_order.get("algoId"))
            return tp_id, sl_id
        except Exception as e:
            self.logger.error(f"❌ Futures bracket order failed: {e}")
            return None, None

    def cancel_open_orders(self, symbol: str):
        try:
            open_orders = self.client.futures_get_open_orders(symbol=symbol, conditional=True)
            for o in open_orders:
                algo_id = o.get("algoId")
                if algo_id:
                    self.client.futures_cancel_order(symbol=symbol, algoId=algo_id, conditional=True)
        except Exception as e:
            self.logger.error(f"❌ Cancel open orders failed: {e}")

    def close_position(self, symbol: str, position: float):
        try:
            side = SIDE_SELL if position > 0 else SIDE_BUY
            self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=abs(position)
            )
        except Exception as e:
            self.logger.error(f"❌ Close position failed: {e}")

    def _fetch_klines(self, symbol: str, interval: str, limit: int):
        return self.client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )