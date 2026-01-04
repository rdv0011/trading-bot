import time
from typing import Optional
from binance.client import Client
from binance.enums import *
from binancebasebroker import BinanceBaseBroker, MIN_TRADEABLE_QUANTITY, PositionResult

class BinanceSpotBroker(BinanceBaseBroker):

    def setup_client(self):
        self.client = Client(
            self.config["api_key"],
            self.config["api_secret"],
            testnet=self.config.get("testnet", True)
        )
        self.logger.info("✅ Connected to Binance Spot")

    def get_cash(self, quote_asset_symbol="USDT") -> float:
        balances = self.client.get_account()["balances"]
        bal = next((b for b in balances if b["asset"] == quote_asset_symbol), None)
        return float(bal["free"]) if bal else 0.0

    def get_position(self, symbol: str) -> Optional[PositionResult]:
        base = symbol.replace("USDT", "")
        balances = self.client.get_account()["balances"]
        bal = next((b for b in balances if b["asset"] == base), None)
        qty = float(bal["free"]) if bal else 0.0
        return qty if qty >= MIN_TRADEABLE_QUANTITY else None

    def get_last_price(self, symbol: str) -> float:
        return float(self.client.get_symbol_ticker(symbol=symbol)["price"])

    def _create_market_order(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        try:
            if side != SIDE_BUY:
                raise Exception("Spot shorting not supported")
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            return str(order.get("orderId"))
        except Exception as e:
            self.logger.error(f"❌ Spot market order failed: {e}")
            return None

    def _create_bracket_order(self, symbol, quantity, side, tp_price, sl_price):
        try:
            # Spot only allows long, use OCO
            oco_order = self.client.create_oco_order(
                symbol=symbol,
                side=SIDE_SELL,
                quantity=quantity,
                price=str(tp_price),
                stopPrice=str(sl_price),
                stopLimitPrice=str(sl_price * 0.999),
                stopLimitTimeInForce=TIME_IN_FORCE_GTC
            )
            return oco_order.get("orderListId"), oco_order.get("orderListId")
        except Exception as e:
            self.logger.error(f"❌ Spot OCO order failed: {e}")
            return None, None

    def cancel_open_orders(self, symbol: str):
        try:
            self.client.cancel_open_orders(symbol=symbol)
        except Exception as e:
            self.logger.error(f"❌ Cancel open orders failed: {e}")

    def close_position(self, symbol: str, position: float):
        try:
            self.client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=position
            )
        except Exception as e:
            self.logger.error(f"❌ Close position failed: {e}")

    def _fetch_klines(self, symbol: str, interval: str, limit: int):
        return self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
