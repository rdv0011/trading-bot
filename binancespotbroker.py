from typing import Optional
from binance.client import Client
from binance.enums import *
from binancebasebroker import BinanceBaseBroker, BracketResult, MarketOrderResult, MIN_TRADEABLE_QUANTITY

class BinanceSpotBroker(BinanceBaseBroker):

    def setup_client(self):
        self.client = Client(
            self.config["api_key"],
            self.config["api_secret"],
            testnet=self.config.get("testnet", True)
        )
        self.client.get_account()
        self.logger.info("✅ Connected to Binance Spot")

    def get_cash(self, quote_asset_symbol="USDT") -> float:
        balances = self.client.get_account()["balances"]
        bal = next(b for b in balances if b["asset"] == quote_asset_symbol)
        return float(bal["free"])

    def get_position(self, symbol: str) -> Optional[float]:
        base = symbol.replace("USDT", "")
        balances = self.client.get_account()["balances"]
        bal = next(b for b in balances if b["asset"] == base)
        qty = float(bal["free"])
        return qty if qty >= MIN_TRADEABLE_QUANTITY else None

    def get_last_price(self, symbol: str) -> float:
        return float(self.client.get_symbol_ticker(symbol=symbol)["price"])

    def _klines(self, symbol, interval, limit):
        return self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

    def open_position_with_bracket(self, symbol, signal, quantity, tp_frac, sl_frac):
        if signal != "long":
            return BracketResult(False, "Spot does not support shorting")

        order = self.client.create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=quantity,
        )

        price = self.get_last_price(symbol)

        tp = round(price * (1 + tp_frac), 2)
        sl = round(price * (1 - sl_frac), 2)

        # OCO order (TP + SL)
        self.client.create_oco_order(
            symbol=symbol,
            side=SIDE_SELL,
            quantity=quantity,
            price=str(tp),
            stopPrice=str(sl),
            stopLimitPrice=str(sl * 0.999),
            stopLimitTimeInForce=TIME_IN_FORCE_GTC,
        )

        return BracketResult(
            success=True,
            data={"entry_price": price}
        )

    def cancel_open_orders(self, symbol: str):
        self.client.cancel_open_orders(symbol=symbol)

    def close_position(self, symbol: str, position: float):
        self.client.create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=abs(position),
        )