from typing import Optional
from binance.client import Client
from binance.enums import *
import time
from binancebasebroker import BinanceBaseBroker, BracketResult, MIN_TRADEABLE_QUANTITY

class BinanceFuturesBroker(BinanceBaseBroker):

    def setup_client(self):
        self.client = Client(
            self.config["api_key"],
            self.config["api_secret"],
            testnet=self.config.get("testnet", True)
        )
        self.client.futures_account()
        self.logger.info("✅ Connected to Binance Futures")

    def get_cash(self, quote_asset_symbol="USDT") -> float:
        balances = self.client.futures_account_balance()
        bal = next(b for b in balances if b["asset"] == quote_asset_symbol)
        return float(bal["balance"])

    def get_position(self, symbol: str) -> Optional[float]:
        pos = self.client.futures_position_information(symbol=symbol)[0]
        amt = float(pos["positionAmt"])
        return amt if abs(amt) >= MIN_TRADEABLE_QUANTITY else None

    def get_last_price(self, symbol: str) -> float:
        return float(self.client.futures_symbol_ticker(symbol=symbol)["price"])

    def _klines(self, symbol, interval, limit):
        return self.client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

    def open_position_with_bracket(self, symbol, signal, quantity, tp_frac, sl_frac):
        side = SIDE_BUY if signal == "long" else SIDE_SELL

        order = self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=quantity,
        )

        time.sleep(0.5)
        price = self.get_last_price(symbol)

        tp = price * (1 + tp_frac if signal == "long" else 1 - tp_frac)
        sl = price * (1 - sl_frac if signal == "long" else 1 + sl_frac)

        self.client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL if signal == "long" else SIDE_BUY,
            type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
            stopPrice=round(tp, 2),
            closePosition=True,
        )

        self.client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL if signal == "long" else SIDE_BUY,
            type=FUTURE_ORDER_TYPE_STOP_MARKET,
            stopPrice=round(sl, 2),
            closePosition=True,
        )

        return BracketResult(
            success=True,
            data={"entry_price": price}
        )

    def cancel_open_orders(self, symbol: str):
        self.client.futures_cancel_all_open_orders(symbol=symbol)

    def close_position(self, symbol: str, position: float):
        side = SIDE_SELL if position > 0 else SIDE_BUY
        self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=abs(position),
        )