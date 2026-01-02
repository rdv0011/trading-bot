from binance.client import Client
from binance.enums import *
from basebinancebroker import BaseBinanceBroker, BracketResult, MarketOrderResult

class BinanceFuturesBroker(BaseBinanceBroker):

    def setup_client(self):
        self.client = Client(
            self.config["api_key"],
            self.config["api_secret"],
            testnet=self.testnet
        )
        self.client.futures_account()
        self.logger.info("✅ Connected to Binance Futures")

    # ---------- Futures Specific ----------

    def get_cash(self, quote_asset: str = "USDT") -> float:
        balances = self.client.futures_account_balance()
        for b in balances:
            if b["asset"] == quote_asset:
                return float(b["balance"])
        return 0.0

    def get_last_price(self, symbol: str) -> float:
        return float(self.client.futures_symbol_ticker(symbol=symbol)["price"])

    def _fetch_klines(self, symbol, interval, limit):
        return self.client.futures_klines(
            symbol=symbol, interval=interval, limit=limit
        )

    def _create_market_order(self, symbol, side, quantity):
        order = self.client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY if side == "buy" else SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        price = float(order.get("avgPrice") or self.get_last_price(symbol))
        return MarketOrderResult(str(order["orderId"]), price)

    def open_position_with_bracket(self, symbol, signal, quantity, tp_frac=0.02, sl_frac=0.01):
        side = "buy" if signal == "long" else "sell"
        entry = self._create_market_order(symbol, side, quantity)
        ep = entry.entry_price

        tp = round(ep * (1 + tp_frac), 2) if signal == "long" else round(ep * (1 - tp_frac), 2)
        sl = round(ep * (1 - sl_frac), 2) if signal == "long" else round(ep * (1 + sl_frac), 2)

        close_side = SIDE_SELL if signal == "long" else SIDE_BUY

        self.client.futures_create_order(
            symbol=symbol,
            side=close_side,
            type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
            stopPrice=tp,
            closePosition=True
        )
        self.client.futures_create_order(
            symbol=symbol,
            side=close_side,
            type=FUTURE_ORDER_TYPE_STOP_MARKET,
            stopPrice=sl,
            closePosition=True
        )

        return BracketResult(True, data={
            "entry_price": ep,
            "tp_price": tp,
            "sl_price": sl
        })
