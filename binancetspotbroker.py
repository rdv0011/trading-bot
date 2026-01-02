from binance.client import Client
from binance.enums import *
from basebinancebroker import BaseBinanceBroker, BracketResult, MarketOrderResult

class BinanceSpotBroker(BaseBinanceBroker):

    def setup_client(self):
        self.client = Client(
            self.config["api_key"],
            self.config["api_secret"],
            testnet=self.testnet
        )
        self.client.get_account()
        self.logger.info("✅ Connected to Binance Spot")

    # ---------- Spot Specific ----------

    def get_cash(self, quote_asset: str = "USDT") -> float:
        acct = self.client.get_account()
        for b in acct["balances"]:
            if b["asset"] == quote_asset:
                return float(b["free"])
        return 0.0

    def get_last_price(self, symbol: str) -> float:
        return float(self.client.get_symbol_ticker(symbol=symbol)["price"])

    def _fetch_klines(self, symbol, interval, limit):
        return self.client.get_klines(
            symbol=symbol, interval=interval, limit=limit
        )

    def _create_market_order(self, symbol, side, quantity):
        order = self.client.create_order(
            symbol=symbol,
            side=SIDE_BUY if side == "buy" else SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        fills = order.get("fills", [])
        price = float(fills[0]["price"]) if fills else self.get_last_price(symbol)
        return MarketOrderResult(str(order["orderId"]), price)

    def open_position_with_bracket(self, symbol, signal, quantity, tp_frac=0.02, sl_frac=0.01):
        if signal == "short":
            return BracketResult(False, "Spot does not support shorting")

        entry = self._create_market_order(symbol, "buy", quantity)
        ep = entry.entry_price

        tp = round(ep * (1 + tp_frac), 2)
        sl = round(ep * (1 - sl_frac), 2)

        self.client.create_oco_order(
            symbol=symbol,
            side=SIDE_SELL,
            quantity=quantity,
            price=str(tp),
            stopPrice=str(sl),
            stopLimitPrice=str(sl * 0.999),
            stopLimitTimeInForce=TIME_IN_FORCE_GTC
        )

        return BracketResult(True, data={
            "entry_price": ep,
            "tp_price": tp,
            "sl_price": sl
        })
