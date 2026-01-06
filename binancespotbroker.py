import time
from typing import Optional
from collections import defaultdict
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from binance.enums import *
from binancebasebroker import BinanceBaseBroker, MIN_TRADEABLE_QUANTITY, MarketOrderResult, PositionResult, BracketOrderResult
from binance.enums import (
    SIDE_SELL,
    ORDER_TYPE_TAKE_PROFIT_LIMIT,
    ORDER_TYPE_STOP_LOSS_LIMIT,
    TIME_IN_FORCE_GTC,
)


class BinanceSpotBroker(BinanceBaseBroker):

    def setup_client(self):
        testnet = self.config.get("testnet", True)

        kwargs = dict(
            api_key=self.config["api_key"],
            api_secret=self.config["api_secret"],
            testnet=testnet,
        )

        if testnet:
            # Spot testnet requires explicit base_endpoint
            kwargs["base_endpoint"] = "https://testnet.binance.vision"

        self.client = Client(**kwargs)
        self.logger.info("✅ Connected to Binance Spot")

    def get_cash(self, quote_asset_symbol="USDT") -> float:
        balances = self.client.get_account()["balances"]
        bal = next((b for b in balances if b["asset"] == quote_asset_symbol), None)
        return float(bal["free"]) if bal else 0.0

    def get_position(self, symbol: str) -> Optional[PositionResult]:
        base = symbol.replace("USDT", "")

        balances = self.client.get_account().get("balances", [])
        bal = next((b for b in balances if b["asset"] == base), None)

        if not bal:
            return None

        free = float(bal.get("free", 0.0))
        locked = float(bal.get("locked", 0.0))
        total_qty = free + locked

        if total_qty < MIN_TRADEABLE_QUANTITY:
            return None

        return PositionResult(
            amount=total_qty,
            entry_price=0.0  # spot API does not expose entry price
        )
    
    def get_last_price(self, symbol: str) -> float:
        return float(self.client.get_symbol_ticker(symbol=symbol)["price"])

    def _create_market_order(self, symbol: str, side: str, quantity: float) -> Optional[MarketOrderResult]:
        try:
            if side != SIDE_BUY:
                raise Exception("Spot shorting not supported")

            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            order_id=str(order.get("orderId"))

            fills = order.get("fills", [])
            if not fills:
                return MarketOrderResult(
                    order_id=order_id,
                    entry_price=0.0
                )

            total_qty = 0.0
            total_cost = 0.0

            for f in fills:
                price = float(f["price"])
                qty = float(f["qty"])
                total_qty += qty
                total_cost += price * qty

            if total_qty <= 0:
                return MarketOrderResult(
                    order_id=order_id,
                    entry_price=0.0
                )

            entry_price = total_cost / total_qty

            return MarketOrderResult(
                order_id=order_id,
                entry_price=entry_price
            )

        except Exception as e:
            self.logger.error(f"❌ Spot market order failed: {e}")
            return None

    def _create_bracket_order(self, symbol, quantity, side, tp_price, sl_price) -> Optional[BracketOrderResult]:
        try:
            oco_order = self.client.create_oco_order(
                symbol=symbol,
                side=SIDE_SELL,
                quantity=quantity,

                # ---------- TAKE PROFIT ----------
                aboveType=ORDER_TYPE_TAKE_PROFIT_LIMIT,
                abovePrice=str(tp_price),
                aboveStopPrice=str(tp_price),
                aboveTimeInForce=TIME_IN_FORCE_GTC,

                # ---------- STOP LOSS ----------
                belowType=ORDER_TYPE_STOP_LOSS_LIMIT,
                belowStopPrice=str(sl_price),
                belowPrice=str(round(sl_price * 0.999, 2)),
                belowTimeInForce=TIME_IN_FORCE_GTC,
            )
            
            orders = oco_order.get("orders", [])
            if len(orders) != 2:
                self.logger.error(
                    f"❌ Invalid OCO response for {symbol}: {oco_order}"
                )
                return None
            
            # Binance guarantees order: [TP, SL] but we still detect defensively
            tp_order_id = str(orders[0].get("orderId"))
            sl_order_id = str(orders[1].get("orderId"))

            if not tp_order_id or not sl_order_id:
                self.logger.error(
                    f"❌ Missing order IDs in OCO response for {symbol}: {orders}"
                )
                return None

            self.logger.info(
                f"✅ Bracket order created for {symbol} "
                f"(TP={tp_order_id}, SL={sl_order_id})"
            )

            return BracketOrderResult(
                tp_order_id=tp_order_id,
                sl_order_id=sl_order_id,
            )

        except Exception as e:
            self.logger.error(f"❌ Spot OCO order failed: {e}")
            return None

    def cancel_open_orders(self, symbol: str, max_retries: int, base_delay: float):
        try:
            orders = self.client.get_open_orders(symbol=symbol)

            if not orders:
                self.logger.info(f"ℹ️ No open spot orders to cancel for {symbol}")
                return

            # Group orders by orderListId (OCO/bracket)
            orders_by_list_id = defaultdict(list)
            for o in orders:
                order_list_id = o.get("orderListId", -1)
                orders_by_list_id[order_list_id].append(o)

            for order_list_id, grouped_orders in orders_by_list_id.items():
                # Cancel only ONE order per orderListId
                order_to_cancel = grouped_orders[0]
                order_id = order_to_cancel.get("orderId")

                for attempt in range(1, max_retries + 1):
                    try:
                        self.client.cancel_order(
                            symbol=symbol,
                            orderId=order_id
                        )

                        if order_list_id != -1:
                            self.logger.info(
                                f"✅ Cancelled OCO orderListId={order_list_id} "
                                f"via orderId={order_id} for {symbol}"
                            )
                        else:
                            self.logger.info(
                                f"✅ Cancelled standalone order {order_id} for {symbol}"
                            )

                        break  # success → stop retrying

                    except (BinanceRequestException, BinanceAPIException) as e:
                        if attempt >= max_retries:
                            self.logger.error(
                                f"❌ Failed to cancel orderId={order_id} "
                                f"(orderListId={order_list_id}) for {symbol} "
                                f"after {attempt} attempts: {e}"
                            )
                        else:
                            delay = base_delay * (2 ** (attempt - 1))
                            self.logger.warning(
                                f"⚠️ Retry {attempt}/{max_retries} cancelling "
                                f"orderId={order_id} (orderListId={order_list_id}) "
                                f"in {delay:.2f}s: {e}"
                            )
                            time.sleep(delay)

            self.logger.info(f"✅ Spot open orders cancellation completed for {symbol}")

        except (BinanceRequestException, BinanceAPIException) as e:
            self.logger.error(
                f"❌ Failed to fetch open orders for {symbol}: {e}"
            )
        except Exception:
            self.logger.exception(
                f"🔥 Unexpected error while cancelling open orders for {symbol}"
            )

    def close_position(self, symbol: str, position: float):
        try:
            result = self.client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=position
            )
            if result and result.get("status") == ORDER_STATUS_FILLED:
                self.logger.info(f"✅ Spot position closed for {symbol}, qty: {position}")
            else:
                self.logger.error(f"❌ Spot close position failed for {symbol} {result}")
        except Exception as e:
            self.logger.error(f"❌ Close position failed: {e}")

    def _fetch_klines(self, symbol: str, interval: str, limit: int):
        return self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
