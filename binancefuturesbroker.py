import time
from typing import Optional
from binance.client import Client
from binance.enums import *
from binancebasebroker import BinanceBaseBroker, MIN_TRADEABLE_QUANTITY, MarketOrderResult, PositionResult, BracketOrderResult

from binance.client import Client
from binance.enums import *

class BinanceFuturesBroker(BinanceBaseBroker):

    def setup_client(self):
        self.client = Client(
            api_key=self.config["api_key"],
            api_secret=self.config["api_secret"],
            testnet=self.config.get("testnet", True)
        )
        self.logger.info("✅ Connected to Binance Futures")

    def get_cash(self, quote_asset_symbol="USDT") -> float:
        now = time.time()
        if self._cached_balance is not None and now - self._balance_cache_time < self._balance_cache_duration:
            return self._cached_balance
        balances = self.client.futures_account_balance()
        bal = next((b for b in balances if b["asset"] == quote_asset_symbol), None)
        result = float(bal["balance"]) if bal else 0.0
        self._cached_balance = result
        self._balance_cache_time = now
        return result

    def get_position(self, symbol: str) -> Optional[PositionResult]:
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if not positions:
                return None
            pos = positions[0]
            amt = float(pos.get("positionAmt", 0.0))
            amt = amt if abs(amt) >= MIN_TRADEABLE_QUANTITY else None
            price = float(pos.get("entryPrice", 0.0))
            return PositionResult(amount=amt, entry_price=price)
        except Exception as e:
            self.logger.error(f"❌ Error fetching position for {symbol}: {e}")
            return None

    def get_last_price(self, symbol: str) -> float:
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception:
            return 0.0

    def _create_market_order(self, symbol: str, side: str, quantity: float) -> Optional[MarketOrderResult]:
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            return MarketOrderResult(order_id=str(order.get("orderId")), entry_price=None)
        except Exception as e:
            self.logger.error(f"❌ Futures market order failed: {e}")
            return None

    def _create_bracket_order(self, symbol, amount, side, tp_price, sl_price) -> Optional[BracketOrderResult]:
        try:
            exit_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
            tp_order = self.client.futures_create_order(
                symbol=symbol,
                side=exit_side,
                type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=tp_price,
                closePosition=True
            )
            sl_order = self.client.futures_create_order(
                symbol=symbol,
                side=exit_side,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=sl_price,
                closePosition=True
            )
            tp_id = str(tp_order.get("algoId"))
            sl_id = str(sl_order.get("algoId"))
            return BracketOrderResult(tp_order_id=tp_id, sl_order_id=sl_id)
        except Exception as e:
            self.logger.error(f"❌ Futures bracket order failed: {e}")
            return None

    def cancel_open_orders(self, symbol: str, max_retries: int, base_delay: float):
        last_error = None
        for attempt in range(max_retries):
            try:
                open_orders = self.client.futures_get_open_orders(symbol=symbol, conditional=True)
                if not open_orders:
                    return  # nothing to cancel

                for o in open_orders:
                    algo_id = o.get("algoId")
                    if algo_id:
                        self.client.futures_cancel_order(symbol=symbol, algoId=algo_id, conditional=True)

                # Verify all orders were cancelled
                remaining = self.client.futures_get_open_orders(symbol=symbol, conditional=True)
                if not remaining:
                    return

                last_error = f"{len(remaining)} order(s) still open after cancel"
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))

        self.logger.error(f"❌ Cancel open orders failed after {max_retries} retries: {last_error}")

    def close_position(self, symbol: str, position: float):
        try:
            side = SIDE_SELL if position > 0 else SIDE_BUY
            side_label = "SELL" if side == SIDE_SELL else "BUY"
            self.logger.info(
                "🔵 close_position: %s %s qty=%s reduceOnly=True",
                symbol, side_label, f"{abs(position):.4f}",
            )
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=abs(position),
                reduceOnly=True
            )
            self.logger.info(
                "🔵 close_position result: orderId=%s status=%s executedQty=%s",
                order.get("orderId", "?"),
                order.get("status", "?"),
                order.get("executedQty", "?"),
            )
        except Exception as e:
            self.logger.error(f"❌ Close position failed for {symbol}: {e}")

    def get_liquidation_price(self, symbol: str) -> Optional[float]:
        """Fetch the current liquidation price from the exchange position info."""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if not positions:
                return None
            liq_price = positions[0].get("liquidationPrice")
            if liq_price is not None:
                val = float(liq_price)
                return val if val > 0 else None
            return None
        except Exception as e:
            self.logger.error(f"❌ Error fetching liquidation price for {symbol}: {e}")
            return None

    def _fetch_klines(self, symbol: str, interval: str, limit: int):
        return self.client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

    def get_position_leverage(self, symbol: str) -> Optional[int]:
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if not positions:
                return None
            return int(positions[0].get("leverage", 0)) or None
        except Exception as e:
            self.logger.error(f"❌ Error fetching leverage for {symbol}: {e}")
            return None

    def set_leverage(self, symbol: str, leverage: int, margin_type: str = "ISOLATED") -> bool:
        # ── Check current margin type to avoid unnecessary API calls ──
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            current_margin = positions[0].get("marginType", "").upper() if positions else ""
            current_lev = int(positions[0].get("leverage", 0)) if positions else 0
            if current_margin != margin_type.upper():
                for _attempt in range(2):
                    try:
                        self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
                        break
                    except Exception as e:
                        err_str = str(e)
                        if "No need to change margin type" in err_str:
                            break
                        if "-1007" in err_str and _attempt == 0:
                            self.logger.warning(f"⚠️ Margin type change timed out for {symbol}, retrying...")
                            time.sleep(1)
                            continue
                        self.logger.warning(f"⚠️ Could not set margin type for {symbol}: {e}")
                        break
        except Exception as e:
            self.logger.warning(f"⚠️ Could not read current margin type for {symbol}: {e}")

        # ── Set leverage ──
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
        except Exception as e:
            err_str = str(e)
            # code -4161: leverage reduction not supported in isolated margin with open positions
            if "4161" in err_str:
                self.logger.warning(f"⚠️ Leverage reduction not supported with open positions in isolated margin — skipping for {symbol}")
            else:
                self.logger.error(f"❌ Set leverage failed for {symbol}: {e}")
            return False

        # ── Verify ──
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if positions:
                confirmed_leverage = int(positions[0].get("leverage", 0))
                confirmed_margin = positions[0].get("marginType", "unknown")
                if confirmed_leverage != leverage:
                    self.logger.warning(
                        f"⚠️ Leverage mismatch for {symbol}: requested={leverage}x confirmed={confirmed_leverage}x"
                    )
                else:
                    self.logger.info(
                        f"✅ Leverage confirmed: {confirmed_leverage}x ({confirmed_margin}) for {symbol}"
                    )
        except Exception as e:
            self.logger.warning(f"⚠️ Could not verify leverage for {symbol}: {e}")

        return True