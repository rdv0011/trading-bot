from __future__ import annotations

import time as _time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from binancebasebroker import (
    BinanceBaseBroker,
    MIN_TRADEABLE_QUANTITY,
    SIGNAL_HOLD,
    SIGNAL_LONG,
    SIGNAL_SHORT,
)
from tactical.tacticalml import TacticalSignal

TRADEABLE_QUANTITY_PRECISION = 3
MAX_SCALE_COUNT = 3
SCALE_INCREMENT_FRAC = 0.5
PARTIAL_CLOSE_FRAC = 0.33
CONSECUTIVE_SIGNALS_REQUIRED = 2
MIN_LIQUIDATION_BUFFER_FRAC = 0.008  # 0.8% minimum gap between SL trigger and liquidation


@dataclass
class StrategicDecision:
    allow_trading: bool
    market_regime: str
    volatility_state: str
    recommended_leverage: float
    max_exposure_frac: float
    stake_long_frac: float
    stake_short_frac: float
    stop_loss_frac: float
    take_profit_frac: float
    max_hold_hours: float
    margin_type: str = "ISOLATED"
    confidence: float = 1.0


@dataclass
class _PositionState:
    side: str
    amount: float
    entry_price: float
    entry_time: datetime
    scale_count: int = 0
    signal_history: deque = field(default_factory=lambda: deque(maxlen=10))


class PositionManager:
    def __init__(
        self,
        broker: BinanceBaseBroker,
        asset: str,
        quote_symbol: str,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self._broker = broker
        self._asset = asset
        self._quote_symbol = quote_symbol
        self._symbol = f"{asset}{quote_symbol}"
        self.log = logger if logger is not None else print
        self._state: Optional[_PositionState] = None
        self._reconcile_on_startup()

    def on_signal(
        self,
        tactical: TacticalSignal,
        strategic: StrategicDecision,
        current_price: float,
    ):
        if not strategic.allow_trading:
            self.log("🚫 StrategicML veto — trading blocked")
            if self._state is not None:
                self._full_close("STRATEGIC_VETO")
            return

        if strategic.market_regime == "chop":
            self.log("⏸ Chop regime — no new entries")
            return

        signal = tactical.signal

        if self._state is None:
            if signal != SIGNAL_HOLD:
                ok = self._broker.set_leverage(
                    self._symbol,
                    int(strategic.recommended_leverage),
                    strategic.margin_type,
                )
                if not ok:
                    self.log("⚠️ Leverage setup failed — skipping entry")
                    return
                self._open_position(signal, current_price, strategic)
            return

        self._state.signal_history.append(signal)

        is_long = self._state.side == SIGNAL_LONG
        same_direction = (is_long and signal == SIGNAL_LONG) or (
            not is_long and signal == SIGNAL_SHORT
        )
        opposite_direction = (is_long and signal == SIGNAL_SHORT) or (
            not is_long and signal == SIGNAL_LONG
        )

        elapsed_hours = (
            datetime.now() - self._state.entry_time
        ).total_seconds() / 3600.0

        if elapsed_hours >= strategic.max_hold_hours:
            self._full_close("MAX_HOLD_TIME")
            return

        if same_direction:
            consecutive = self._count_consecutive_tail(signal)
            if (
                consecutive >= CONSECUTIVE_SIGNALS_REQUIRED
                and self._state.scale_count < MAX_SCALE_COUNT
            ):
                self._scale_up(signal, current_price, strategic)
            return

        if opposite_direction:
            self._partial_close(current_price)

    def emergency_close(self):
        if self._state is not None:
            self._full_close("EMERGENCY")

    def emergency_close_live(self):
        try:
            self._broker.cancel_open_orders(self._symbol, max_retries=3, base_delay=0.5)
        except Exception as exc:
            self.log(f"⚠️ cancel_open_orders failed during emergency close: {exc}")

        live = self._broker.get_position(self._symbol)
        if live and live.amount and abs(live.amount) >= MIN_TRADEABLE_QUANTITY:
            # Pass signed amount: positive for LONG (sell), negative for SHORT (buy to cover)
            self._broker.close_position(self._symbol, live.amount)
            self.log(f"🔴 EMERGENCY CLOSE (live) qty={abs(live.amount)}")
        else:
            self.log("✅ No open position on exchange during emergency close")

        self._state = None

    def _reconcile_on_startup(self):
        live = self._broker.get_position(self._symbol)
        if live and live.amount and abs(live.amount) >= MIN_TRADEABLE_QUANTITY:
            side = SIGNAL_LONG if live.amount > 0 else SIGNAL_SHORT
            self._state = _PositionState(
                side=side,
                amount=abs(live.amount),
                entry_price=live.entry_price,
                entry_time=datetime.now(),
            )
            self.log(
                f"♻️ Reconciled existing {side} position on startup: "
                f"qty={abs(live.amount)} entry={live.entry_price}"
            )
        else:
            self.log("✅ No open position found on startup — starting flat")

    def _check_liquidation_buffer(
        self,
        sl_price: float,
        entry_price: float,
        context: str = "",
    ):
        """Warn if the stop-loss trigger is dangerously close to the liquidation price."""
        try:
            liq_price = self._broker.get_liquidation_price(self._symbol)
            if liq_price is None or liq_price <= 0:
                return  # exchange didn't report a liquidation price

            if self._state and self._state.side == SIGNAL_SHORT:
                # For SHORT: SL is above entry, liquidation is above SL
                buffer_pct = (liq_price - sl_price) / entry_price
            else:
                # For LONG: SL is below entry, liquidation is below SL
                buffer_pct = (sl_price - liq_price) / entry_price

            if buffer_pct < MIN_LIQUIDATION_BUFFER_FRAC:
                self.log(
                    f"⚠️ LIQUIDATION RISK ({context}): "
                    f"SL={sl_price:.2f} liq={liq_price:.2f} "
                    f"buffer={buffer_pct*100:.2f}% — below {MIN_LIQUIDATION_BUFFER_FRAC*100:.1f}% threshold"
                )
            else:
                self.log(
                    f"  🛡 Liq buffer OK ({context}): "
                    f"{buffer_pct*100:.2f}% gap between SL and liquidation"
                )
        except Exception as exc:
            self.log(f"⚠️ Could not check liquidation buffer: {exc}")

    def _open_position(
        self, signal: str, price: float, strategic: StrategicDecision
    ):
        cash = self._broker.get_cash(self._quote_symbol)
        stake_frac = (
            strategic.stake_long_frac
            if signal == SIGNAL_LONG
            else strategic.stake_short_frac
        )
        qty = round(
            (cash * stake_frac * strategic.max_exposure_frac) / price,
            TRADEABLE_QUANTITY_PRECISION,
        )

        if qty < MIN_TRADEABLE_QUANTITY:
            self.log("⚠️ Quantity below minimum, skipping entry")
            return

        res = self._broker.open_position_with_bracket(
            self._symbol,
            signal,
            qty,
            tp_frac=strategic.take_profit_frac,
            sl_frac=strategic.stop_loss_frac,
        )

        if not res.success:
            self.log(f"❌ Entry failed: {res.error}")
            return

        entry_price = res.data["entry_price"]
        self._state = _PositionState(
            side=signal,
            amount=qty,
            entry_price=entry_price,
            entry_time=datetime.now(),
        )
        self._state.signal_history.append(signal)
        self.log(f"🟢 OPEN {signal.upper()} @ {entry_price} qty={qty}")

        # Warn if liquidation is too close to stop-loss
        sl_price = res.data.get("sl_price")
        if sl_price:
            self._check_liquidation_buffer(sl_price, entry_price, context=f"entry@{entry_price}")

    def _scale_up(self, signal: str, price: float, strategic: StrategicDecision):
        cash = self._broker.get_cash(self._quote_symbol)
        stake_frac = (
            strategic.stake_long_frac
            if signal == SIGNAL_LONG
            else strategic.stake_short_frac
        )
        add_qty = round(
            (cash * stake_frac * SCALE_INCREMENT_FRAC) / price,
            TRADEABLE_QUANTITY_PRECISION,
        )

        if add_qty < MIN_TRADEABLE_QUANTITY:
            self.log("⚠️ Scale-up quantity below minimum, skipping")
            return

        # Record position size before scale-up to verify later
        pos_before = self._broker.get_position(self._symbol)
        qty_before = abs(pos_before.amount) if pos_before else 0.0

        # Cancel existing TP/SL orders before placing new bracket to avoid -4130
        self._broker.cancel_open_orders(self._symbol, max_retries=3, base_delay=0.5)

        res = self._broker.open_position_with_bracket(
            self._symbol,
            signal,
            add_qty,
            tp_frac=strategic.take_profit_frac,
            sl_frac=strategic.stop_loss_frac,
        )

        if not res.success:
            self.log(f"❌ Scale-up failed: {res.error}")
            # Verify actual position — open_position_with_bracket may have closed
            # the full position on TP/SL failure; reset state if flat
            live = self._broker.get_position(self._symbol)
            if live is None or abs(live.amount) < MIN_TRADEABLE_QUANTITY:
                self.log("⚠️ Position was fully closed during failed scale-up — resetting state")
                self._state = None
            elif abs(live.amount) <= qty_before:
                # Position was NOT increased (market order failed or was rejected)
                self.log(
                    f"⚠️ Scale-up market order likely failed — "
                    f"position unchanged at {abs(live.amount):.4f}"
                )
            else:
                # Position increased but TP/SL failed — position is unprotected!
                self.log(
                    f"⚠️ Scale-up added {abs(live.amount) - qty_before:.4f} but TP/SL placement failed — "
                    f"position has no stop-loss protection!"
                )
            return

        # Verify position actually increased
        live = self._broker.get_position(self._symbol)
        if live is None or abs(live.amount) <= qty_before + MIN_TRADEABLE_QUANTITY:
            self.log(
                f"⚠️ Scale-up market order did not increase position "
                f"(before={qty_before:.4f}, after={abs(live.amount) if live else 0:.4f})"
            )
            return

        actual_added = abs(live.amount) - qty_before
        self._state.amount = abs(live.amount)
        self._state.scale_count += 1
        self.log(
            f"📈 SCALE UP {signal.upper()} +{actual_added:.3f} "
            f"(total={self._state.amount:.4f}, scale#{self._state.scale_count})"
        )
        if res.success:
            tp_price = res.data.get("tp_price", "?")
            sl_price = res.data.get("sl_price", "?")
            self.log(f"  🛡 TP={tp_price} SL={sl_price}")
            if sl_price and isinstance(sl_price, (int, float)):
                self._check_liquidation_buffer(
                    sl_price,
                    res.data.get("entry_price", price),
                    context=f"scale-up#{self._state.scale_count}",
                )

    def _partial_close(self, price: float):
        if self._state is None:
            return

        close_qty = round(
            self._state.amount * PARTIAL_CLOSE_FRAC, TRADEABLE_QUANTITY_PRECISION
        )

        if close_qty < MIN_TRADEABLE_QUANTITY:
            self._full_close("REVERSAL_FULL")
            return

        # Pass signed amount: positive for LONG (sell), negative for SHORT (buy to cover)
        signed_qty = (
            close_qty
            if self._state.side == SIGNAL_LONG
            else -close_qty
        )
        self._broker.close_position(self._symbol, signed_qty)
        self._state.amount -= close_qty
        self.log(f"🔽 PARTIAL CLOSE -{close_qty} (remaining={self._state.amount:.4f})")

        if self._state.amount < MIN_TRADEABLE_QUANTITY:
            self._state = None

    def _full_close(self, reason: str):
        if self._state is None:
            return

        live_before = self._broker.get_position(self._symbol)
        pos_before = abs(live_before.amount) if live_before else 0.0

        self._broker.cancel_open_orders(self._symbol, max_retries=3, base_delay=0.5)

        # Pass signed amount: positive for LONG (sell), negative for SHORT (buy to cover)
        signed_qty = (
            self._state.amount
            if self._state.side == SIGNAL_LONG
            else -self._state.amount
        )
        self._broker.close_position(self._symbol, signed_qty)

        # Verify close succeeded by checking exchange (retry up to 3 times)
        live = None
        for attempt in range(3):
            live = self._broker.get_position(self._symbol)
            if live is None or abs(live.amount) < MIN_TRADEABLE_QUANTITY:
                self.log(f"🔵 FULL CLOSE reason={reason}")
                self._state = None
                return
            if attempt < 2:
                delta = abs(live.amount) - pos_before if live_before else 0
                self.log(
                    f"⚠️ FULL CLOSE retry #{attempt + 1}: "
                    f"remaining={abs(live.amount):.4f} (delta={delta:+.4f}), retrying..."
                )
                # Use signed amount based on exchange-reported position direction
                signed_qty = (
                    abs(live.amount)
                    if live.amount > 0
                    else -abs(live.amount)
                )
                self._broker.close_position(self._symbol, signed_qty)
                _time.sleep(0.5 * (2 ** attempt))
                live_before = live

        remaining = abs(live.amount) if live else 0.0
        self.log(
            f"⚠️ FULL CLOSE reason={reason} — FAILED after 3 attempts, "
            f"remaining={remaining:.4f}"
        )

    def _count_consecutive_tail(self, signal: str) -> int:
        count = 0
        for s in reversed(self._state.signal_history):
            if s == signal:
                count += 1
            else:
                break
        return count

    @property
    def has_position(self) -> bool:
        return self._state is not None

    @property
    def position_side(self) -> Optional[str]:
        return self._state.side if self._state else None
