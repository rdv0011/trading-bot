from __future__ import annotations

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
            self._broker.close_position(self._symbol, abs(live.amount))
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
            return

        self._state.amount += add_qty
        self._state.scale_count += 1
        self.log(
            f"📈 SCALE UP {signal.upper()} +{add_qty} "
            f"(total={self._state.amount}, scale#{self._state.scale_count})"
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

        self._broker.close_position(self._symbol, close_qty)
        self._state.amount -= close_qty
        self.log(f"🔽 PARTIAL CLOSE -{close_qty} (remaining={self._state.amount:.4f})")

        if self._state.amount < MIN_TRADEABLE_QUANTITY:
            self._state = None

    def _full_close(self, reason: str):
        if self._state is None:
            return

        self._broker.cancel_open_orders(self._symbol, max_retries=3, base_delay=0.5)
        self._broker.close_position(self._symbol, self._state.amount)
        self.log(f"🔵 FULL CLOSE reason={reason}")
        self._state = None

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
