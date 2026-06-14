# Current Issues — Analysis & Fix Plan

**Date:** 2026-06-11
**Branch:** (current)
**Status:** Analysis complete — ready for implementation

---

## Issue 1: RiskGuard equity=0 — lack of diagnostic logging and no deactivation mechanism

### Problem

When RiskGuard halts trading with `equity=0.00`, there is no logging explaining **why** it halted (daily loss limit vs drawdown limit). Additionally, once halted, RiskGuard stays halted **permanently** — there is no mechanism to auto-reset on a new day or to deactivate programmatically.

**Observed log:**
```
2026-06-11 04:40:06,873 - INFO - 🛑 RiskGuard halted trading — equity=0.00
```

### Root cause analysis

**File:** `riskguard.py` (43 lines)

1. **No reason logged on halt** — `update()` sets `_halted = True` but never logs which threshold was breached or the current metric values.

2. **Halt is permanent** — `_halted` is set once and never reset. There is no `reset()` method, no new-day reset, no cooldown. The only way to unhalt is to create a new `RiskGuard` instance (restart the bot).

3. **Equity=0 bypasses guard logic** — When `current_equity = 0.0`:
   - `self._start_of_day_equity = 0.0` → `self._start_of_day_equity > 0` is `False` → daily loss check is **skipped**
   - `self._peak_equity` stays at `0.0` if never above 0 → `self._peak_equity > 0` is `False` → drawdown check is **skipped**
   - Result: if equity starts at 0, RiskGuard never actually checks any limits, but `_halted` stays `False`... Wait — that contradicts the log.

   Re-tracing: `on_trading_iteration` calls `self.get_cash()` which returns USDT balance. If the balance was e.g. 100, then dropped to 0, the sequence is:
   - Day 1, equity=100 → `_start_of_day_equity=100, _peak_equity=100`
   - Later, equity=0 → `daily_loss = (100-0)/100 = 1.0` → `1.0 >= 0.05` → `_halted = True` ✓
   - So equity=0 **does** trigger correctly on the drop. The log just doesn't say WHY.

4. **The `_start_of_day_equity > 0` guard is a silent skip** — If for any reason `_start_of_day_equity` is 0 (e.g. first call of the day when wallet is empty), the entire daily loss check is silently bypassed with no logging.

### Fix plan

#### 1.1 — Add diagnostic logging on halt (`riskguard.py`)

In `update()`, before returning, log the reason:
- "🛑 RiskGuard halted — daily loss {daily_loss:.2%} >= max {max_daily_loss_frac:.2%}"
- "🛑 RiskGuard halted — drawdown {drawdown:.2%} >= max {max_drawdown_frac:.2%}"
- Include `current_equity`, `start_of_day_equity`, `peak_equity` in the log

#### 1.2 — Add new-day auto-reset of halt state (`riskguard.py`)

In `update()`, when the date changes (`self._last_date != today`):
- Reset `_halted = False` (if it was set)
- Log: "🔄 New day — RiskGuard halt lifted (start_of_day_equity={current_equity})"

This gives a natural deactivation: once a new day starts, the circuit breaker resets and trading can resume.

#### 1.3 — Add explicit `reset()` method (`riskguard.py`)

```python
def reset(self, current_equity: float):
    """Manually reset RiskGuard state — e.g. after restart or manual override."""
    self._halted = False
    self._start_of_day_equity = current_equity
    self._peak_equity = current_equity
    self._last_date = date.today()
```

#### 1.4 — Add logging for silent skips (`riskguard.py`)

When `_start_of_day_equity == 0` on a new day: log a warning.
When `_peak_equity == 0` on first call: log a warning.

#### 1.5 — Update tests (`tests/test_riskguard.py`)

- Test that new day resets `_halted` to `False`
- Test `reset()` method
- Test logging output (optional — can verify via capsys)

---

## Issue 2: FULL CLOSE leads to strange Sell/Buy orders — missing order instrumentation

### Problem

When a `FULL CLOSE` is triggered (e.g. `MAX_HOLD_TIME`), the sequence can produce unexpected orders: "Sell 0.016 BTC and Buy 0.016 BTC in a second". There is insufficient logging to trace what happened.

**Observed log:**
```
2026-06-10 01:30:16,134 - INFO - ⚠️ FULL CLOSE retry #1: remaining=0.0160 (delta=+0.0000), retrying...
2026-06-10 01:30:17,171 - INFO - 🔵 FULL CLOSE reason=MAX_HOLD_TIME
2026-06-10 01:30:17,174 - INFO - 💤 Sleeping 288s until next 5m candle close
```

> **Note:** The retry message on line 1 (`⚠️ FULL CLOSE retry #1`) does **not** exist in the current codebase — it was from a previous version with retry logic. This confirms the logging gap.

### Root cause analysis

**File:** `positionmanager.py` (275 lines)

The `_full_close` method:
```python
def _full_close(self, reason: str):
    if self._state is None:
        return
    self._broker.cancel_open_orders(self._symbol, max_retries=3, base_delay=0.5)
    self._broker.close_position(self._symbol, self._state.amount)
    self.log(f"🔵 FULL CLOSE reason={reason}")
    self._state = None
```

**Race condition scenario:**

1. `cancel_open_orders` cancels the TP/SL bracket orders (`closePosition=True` on futures)
2. But a bracket order may have **already triggered** (SL hit) and is being filled by the exchange
3. `close_position` places a market order to close the position
4. The already-triggered bracket order fills simultaneously with the market order
5. Both try to close the same position → position goes **negative** (overshoot)
6. The overshoot creates an opposite position → the bot places a second order to correct

**Specific to Binance Futures:**
- Bracket orders use `closePosition=True` — they close the entire position when triggered
- If the trigger happens between `cancel_open_orders` and `close_position`, the position is already gone
- The market order in `close_position` then opens a **new opposite position** (selling BTC you don't have in a long → short)
- This explains "Sell 0.016 BTC and Buy 0.016 BTC": the market order overshoots, then the next signal buys back

**Additional scenario without race condition:**
- `close_position` uses `abs(position)` and determines side by sign: `SIDE_SELL if position > 0 else SIDE_BUY`
- If `_state.amount` is stale or incorrect, the close could go the wrong direction
- No verification happens after close

### Fix plan

#### 2.1 — Add pre-close instrumentation logging (`positionmanager.py`)

In `_full_close`, log the full state before any action:
```python
self.log(f"🔵 FULL CLOSE reason={reason} — side={self._state.side} "
         f"amount={self._state.amount:.4f} entry={self._state.entry_price}")
```

#### 2.2 — Add cancel result logging (`positionmanager.py`)

Wrap `cancel_open_orders` with try/except and log success/failure:
```python
try:
    self._broker.cancel_open_orders(self._symbol, max_retries=3, base_delay=0.5)
    self.log(f"🔵 FULL CLOSE — cancelled open orders for {self._symbol}")
except Exception as e:
    self.log(f"⚠️ FULL CLOSE — cancel_open_orders failed: {e}")
```

#### 2.3 — Add close order placement logging (`positionmanager.py`)

Log the exact order that will be placed:
```python
close_side = "SELL" if self._state.side == SIGNAL_LONG else "BUY"
self.log(f"🔵 FULL CLOSE — placing {close_side} market order qty={self._state.amount:.4f}")
```

#### 2.4 — Add post-close verification (`positionmanager.py`)

After closing, query the exchange to verify the position is gone:
```python
try:
    live_pos = self._broker.get_position(self._symbol)
    if live_pos and live_pos.amount and abs(live_pos.amount) >= MIN_TRADEABLE_QUANTITY:
        self.log(f"⚠️ FULL CLOSE — position still open after close: "
                 f"qty={abs(live_pos.amount):.4f} side={'LONG' if live_pos.amount > 0 else 'SHORT'}")
        # Optional: retry close
except Exception as e:
    self.log(f"⚠️ FULL CLOSE — verification failed: {e}")
```

#### 2.5 — Add order-level logging in broker (`binancefuturesbroker.py`)

In `close_position()`, log the exact order being placed and its result:
```python
def close_position(self, symbol: str, position: float):
    try:
        side = SIDE_SELL if position > 0 else SIDE_BUY
        self.logger.info(f"🔵 close_position: {symbol} {side} qty={abs(position):.4f}")
        order = self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=abs(position)
        )
        self.logger.info(f"🔵 close_position result: orderId={order.get('orderId')} "
                        f"status={order.get('status')} "
                        f"executedQty={order.get('executedQty', 'N/A')}")
    except Exception as e:
        self.logger.error(f"❌ Close position failed: {e}")
```

#### 2.6 — Same instrumentation for `BinanceSpotBroker.close_position()` (`binancespotbroker.py`)

Apply the same logging pattern.

#### 2.7 — Update tests (`tests/test_positionmanager.py`)

- Ensure existing tests pass with new logging
- Add test for the verification logic (if feasible with mocks)

---

## Implementation order

| Task | File(s) | Dependencies | Effort |
|------|---------|-------------|--------|
| 1.1 RiskGuard halt logging | `riskguard.py` | None | Small |
| 1.2 New-day auto-reset | `riskguard.py` | None | Trivial |
| 1.3 Reset method | `riskguard.py` | None | Trivial |
| 1.4 Skip logging | `riskguard.py` | None | Trivial |
| 1.5 Test updates | `tests/test_riskguard.py` | 1.1–1.4 | Small |
| 2.1 Pre-close logging | `positionmanager.py` | None | Trivial |
| 2.2 Cancel result logging | `positionmanager.py` | None | Trivial |
| 2.3 Close order logging | `positionmanager.py` | None | Trivial |
| 2.4 Post-close verification | `positionmanager.py` | None | Small |
| 2.5 Broker order logging | `binancefuturesbroker.py` | None | Small |
| 2.6 Spot broker logging | `binancespotbroker.py` | None | Small |
| 2.7 Test updates | `tests/test_positionmanager.py` | 2.1–2.4 | Small |

Tasks 1.x and 2.x are **independent** and can be worked in parallel.
