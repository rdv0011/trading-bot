# Implementation Plan — Architecture Improvements
**Date:** 2026-05-20  
**Branch:** `feature/dual-ml-system`  
**Source:** `plan/architecture_analysis.md`

Items are ordered by execution sequence. Each task is self-contained and can be committed independently.

---

## Sequencing rationale

Tasks 1–3 are pure correctness fixes with no dependencies between them — they can be worked in parallel.  
Tasks 4–6 build on the corrected live-trading path established by task 3.  
Tasks 7–9 are independent improvements that do not touch the trading path.  
Tasks 10–11 are low-priority polish items.

---

## Task 1 — Align position closing: `DualMLStrategy.on_abrupt_closing` → query exchange directly

**Analysis ref:** §1  
**Files:** `dualmlstrategy.py`, `positionmanager.py`  
**Priority:** High  
**Effort:** Small

### Problem
`DualMLStrategy.on_abrupt_closing()` delegates to `PositionManager.emergency_close()`, which uses `_state.amount`. If `_state` is `None` (position already closed by a bracket order), the exchange is never queried and any residual position is silently left open.

`MLStrategy.on_abrupt_closing()` is the correct pattern: it calls `self.get_position()` directly from the exchange, independent of in-memory state.

### What to change

**`positionmanager.py` — add `emergency_close_live()`**

Add a second close method that queries the exchange directly, mirroring `MLStrategy`:

```python
def emergency_close_live(self):
    """
    Close any open position by querying the exchange directly.
    Used on shutdown — does not rely on _state.
    """
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
```

**`dualmlstrategy.py` — call `emergency_close_live()` instead of `emergency_close()`**

```python
def on_abrupt_closing(self):
    try:
        self.log_message("⚠️ Abrupt closing — emergency position close")
        self.position_manager.emergency_close_live()
    except Exception as e:
        self.log_message(f"❌ Emergency close error: {e}")
        self.log_message(traceback.format_exc())
```

### Verification
- `python -c "from positionmanager import PositionManager"` passes
- `python -c "from dualmlstrategy import DualMLStrategy"` passes
- Read `positionmanager.py` and confirm `emergency_close()` (in-memory path) is preserved for use by `on_signal` veto path; only `on_abrupt_closing` uses the new live method

---

## Task 2 — Add startup reconciliation to `PositionManager`

**Analysis ref:** §1  
**Files:** `positionmanager.py`  
**Priority:** High  
**Effort:** Small  
**Depends on:** Task 1 (adds `emergency_close_live`; reconciliation is independent but logically grouped)

### Problem
After a hard kill (`kill -9`, OOM, power loss), `_state` is `None` on restart. The bot opens a new position on top of any existing one.

### What to change

**`positionmanager.py` — add `_reconcile_on_startup()` and call it from `__init__`**

```python
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
```

Call at the end of `__init__`:
```python
def __init__(self, broker, asset, quote_symbol, logger=None):
    ...
    self._state: Optional[_PositionState] = None
    self._reconcile_on_startup()   # ← add this line
```

### Verification
- Import passes
- Manually verify: if `broker.get_position()` returns a position, `_state` is populated; if it returns `None`, `_state` remains `None`

---

## Task 3 — Fix quote symbol extraction in `PositionManager`

**Analysis ref:** §7  
**Files:** `positionmanager.py`  
**Priority:** Medium  
**Effort:** Trivial  
**Depends on:** None (independent, but group with Tasks 1–2 since same file)

### Problem
`self._symbol.replace(self._asset, "")` is fragile — breaks if the asset ticker appears in the quote symbol.

### What to change

**`positionmanager.py` — store `_quote_symbol` directly**

In `__init__`:
```python
def __init__(self, broker, asset, quote_symbol, logger=None):
    self._broker = broker
    self._asset = asset
    self._quote_symbol = quote_symbol          # ← add
    self._symbol = f"{asset}{quote_symbol}"
    ...
```

Replace both `get_cash` call sites:
```python
# _open_position and _scale_up:
cash = self._broker.get_cash(self._quote_symbol)   # replaces .replace() call
```

### Verification
- Grep confirms zero remaining `self._symbol.replace(self._asset` occurrences
- Import passes

---

## Task 4 — Fix `DualMLStrategy`: separate training data from prediction row

**Analysis ref:** §10 (most impactful correctness bug)  
**Files:** `dualmlstrategy.py`, `tactical/tacticalml.py`  
**Priority:** Critical  
**Effort:** Small  
**Depends on:** None

### Problem
`make_labels()` calls `dropna()` which removes the last `H=20` candles (100 minutes on 5m). The model predicts on a candle that is 100 minutes old. The bot trades on stale signals.

### What to change

**`tactical/tacticalml.py` — update `fit_and_predict` signature**

Accept a separate prediction row so the caller controls the split:

```python
def fit_and_predict(
    self,
    df_train: pd.DataFrame,   # labeled data for training (has TARGET_COLUMN)
    df_pred: pd.DataFrame,    # single-row feature df for prediction (no label needed)
    features: List[str],
) -> TacticalSignal:
    X_train = df_train[features].iloc[:-1]   # also exclude last training row (§6 fix)
    y_train = df_train[TARGET_COLUMN].iloc[:-1]

    seed = SEED_BASE + len(self._pred_history)
    mdl = create_model(self.model_cls, seed, self.model_params)
    mdl.fit(X_train, y_train)

    last_row = df_pred[features].iloc[[-1]]
    prediction = float(mdl.predict(last_row)[0])
    ...
```

**`dualmlstrategy.py` — split the dataframe before calling `fit_and_predict`**

```python
def on_trading_iteration(self):
    df_raw = self.get_historical_prices(...)

    # Training data: needs labels (drops last H rows via dropna)
    df_train = make_features(df_raw, self.tf_cfg_tactical)
    df_train = make_labels(df_train, self.tf_cfg_tactical)
    features = get_features(df_train)

    # Prediction row: most recent candle, features only
    df_pred = make_features(df_raw, self.tf_cfg_tactical).iloc[[-1]]

    tactical_signal = self.tactical_ml.fit_and_predict(df_train, df_pred, features)
    ...
```

**`dualmlstrategy.py` — update `initialize` warmup call to match new signature**

`TacticalML.warmup()` uses `df[features]` and `df[TARGET_COLUMN]` internally — its signature does not change. Only `fit_and_predict` changes.

### Verification
- `python -c "from dualmlstrategy import DualMLStrategy"` passes
- `python -c "from tactical.tacticalml import TacticalML"` passes
- Confirm `df_pred` index is the most recent timestamp in `df_raw`

---

## Task 5 — Fix `TacticalML.warmup`: exclude last row from each training window

**Analysis ref:** §6  
**Files:** `tactical/tacticalml.py`  
**Priority:** High  
**Effort:** Trivial  
**Depends on:** Task 4 (same file, group together)

### Problem
`warmup()` trains on `df.iloc[i - window : i]` which includes row `i-1`. Row `i-1` has `future_ret` computed from `close.shift(-H)` — its target is future data relative to the training window end. The model is trained to reproduce a value it should be predicting.

### What to change

**`tactical/tacticalml.py` — in `warmup()`, exclude the last row of each training slice**

```python
train_df = df.iloc[i - window : i - 1]   # was: df.iloc[i - window : i]
```

This ensures the model never sees the target for the candle at the boundary of its training window.

### Verification
- Warmup still completes without error
- `len(train_df)` is `window - 1` (acceptable — window is typically 50–240 candles)

---

## Task 6 — Fix data leakage: temporal split in `mltraining.py` and `strategictraining.py`

**Analysis ref:** §5  
**Files:** `mltraining.py`, `strategic/strategictraining.py`  
**Priority:** High  
**Effort:** Trivial  
**Depends on:** None

### Problem
`train_test_split(shuffle=True)` (the default) on time-series data. The validation set contains rows from the middle of the history; the training set contains rows from the future relative to some validation rows.

### What to change

**`mltraining.py` — `train_best_param_multi_model()` around line 394**

```python
# Replace:
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=test_size, random_state=random_state
)

# With:
n_train = int(len(X) * (1.0 - test_size))
X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
Y_train, Y_val = Y.iloc[:n_train], Y.iloc[n_train:]
```

**`strategic/strategictraining.py` — `_train_strategic_model()` around line 121**

```python
# Replace:
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=SEED_BASE
)

# With:
n_train = int(len(X) * 0.8)
X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
Y_train, Y_val = Y.iloc[:n_train], Y.iloc[n_train:]
```

Also remove the now-unused `from sklearn.model_selection import train_test_split` import from `strategictraining.py` if it is no longer used elsewhere in that file. Check `mltraining.py` — `train_test_split` may still be used in other functions; only remove the import if confirmed unused.

### Verification
- `python -c "from strategic.strategictraining import run_training"` passes
- `python -c "from mltraining import train_best_param_multi_model"` passes
- Grep confirms no remaining `train_test_split` calls on the affected functions

---

## Task 7 — Fix bracket order fill confirmation race condition

**Analysis ref:** §8  
**Files:** `binancebasebroker.py`  
**Priority:** High  
**Effort:** Small  
**Depends on:** None

### Problem
`time.sleep(0.5)` then single `get_position()` call. If the fill takes longer, `entry_price` falls back to `get_last_price()` and TP/SL are placed at the wrong prices.

### What to change

**`binancebasebroker.py` — replace single sleep+check with retry loop**

```python
# Replace the current block (lines ~118-128):
entry_price = None
for attempt in range(5):
    time.sleep(0.2 * (2 ** attempt))   # 0.2, 0.4, 0.8, 1.6, 3.2s
    position = self.get_position(symbol)
    if position and position.entry_price and position.entry_price > 0:
        entry_price = position.entry_price
        break

if entry_price is None:
    self.close_position(symbol, quantity)
    return BracketResult(success=False, error="Fill confirmation timeout after 5 attempts")
```

Total max wait: ~6.2 seconds before aborting. Remove the old `time.sleep(0.5)` line and the fallback `get_last_price()` call.

### Verification
- `python -c "from binancebasebroker import BinanceBaseBroker"` passes
- Read the modified block and confirm the old `get_last_price()` fallback is gone

---

## Task 8 — Add hot-swap reload cooldown to `StrategicML`

**Analysis ref:** §14  
**Files:** `strategic/strategicml.py`  
**Priority:** Low  
**Effort:** Trivial  
**Depends on:** None

### Problem
`glob.glob()` filesystem scan on every 5-minute candle.

### What to change

**`strategic/strategicml.py`**

```python
import time

_RELOAD_CHECK_INTERVAL = 300  # seconds — check at most once per 5m candle

class StrategicML:
    def __init__(self, ...):
        ...
        self._last_reload_check: float = 0.0

    def _check_and_reload(self):
        now = time.time()
        if now - self._last_reload_check < _RELOAD_CHECK_INTERVAL:
            return
        self._last_reload_check = now
        # ... existing reload logic unchanged
```

### Verification
- Import passes
- `_last_reload_check` initialised to `0.0` so the first call always runs

---

## Task 9 — Mark `MlPredictor` as legacy / deprecate

**Analysis ref:** §9  
**Files:** `mlpredictor.py`, `mlstrategy.py`  
**Priority:** Low  
**Effort:** Trivial  
**Depends on:** None

### Problem
`mlpredictor.py` (272 lines) is only used by the legacy `MLStrategy`. It duplicates logic now in `TacticalML` and `StrategicML`. It is invisible dead code from the perspective of the new dual strategy.

### What to change

Add a module-level deprecation notice at the top of `mlpredictor.py`:

```python
# LEGACY: Used only by MLStrategy (--strategy legacy).
# For the dual-ML strategy, see tactical/tacticalml.py and strategic/strategicml.py.
# This module will be removed when MLStrategy is retired.
```

No functional changes. Do not delete yet — `MLStrategy` still uses it.

### Verification
- `python -c "from mlpredictor import MlPredictor"` still passes

---

## Task 10 — Candle-close alignment in `BaseStrategy`

**Analysis ref:** §3  
**Files:** `basestrategy.py`  
**Priority:** Medium  
**Effort:** Small  
**Depends on:** None

### Problem
Fixed `sleep_seconds` causes drift when `on_trading_iteration()` takes variable time. The bot may also act on an incomplete (still-open) candle.

### What to change

**`basestrategy.py` — replace fixed sleep with candle-boundary alignment**

```python
def _sleep_until_next_candle(self, timeframe_minutes: int):
    now = datetime.utcnow()
    seconds_into_candle = (now.minute % timeframe_minutes) * 60 + now.second
    seconds_to_next = timeframe_minutes * 60 - seconds_into_candle + 5  # +5s buffer
    self.log_message(f"💤 Sleeping {seconds_to_next}s until next {timeframe_minutes}m candle close")
    elapsed = 0
    while self.is_running and elapsed < seconds_to_next:
        time.sleep(1)
        elapsed += 1
```

In `run()`, replace:
```python
# Old:
sleep_seconds = int(self.sleep_time[:-1]) * 60
self.log_message(f"💤 Sleeping for {sleep_seconds} seconds...")
elapsed = 0
while self.is_running and elapsed < sleep_seconds:
    time.sleep(1)
    elapsed += 1

# New:
timeframe_minutes = int(self.sleep_time[:-1])
self._sleep_until_next_candle(timeframe_minutes)
```

### Verification
- Import passes
- Manually verify: at minute 02:03 on a 5m timeframe, `seconds_to_next` = (5 - 2) * 60 - 3 + 5 = 182 seconds

---

## Task 11 — Add `RiskGuard` circuit breaker

**Analysis ref:** §2  
**Files:** `riskguard.py` (new file), `dualmlstrategy.py`, `positionmanager.py`  
**Priority:** High  
**Effort:** Medium  
**Depends on:** Tasks 1–3 (position state must be reliable before adding risk controls)

### Problem
No daily loss limit, no max drawdown circuit breaker, no consecutive loss counter.

### What to change

**New file `riskguard.py`**

```python
from dataclasses import dataclass, field
from datetime import date

@dataclass
class RiskGuard:
    max_daily_loss_frac: float = 0.05    # halt if daily loss exceeds 5% of start-of-day equity
    max_drawdown_frac: float = 0.15      # halt if drawdown from peak exceeds 15%
    max_leverage: float = 10.0           # cap leverage regardless of strategic recommendation

    _start_of_day_equity: float = field(default=0.0, init=False)
    _peak_equity: float = field(default=0.0, init=False)
    _last_date: date = field(default=None, init=False)
    _halted: bool = field(default=False, init=False)

    def update(self, current_equity: float) -> bool:
        """
        Update equity tracking. Returns True if trading is allowed, False if halted.
        """
        today = date.today()

        # Reset daily tracking on new day
        if self._last_date != today:
            self._start_of_day_equity = current_equity
            self._last_date = today

        # Update peak
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # Check daily loss
        if self._start_of_day_equity > 0:
            daily_loss = (self._start_of_day_equity - current_equity) / self._start_of_day_equity
            if daily_loss >= self.max_daily_loss_frac:
                self._halted = True

        # Check drawdown
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown >= self.max_drawdown_frac:
                self._halted = True

        return not self._halted

    def clamp_leverage(self, recommended: float) -> float:
        return min(recommended, self.max_leverage)

    @property
    def is_halted(self) -> bool:
        return self._halted
```

**`dualmlstrategy.py` — instantiate `RiskGuard` in `initialize()` and check in `on_trading_iteration()`**

```python
from riskguard import RiskGuard

def initialize(self):
    ...
    self.risk_guard = RiskGuard(
        max_daily_loss_frac=self.parameters.get("max_daily_loss_frac", 0.05),
        max_drawdown_frac=self.parameters.get("max_drawdown_frac", 0.15),
        max_leverage=self.parameters.get("max_leverage", 10.0),
    )

def on_trading_iteration(self):
    current_equity = self.get_cash()
    if not self.risk_guard.update(current_equity):
        self.log_message(f"🛑 RiskGuard halted trading — equity={current_equity:.2f}")
        if self.position_manager.has_position:
            self.position_manager.emergency_close_live()
        return

    # Clamp leverage before passing to position manager
    strategic_decision = self.strategic_ml.predict(df_strategic)
    strategic_decision = replace(
        strategic_decision,
        recommended_leverage=self.risk_guard.clamp_leverage(strategic_decision.recommended_leverage)
    )
    ...
```

### Verification
- `python -c "from riskguard import RiskGuard"` passes
- `python -c "from dualmlstrategy import DualMLStrategy"` passes
- Unit-test: `RiskGuard(max_daily_loss_frac=0.05).update(95.0)` after `update(100.0)` returns `False`

---

## Execution order summary

| Order | Task | Files touched | Severity fixed |
|-------|------|---------------|----------------|
| 1 | Align `on_abrupt_closing` — query exchange directly | `positionmanager.py`, `dualmlstrategy.py` | High |
| 2 | Startup reconciliation in `PositionManager` | `positionmanager.py` | Medium |
| 3 | Fix quote symbol extraction | `positionmanager.py` | Medium |
| 4 | Separate training data from prediction row | `dualmlstrategy.py`, `tactical/tacticalml.py` | Critical |
| 5 | Exclude last row from warmup training windows | `tactical/tacticalml.py` | High |
| 6 | Temporal split in training pipelines | `mltraining.py`, `strategic/strategictraining.py` | High |
| 7 | Bracket order fill confirmation retry | `binancebasebroker.py` | High |
| 8 | Hot-swap reload cooldown | `strategic/strategicml.py` | Low |
| 9 | Deprecate `MlPredictor` | `mlpredictor.py` | Low |
| 10 | Candle-close alignment | `basestrategy.py` | Medium |
| 11 | `RiskGuard` circuit breaker | `riskguard.py` (new), `dualmlstrategy.py` | High |

Tasks 1–3 touch only `positionmanager.py` and `dualmlstrategy.py` — commit together.  
Tasks 4–5 touch only `dualmlstrategy.py` and `tactical/tacticalml.py` — commit together.  
Tasks 6–9 are each independent single-file changes — commit individually.  
Task 10 touches only `basestrategy.py` — commit alone.  
Task 11 is the largest change — commit last after all others are verified.
