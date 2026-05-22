# Architecture Analysis — Trading Bot
**Date:** 2026-05-20  
**Branch:** `feature/dual-ml-system`  
**Scope:** Full codebase (4,301 lines across 21 files)  
**Standard:** Production algorithmic trading systems (FreqTrade, QuantConnect, Zipline patterns)

Each finding is graded **A** (architectural gap), **R** (risk / correctness), or **D** (design improvement).

---

## Executive Summary

The codebase is well-structured for a personal trading bot and shows clear thinking in the two-tier ML design. The core ML pipeline is sound, the broker abstraction is clean, and the hot-swap model reload is a genuinely good pattern. However, there are **six systemic gaps** that separate it from a production-grade system:

1. No position reconciliation after hard kill (`kill -9`, OOM, power loss) — graceful shutdown already closes positions via `on_abrupt_closing()`, but `finally` does not run on a hard kill
2. No risk management layer — no circuit breakers, no daily loss limit, no max drawdown stop
3. Polling architecture with no candle-alignment — the bot can trade on stale data
4. `PositionManager` state is in-memory only — a hard kill loses position tracking
5. Strategic labels are rule-based, not learned — the "ML" model learns a deterministic function
6. `train_test_split(shuffle=True)` on time-series data in `mltraining.py` — data leakage

---

## 1. Position Reconciliation on Restart — R (Medium)

### What exists
Both strategies implement `on_abrupt_closing()`, which is called from `BaseStrategy.shutdown()` inside a `finally` block — so it runs on `KeyboardInterrupt`, normal exit, and any unhandled exception that propagates out of the main loop.

**`MLStrategy.on_abrupt_closing()`** (`mlstrategy.py:357`) queries the exchange directly:
```python
position = self.get_position(self.asset)   # live broker call, not _state
if position is not None and abs(position.amount) >= MIN_TRADEABLE_QUANTITY:
    self.cancel_open_orders(self.asset)
    self.close_position(self.asset, position.amount)
```

**`DualMLStrategy.on_abrupt_closing()`** (`dualmlstrategy.py:132`) delegates to `PositionManager.emergency_close()`, which calls `broker.cancel_open_orders()` and `broker.close_position()` using `_state.amount`. This is slightly weaker: if `_state` is already `None` (e.g. the position was closed by a TP/SL bracket order between the last iteration and the crash), `emergency_close()` is a no-op and the exchange is not queried.

### Remaining gap
The crash handler covers **graceful shutdown** (Ctrl-C, unhandled exception). It does **not** cover the case where the process is killed hard (`kill -9`, OOM killer, power loss, Mac force-quit) — in those cases `finally` does not run. On restart after a hard kill:

- `PositionManager._state` is `None`
- The bot treats itself as flat
- If a real position is still open (bracket orders not yet triggered), the next signal opens a second position on top of it

This is a lower-probability scenario than originally assessed, but still a real risk for an always-on bot.

### Fix
Add startup reconciliation in `PositionManager.__init__` to cover the hard-kill case. The `MLStrategy` pattern of querying the exchange directly is the right model:

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
        self.log(f"♻️ Reconciled existing {side} position on startup: qty={live.amount} entry={live.entry_price}")
    else:
        self.log("✅ No open position found on startup — starting flat")
```

Call this at the end of `PositionManager.__init__`. This mirrors what `MLStrategy.on_abrupt_closing()` already does on the way *out* — querying the live exchange rather than trusting in-memory state.

---

## 2. No Risk Management Layer — R (Critical)

### Problem
There is no circuit breaker, no daily loss limit, no max drawdown stop, and no position size cap relative to account equity. The only risk controls are per-trade TP/SL (set by StrategicML) and the `allow_trading` veto.

**What's missing vs production standards:**

| Control | Present | Notes |
|---------|---------|-------|
| Per-trade stop-loss | ✅ | Via bracket order |
| Per-trade take-profit | ✅ | Via bracket order |
| Max position size (% of equity) | ✅ | `max_exposure_frac` |
| Daily loss limit | ❌ | No tracking |
| Max drawdown circuit breaker | ❌ | No tracking |
| Consecutive loss counter | ❌ | No tracking |
| Max open positions | ❌ | Implicit (1 position) |
| Leverage cap | ❌ | StrategicML can recommend any value |
| Correlation / concentration limit | ❌ | Single asset only |

### Industry Standard
FreqTrade has `max_open_trades`, `stoploss`, `trailing_stop`, `max_drawdown` (kills bot if exceeded), and `daily_loss_limit`. QuantConnect has a `RiskManagementModel` interface that is called before every order.

### Fix
Add a `RiskGuard` class that wraps `PositionManager.on_signal()`:

```python
@dataclass
class RiskGuard:
    max_daily_loss_frac: float = 0.05      # 5% of starting equity
    max_drawdown_frac: float = 0.15        # 15% from peak
    max_leverage: float = 10.0
    _peak_equity: float = field(default=0.0)
    _daily_start_equity: float = field(default=0.0)
    _daily_loss: float = field(default=0.0)
    _halted: bool = field(default=False)

    def check(self, current_equity: float) -> bool:
        """Returns False if trading should be halted."""
        ...
```

---

## 3. Polling Architecture — No Candle Alignment — A

### Problem
`BaseStrategy.run()` sleeps for `sleep_seconds` (e.g. 300s for 5m candles) after each iteration. This means:

1. **Drift**: If `on_trading_iteration()` takes 45 seconds (model training), the next iteration starts 345 seconds after the previous one, not 300. Over 24 hours this drifts by ~1.5 candles.
2. **Stale data**: The bot fetches the last N candles from Binance. The most recent candle may be incomplete (e.g. 2 minutes into a 5-minute candle). The bot is trading on a partial candle.
3. **No candle-close alignment**: Production systems wait for the candle to *close* before acting. Trading on an open candle means the features computed from it will differ from what the model was trained on (which used closed candles).

### Industry Standard
FreqTrade uses `timeframe_to_next_date()` to compute the exact time until the next candle close and sleeps until then. QuantConnect uses an event-driven model where `OnData()` is called exactly when a new bar closes.

### Fix
Replace the fixed sleep with alignment to the next candle boundary:

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

---

## 4. Strategic Labels Are Rule-Based, Not Learned — D

### Problem
`strategictraining.py:_build_strategic_labels()` generates training labels using **hardcoded rules**:

```python
df["recommended_leverage"] = df["regime"].map({"trend": 5.0, "high_vol": 2.0, "chop": 1.0})
df["stake_long_frac"] = np.where(df["regime"] == "trend", 0.2, 0.1)
df["stop_loss_frac"] = np.where(df["regime"] == "high_vol", 0.03, ...)
```

The `MultiOutputRegressor(CatBoostRegressor)` then learns to reproduce these exact rules. This means:
- The model adds zero predictive value over the rules themselves
- The model will never generalize beyond the three regimes
- Validation RMSE will be near-zero (the model perfectly memorizes the rules)
- The entire training pipeline is a very expensive `if/elif/else`

### Industry Standard
Strategic labels should come from **backtested outcomes**: for each historical window, run a parameter search and record which parameters actually produced the best risk-adjusted returns. This is what `mltraining.py:walkforward_label_forward_windows()` does for the tactical model — the same approach should be applied to the strategic model.

### Fix
Generate strategic labels by running `simulate_trades_core()` over a grid of `(leverage, stake_frac, stop_loss, take_profit, max_hold)` combinations for each historical window, and label each candle with the parameters that produced the best Sharpe/objective score. This is already implemented for the tactical pipeline — it needs to be ported to `strategictraining.py`.

---

## 5. Data Leakage in `mltraining.py` — R (High)

### Problem
`train_best_param_multi_model()` at line 394 uses:

```python
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=test_size, random_state=random_state
)
```

`train_test_split` **shuffles by default** (`shuffle=True`). For time-series data this is data leakage: the validation set contains rows from the middle of the time series, and the training set contains rows from the future relative to some validation rows. The model learns future information.

The same issue exists in `strategictraining.py:_train_strategic_model()` at line 121.

### Industry Standard
Time-series splits must be temporal. Use `sklearn.model_selection.TimeSeriesSplit` or a manual temporal split (`iloc[:n_train]` / `iloc[n_train:]`). The temporal split is already used correctly in `run_training()` at lines 187–189 — but then `_train_strategic_model()` re-shuffles the data it receives.

### Fix
```python
# In _train_strategic_model() and train_best_param_multi_model():
X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
Y_train, Y_val = Y.iloc[:n_train], Y.iloc[n_train:]
# where n_train = int(len(X) * 0.8)
```

---

## 6. `TacticalML.fit_and_predict()` Trains on the Prediction Target — R

### Problem
`tacticalml.py:72-78`:

```python
def fit_and_predict(self, df: pd.DataFrame, features: List[str]) -> TacticalSignal:
    X = df[features]
    y = df[TARGET_COLUMN]   # future_ret — computed by make_labels()
    mdl.fit(X, y)
    last_row = X.iloc[[-1]]
    prediction = float(mdl.predict(last_row)[0])
```

`df` is the full tactical window including the **last row**, which has `future_ret` computed from `close.shift(-H)`. For the last `H` rows, `future_ret` is `NaN` and is dropped by `make_labels()`. So the last row in `df` has a valid `future_ret` — meaning the model is trained on data that includes the candle it is predicting for.

More precisely: the model trains on rows `[0..N]` where row `N` has `future_ret` = return over the next `H` candles from now. Then it predicts on row `N`. This is **not** look-ahead bias in the traditional sense (the feature values at row N are all past data), but the training target for row N is future data. The model is being asked to predict something it was just trained to reproduce.

### Fix
Exclude the last row from training:

```python
X_train = df[features].iloc[:-1]
y_train = df[TARGET_COLUMN].iloc[:-1]
mdl.fit(X_train, y_train)
X_pred = df[features].iloc[[-1]]
prediction = float(mdl.predict(X_pred)[0])
```

---

## 7. `PositionManager` Quote Symbol Extraction — R

### Problem
`positionmanager.py:126`:

```python
cash = self._broker.get_cash(self._symbol.replace(self._asset, ""))
```

`self._symbol = "BTCUSDT"`, `self._asset = "BTC"` → `"BTCUSDT".replace("BTC", "") = "USDT"`. This works for BTCUSDT but is fragile:
- `"BTCBTC".replace("BTC", "")` → `""` (double match)
- Any asset whose ticker appears in the quote symbol would break

### Fix
Store `self._quote_symbol` directly in `__init__`:

```python
def __init__(self, broker, asset, quote_symbol, logger=None):
    self._quote_symbol = quote_symbol
    ...

def _open_position(self, ...):
    cash = self._broker.get_cash(self._quote_symbol)
```

---

## 8. `open_position_with_bracket` Has a Race Condition — R

### Problem
`binancebasebroker.py:120-123`:

```python
time.sleep(0.5)
position = self.get_position(symbol)
entry_price = position.entry_price if position is not None else self.get_last_price()
```

If the market order is not filled within 0.5 seconds (possible during high volatility or API latency), `get_position()` returns `None` and `entry_price` falls back to `get_last_price()`. The TP/SL prices are then computed from the *current market price*, not the actual fill price. On a fast-moving market this can place the stop-loss on the wrong side of the entry.

### Fix
Retry with exponential backoff until the position is confirmed, with a hard timeout:

```python
for attempt in range(5):
    time.sleep(0.2 * (2 ** attempt))
    position = self.get_position(symbol)
    if position and position.entry_price and position.entry_price > 0:
        entry_price = position.entry_price
        break
else:
    # Could not confirm fill — close and abort
    self.close_position(symbol, quantity)
    return BracketResult(success=False, error="Fill confirmation timeout")
```

---

## 9. `MlPredictor` Is a Dead Code Path — D

### Problem
`mlpredictor.py` (272 lines) implements a full predictor class with hot-swap model loading, prediction history, and adaptive thresholding. It is **never imported or used** by `DualMLStrategy` or `MLStrategy`. `MLStrategy` uses it (via `mlstrategy.py`), but `DualMLStrategy` — the new primary strategy — uses `TacticalML` and `StrategicML` directly.

`MlPredictor` duplicates logic that now lives in `TacticalML` (prediction history, adaptive thresholding, rolling retraining) and `StrategicML` (hot-swap loading). It is the legacy predictor from before the dual-ML refactor.

### Fix
Either:
- **Delete** `mlpredictor.py` if `MLStrategy` (legacy) is being deprecated
- **Or** document clearly that it is only used by the legacy `MLStrategy` and add a deprecation notice

---

## 10. `DualMLStrategy` Calls `make_labels()` on Live Data — R

### Problem
`dualmlstrategy.py:100`:

```python
df_tactical = make_features(df_tactical, self.tf_cfg_tactical)
df_tactical = make_labels(df_tactical, self.tf_cfg_tactical)   # ← this line
features = get_features(df_tactical)
```

`make_labels()` computes `future_ret = close.shift(-H)` and then calls `dropna()`. This drops the last `H` rows from `df_tactical`. In live trading, those last `H` rows are the most recent candles — the ones the model should be predicting on. After `dropna()`, the "last row" that `TacticalML.fit_and_predict()` predicts on is actually `H` candles old.

For `H = tf_cfg.label_horizon_candles` on a 5m timeframe with `label_horizon_minutes=100`, `H = 20` candles = 100 minutes. The bot is making trading decisions based on a prediction for a candle that is 100 minutes in the past.

### Fix
In live trading, labels are not needed for prediction — only for training. Separate the training data (which needs labels) from the prediction row (which does not):

```python
df_for_training = make_features(df_tactical, self.tf_cfg_tactical)
df_for_training = make_labels(df_for_training, self.tf_cfg_tactical)  # drops last H rows
features = get_features(df_for_training)

# Prediction row: most recent candle with features but no label needed
df_pred_row = make_features(df_tactical, self.tf_cfg_tactical).iloc[[-1]]

tactical_signal = self.tactical_ml.fit_and_predict(df_for_training, df_pred_row, features)
```

This is the most impactful correctness bug in the live trading path.

---

## 11. Broker Abstraction Leaks Exchange-Specific Concepts — A

### Problem
`BinanceBaseBroker` is named "Binance" — it is not a generic broker interface. The abstract methods reference Binance-specific concepts:
- `_create_bracket_order` — bracket orders are a Binance concept; other exchanges use OCO orders or separate TP/SL orders
- `cancel_open_orders(symbol, max_retries, base_delay)` — retry logic belongs in the implementation, not the interface
- `set_leverage` — futures-only concept leaking into the base class

The `from binance.enums import *` wildcard import in `binancebasebroker.py` pollutes the namespace and makes it impossible to swap the exchange without touching the base class.

### Fix
Define a pure `Broker` ABC with exchange-agnostic methods, then have `BinanceBaseBroker` implement it:

```python
class Broker(ABC):
    @abstractmethod
    def get_cash(self, quote_asset: str) -> float: ...
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[PositionResult]: ...
    @abstractmethod
    def place_bracket_order(self, symbol, side, qty, tp_frac, sl_frac) -> BracketResult: ...
    @abstractmethod
    def cancel_orders(self, symbol: str) -> None: ...
    @abstractmethod
    def close_position(self, symbol: str, qty: float) -> None: ...
    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...
```

---

## 12. No Observability — A

### Problem
The bot has no structured metrics, no alerting, and no way to monitor it remotely. The only output is log lines to stdout/file. There is no way to know:
- Current P&L
- Number of trades today
- Whether the bot is alive (heartbeat)
- Whether the strategic model is stale (last trained > N days ago)
- Whether Binance API calls are failing at elevated rates

### Industry Standard
Production trading bots emit structured metrics (Prometheus/StatsD) and send alerts (Telegram, PagerDuty) on critical events. FreqTrade has a Telegram bot integration and a REST API for monitoring.

### Fix (minimal viable)
Add a `Heartbeat` that writes a JSON status file every iteration:

```python
# status.json — written every candle
{
  "last_iteration_utc": "2026-05-20T02:05:01Z",
  "position": {"side": "long", "amount": 0.001, "entry_price": 95000},
  "strategic_model_age_hours": 12.3,
  "tactical_signal": "LONG",
  "strategic_allow": true,
  "errors_last_hour": 0
}
```

A simple cron job or external monitor can alert if `last_iteration_utc` is stale.

---

## 13. `simulate_trades_core` Cannot Model Partial Fills or Slippage Variance — D

### Problem
The simulation uses fixed `TAKER_FEE=0.0004` and `SLIPPAGE=0.0003` for every trade. In reality:
- Slippage is proportional to position size and market depth
- During high volatility, slippage can be 10–50x the assumed value
- The simulation never models a trade that fails to fill

This means backtest results are systematically optimistic during volatile periods — exactly when the bot is most likely to be trading (high_vol regime with 0.5x stake).

### Fix
Add a `slippage_model` parameter to `simulate_trades_core` that scales slippage by `vol_ratio`:

```python
slippage = SLIPPAGE * max(1.0, vol_ratio * 0.5)
```

---

## 14. `_check_and_reload` Calls `get_latest_model_paths` Every Candle — D

### Problem
`strategicml.py:_check_and_reload()` calls `get_latest_model_paths()` on every call to `predict()`, which is every 5-minute candle. `get_latest_model_paths()` calls `glob.glob()` on the model directory. This is a filesystem scan every 5 minutes — harmless now but will slow down if the model directory accumulates files.

### Fix
Cache the check with a time-based cooldown:

```python
_RELOAD_CHECK_INTERVAL = 300  # seconds

def _check_and_reload(self):
    now = time.time()
    if now - self._last_reload_check < _RELOAD_CHECK_INTERVAL:
        return
    self._last_reload_check = now
    # ... existing reload logic
```

---

## Priority Matrix

| # | Issue | Severity | Effort | Impact |
|---|-------|----------|--------|--------|
| 10 | `make_labels()` on live data — predicting stale candle | **Critical** | Small | Live trading correctness |
| 1 | No position reconciliation on restart (hard-kill only) | **Medium** | Small | Prevents duplicate positions after hard kill |
| 5 | `train_test_split` shuffle on time-series | **High** | Trivial | Backtest validity |
| 6 | TacticalML trains on prediction row | **High** | Small | Model quality |
| 2 | No risk management layer | **High** | Medium | Capital protection |
| 8 | Race condition in bracket order fill confirmation | **High** | Small | Order correctness |
| 3 | No candle-close alignment | **Medium** | Small | Signal quality |
| 4 | Strategic labels are rule-based | **Medium** | Large | Model value |
| 7 | Quote symbol string manipulation | **Medium** | Trivial | Robustness |
| 9 | `MlPredictor` dead code | **Low** | Trivial | Clarity |
| 11 | Broker abstraction leaks Binance | **Low** | Medium | Extensibility |
| 12 | No observability | **Low** | Medium | Operations |
| 13 | Fixed slippage model | **Low** | Small | Backtest accuracy |
| 14 | Filesystem scan every candle | **Low** | Trivial | Performance |
