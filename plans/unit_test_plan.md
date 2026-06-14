# Unit Test Plan
**Date:** 2026-05-20
**Branch:** `feature/architecture-improvements`
**Status:** Ready to implement

---

## Guiding principles

1. **No network, no filesystem, no real broker** — every external call is mocked via `unittest.mock`.
2. **Pure functions first** — functions with no side effects are the highest-value, lowest-effort tests.
3. **State machines second** — classes with internal state (`RiskGuard`, `PositionManager`) are tested by driving them through all meaningful state transitions.
4. **Integration seams last** — the broker interface (`open_position_with_bracket`) is tested with a mock broker to verify orchestration logic without hitting Binance.
5. **No ML training in tests** — `TacticalML.fit_and_predict` and `TacticalML.warmup` require a real model; they are excluded from unit tests (covered by integration/smoke tests separately).

---

## Test infrastructure

### Directory layout
```
tests/
  __init__.py
  conftest.py                        # shared fixtures (mock broker, sample DataFrames)
  test_riskguard.py
  test_positionmanager.py
  test_mltrainingcore.py
  test_timeframe_config.py
  test_strategicfeatures.py
  test_broker_bracket.py
```

### Dependencies (already in conda env)
- `pytest` — test runner
- `unittest.mock` — stdlib, no extra install
- `pandas`, `numpy` — already present

### `conftest.py` — shared fixtures
```python
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from binancebasebroker import PositionResult, BracketOrderResult, MarketOrderResult

@pytest.fixture
def sample_ohlcv():
    """200-row synthetic OHLCV DataFrame with DatetimeIndex."""
    idx = pd.date_range("2024-01-01", periods=200, freq="5min")
    rng = np.random.default_rng(42)
    close = 40000 + np.cumsum(rng.normal(0, 50, 200))
    df = pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.002,
        "low":    close * 0.998,
        "close":  close,
        "volume": rng.uniform(1, 10, 200),
    }, index=idx)
    return df

@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.get_cash.return_value = 1000.0
    broker.get_position.return_value = None
    broker.open_position_with_bracket.return_value = MagicMock(
        success=True, data={"entry_price": 40000.0}
    )
    return broker
```

---

## Module-by-module breakdown

---

### 1. `riskguard.py` → `tests/test_riskguard.py`

**Why:** Pure state machine, zero external dependencies. Every branch is reachable with simple numeric inputs. Highest test ROI in the codebase.

#### Test cases

| ID | Description | Setup | Assert |
|----|-------------|-------|--------|
| RG-01 | First call initialises start-of-day equity | `rg.update(1000)` | returns `True`; `_start_of_day_equity == 1000` |
| RG-02 | Below daily loss threshold → allowed | `update(1000)` then `update(960)` (4% loss) | returns `True` |
| RG-03 | Exactly at daily loss threshold → halted | `update(1000)` then `update(950)` (5% loss) | returns `False`; `is_halted == True` |
| RG-04 | Exceeds daily loss threshold → halted | `update(1000)` then `update(900)` (10% loss) | returns `False` |
| RG-05 | Drawdown below threshold → allowed | `update(1000)`, `update(1100)`, `update(940)` (14.5% from peak) | returns `True` |
| RG-06 | Drawdown at threshold → halted | `update(1000)`, `update(1100)`, `update(935)` (15% from peak 1100) | returns `False` |
| RG-07 | Once halted, stays halted even if equity recovers | halt via RG-03, then `update(1100)` | returns `False` |
| RG-08 | New day resets start-of-day equity | mock `date.today()` to return tomorrow; call `update(800)` | `_start_of_day_equity == 800`; returns `True` (fresh day) |
| RG-09 | `clamp_leverage` below cap → unchanged | `clamp_leverage(5.0)` with `max_leverage=10.0` | returns `5.0` |
| RG-10 | `clamp_leverage` above cap → clamped | `clamp_leverage(15.0)` with `max_leverage=10.0` | returns `10.0` |
| RG-11 | `clamp_leverage` exactly at cap → unchanged | `clamp_leverage(10.0)` | returns `10.0` |
| RG-12 | Peak equity tracks correctly | `update(1000)`, `update(1200)`, `update(1100)` | `_peak_equity == 1200` |

**Mocking needed:** `date.today()` for RG-08 only — use `unittest.mock.patch("riskguard.date")`.

---

### 2. `positionmanager.py` → `tests/test_positionmanager.py`

**Why:** Core trading state machine with 6 distinct state transitions. All broker calls are mockable. The reconciliation and emergency close paths are critical correctness fixes from this branch.

#### Fixtures
```python
@pytest.fixture
def pm(mock_broker):
    from positionmanager import PositionManager
    # get_position returns None → starts flat
    mock_broker.get_position.return_value = None
    return PositionManager(mock_broker, "BTC", "USDT", logger=lambda _: None)

@pytest.fixture
def strategic():
    from positionmanager import StrategicDecision
    return StrategicDecision(
        allow_trading=True, market_regime="trend", volatility_state="normal",
        recommended_leverage=5.0, max_exposure_frac=0.5,
        stake_long_frac=0.1, stake_short_frac=0.05,
        stop_loss_frac=0.02, take_profit_frac=0.04, max_hold_hours=4.0,
    )
```

#### Test cases

| ID | Description | Assert |
|----|-------------|--------|
| PM-01 | Startup with no live position → `_state` is None | `pm.has_position == False` |
| PM-02 | Startup with live LONG position → `_state` populated | `pm.has_position == True`; `pm.position_side == "long"` |
| PM-03 | Startup with live SHORT position → `_state` populated | `pm.position_side == "short"` |
| PM-04 | `on_signal` with HOLD and no position → no broker call | `open_position_with_bracket` not called |
| PM-05 | `on_signal` LONG with no position → opens long | `open_position_with_bracket` called once with `"long"` |
| PM-06 | `on_signal` SHORT with no position → opens short | called with `"short"` |
| PM-07 | `on_signal` with `allow_trading=False` and open position → full close | `close_position` called; `has_position == False` |
| PM-08 | `on_signal` with `allow_trading=False` and no position → no close call | `close_position` not called |
| PM-09 | `on_signal` chop regime → no new entry | `open_position_with_bracket` not called |
| PM-10 | Same-direction signal, 1 consecutive → no scale-up | `open_position_with_bracket` called once (entry only) |
| PM-11 | Same-direction signal, 2 consecutive → scale-up | `open_position_with_bracket` called twice |
| PM-12 | Opposite-direction signal → partial close | `close_position` called with `PARTIAL_CLOSE_FRAC * amount` |
| PM-13 | Max hold time exceeded → full close | set `entry_time` to 5h ago; `close_position` called |
| PM-14 | `emergency_close` with state → full close | `close_position` called; `has_position == False` |
| PM-15 | `emergency_close` with no state → no broker call | `close_position` not called |
| PM-16 | `emergency_close_live` with live position → queries exchange and closes | `get_position` called; `close_position` called |
| PM-17 | `emergency_close_live` with no live position → no close call | `close_position` not called |
| PM-18 | `emergency_close_live` when `cancel_open_orders` raises → still attempts close | `close_position` still called |
| PM-19 | Broker entry failure → `_state` remains None | mock `open_position_with_bracket` to return `success=False` |
| PM-20 | `_quote_symbol` used in `get_cash` call | `get_cash` called with `"USDT"` not `"BTCUSDT".replace("BTC","")` |

**Mocking needed:** `mock_broker` fixture; `datetime.now()` for PM-13 (patch `positionmanager.datetime`).

---

### 3. `mltrainingcore.py` → `tests/test_mltrainingcore.py`

**Why:** Contains the most pure functions in the codebase. `detect_regime`, `adaptive_thresholding`, `calculate_metrics`, `get_param_row`, `time_to_candles` are all deterministic with no I/O.

#### Test cases

**`detect_regime`**

| ID | Description | Input row | Expected |
|----|-------------|-----------|----------|
| MC-01 | Weak trend → chop | `trend_strength < 0.6` | `"chop"` |
| MC-02 | Strong trend, low vol ratio → trend | `trend_strength > 0.6`, `vol_ratio < 1.4` | `"trend"` |
| MC-03 | Strong trend, high vol ratio → high_vol | `trend_strength > 0.6`, `vol_ratio > 1.4` | `"high_vol"` |
| MC-04 | ATR near zero → no division error | `atr14=0` | does not raise |

**`adaptive_thresholding`**

| ID | Description | Assert |
|----|-------------|--------|
| MC-05 | Series shorter than `num_candles` → returns `(nan, nan)` | both are `np.nan` |
| MC-06 | Sufficient series → max_th > min_th | `max_th > min_th` |
| MC-07 | All-same values → max_th == min_th | equal |

**`calculate_metrics`**

| ID | Description | Assert |
|----|-------------|--------|
| MC-08 | Empty trades list → score = -0.1 | `objective == -0.1`; `metrics["trades_count"] == 0` |
| MC-09 | All winning trades → win_rate = 1.0 | `metrics["win_rate"] == 1.0` |
| MC-10 | All losing trades → downside > 0 | `metrics["downside"] > 0` |
| MC-11 | Mixed trades → composite score formula | verify formula: `3*mean + 2*wr - 1*ds + 0.5*act` |
| MC-12 | Activity capped at 1.0 | 100 trades with `expected_trades=10` → `activity == 1.0` |

**`get_param_row`**

| ID | Description | Assert |
|----|-------------|--------|
| MC-13 | Single dict input → returns same dict | identity |
| MC-14 | List input, valid index → correct element | `param_list[2]` at `idx=2` |
| MC-15 | List input, out-of-range index → returns first | `param_list[0]` |
| MC-16 | Empty list → returns None | `None` |
| MC-17 | Invalid type → raises `TypeError` | `pytest.raises(TypeError)` |

**`time_to_candles`**

| ID | Description | Assert |
|----|-------------|--------|
| MC-18 | Minutes input → correct candle count | `time_to_candles(minutes=10, timeframe_minutes=5)` == 2 |
| MC-19 | Hours input → correct candle count | `time_to_candles(hours=1, timeframe_minutes=5)` == 12 |
| MC-20 | Result below min_candles → clamped | `time_to_candles(minutes=1, timeframe_minutes=5, min_candles=3)` == 3 |
| MC-21 | Neither minutes nor hours → raises `ValueError` | `pytest.raises(ValueError)` |

**`make_features` / `make_labels`** (integration-level, uses `sample_ohlcv` fixture)

| ID | Description | Assert |
|----|-------------|--------|
| MC-22 | `make_features` output has expected columns | `ret1`, `atr14`, `regime`, `hour_sin` all present |
| MC-23 | `make_features` drops NaN rows | no NaN in result |
| MC-24 | `make_labels` adds `future_ret` column | column present |
| MC-25 | `make_labels` result shorter than input | `len(result) < len(input)` (H rows dropped) |
| MC-26 | `get_features` excludes target columns | `future_ret`, `future_close`, `regime` not in result |

---

### 4. `timeframe_config.py` → `tests/test_timeframe_config.py`

**Why:** Frozen dataclass with derived properties. Pure math — no mocking needed.

#### Test cases

| ID | Description | Assert |
|----|-------------|--------|
| TC-01 | `candles_per_hour` for 5m → 12 | `== 12` |
| TC-02 | `candles_per_hour` for 1h → 1 | `== 1` |
| TC-03 | `label_horizon_candles` for 5m (100min horizon) → 20 | `== 20` |
| TC-04 | `adaptive_history_candles` for 5m (50h) → 600 | `== 600` |
| TC-05 | `ema_spans` for 5m → tuple of ints, all ≥ 1 | all positive ints |
| TC-06 | `candles(minutes=10)` on 5m → 2 | `== 2` |
| TC-07 | `candles(minutes=3)` on 5m → 1 (min clamp) | `== 1` |
| TC-08 | TIMEFRAMES preset "5m" has correct `minutes` | `== 5` |
| TC-09 | TIMEFRAMES preset "1h" has correct `minutes` | `== 60` |

---

### 5. `strategic/strategicfeatures.py` → `tests/test_strategicfeatures.py`

**Why:** `_classify_vol_state` is a pure function with 4 branches. `make_strategic_features` is testable with synthetic data.

#### Test cases

**`_classify_vol_state`**

| ID | Description | Input | Expected |
|----|-------------|-------|----------|
| SF-01 | Below 1.0 → low (0.0) | `0.8` | `0.0` |
| SF-02 | Between 1.0 and 1.6 → normal (1.0) | `1.2` | `1.0` |
| SF-03 | Between 1.6 and 2.5 → high (2.0) | `2.0` | `2.0` |
| SF-04 | At or above 2.5 → extreme (3.0) | `2.5` | `3.0` |
| SF-05 | Exactly at boundary 1.6 → high (2.0) | `1.6` | `2.0` |

**`make_strategic_features`** (uses `sample_ohlcv` fixture, needs ≥ 200 rows)

| ID | Description | Assert |
|----|-------------|--------|
| SF-06 | Output has `vol_state` column | present |
| SF-07 | `vol_state` values are in `{0.0, 1.0, 2.0, 3.0}` | all values in set |
| SF-08 | Output has `drawdown` column, all values ≤ 0 | `(df["drawdown"] <= 0).all()` |
| SF-09 | No NaN in output | `df.isna().sum().sum() == 0` |

---

### 6. `binancebasebroker.py` → `tests/test_broker_bracket.py`

**Why:** `open_position_with_bracket` is a concrete method on the abstract base class with complex branching (invalid signal, fill timeout, bracket failure). All abstract methods can be mocked via a concrete stub subclass.

#### Stub subclass
```python
class StubBroker(BinanceBaseBroker):
    def setup_client(self): pass
    def get_cash(self, *a): return 1000.0
    def get_position(self, *a): return None
    def get_last_price(self, *a): return 40000.0
    def _create_market_order(self, *a): ...  # set per test
    def _create_bracket_order(self, *a): ...  # set per test
    def cancel_open_orders(self, *a): pass
    def close_position(self, *a): pass
```

#### Test cases

| ID | Description | Assert |
|----|-------------|--------|
| BB-01 | Invalid signal → `success=False`, no market order | `error == "Invalid signal"` |
| BB-02 | `_create_market_order` returns None → `success=False` | `error` contains "None" |
| BB-03 | Entry price returned immediately → bracket placed at correct TP/SL | `tp_price == round(40000 * 1.02, 2)` for LONG |
| BB-04 | Entry price None, fill confirmed on 2nd retry → success | `success=True`; `get_position` called twice |
| BB-05 | Entry price None, all 5 retries fail → `success=False`, position closed | `close_position` called; `error` contains "timeout" |
| BB-06 | Bracket order fails → position closed, `success=False` | `close_position` called |
| BB-07 | SHORT signal → TP below entry, SL above entry | `tp_price < entry`; `sl_price > entry` |
| BB-08 | LONG signal → TP above entry, SL below entry | `tp_price > entry`; `sl_price < entry` |
| BB-09 | Exception in `_create_market_order` → `success=False` | `success=False`; `error` is non-empty string |

**Mocking needed:** `time.sleep` patched to no-op to keep tests fast (`patch("binancebasebroker.time.sleep")`).

---

## Execution order

| Priority | File | Reason |
|----------|------|--------|
| 1 | `test_riskguard.py` | Smallest, purest, highest confidence |
| 2 | `test_timeframe_config.py` | Pure math, zero mocking |
| 3 | `test_mltrainingcore.py` | Pure functions + DataFrame fixtures |
| 4 | `test_strategicfeatures.py` | Pure functions + DataFrame fixtures |
| 5 | `test_broker_bracket.py` | Concrete method, stub broker |
| 6 | `test_positionmanager.py` | Most complex, most mocking |

---

## What is explicitly excluded

| Module | Reason excluded |
|--------|----------------|
| `TacticalML.fit_and_predict` / `warmup` | Requires real CatBoost training — slow, non-deterministic; belongs in integration tests |
| `StrategicML.predict` | Requires a trained model file on disk |
| `mltraining.py` (full pipeline) | End-to-end training pipeline; integration test territory |
| `strategictraining.py` | Same — requires Binance data fetch |
| `binancefuturesbroker.py` / `binancespotbroker.py` | Thin wrappers over Binance SDK; tested via broker integration tests |
| `DualMLStrategy.on_trading_iteration` | Requires full broker + model stack; end-to-end test |
| `mlio.py` (filesystem ops) | `save_model`/`load_model` require temp dirs; belongs in integration tests with `tmp_path` fixture |

---

## Coverage targets

| Module | Target line coverage |
|--------|---------------------|
| `riskguard.py` | 100% |
| `timeframe_config.py` | 95%+ |
| `mltrainingcore.py` (pure functions) | 90%+ |
| `strategic/strategicfeatures.py` | 95%+ |
| `binancebasebroker.py` (`open_position_with_bracket`) | 90%+ |
| `positionmanager.py` | 80%+ |

---

## Implementation notes

- All tests use `pytest` style (functions, not classes) for readability.
- `conftest.py` holds all shared fixtures to avoid duplication.
- `time.sleep` must always be patched in broker tests — otherwise the exponential backoff loop adds ~6s per test.
- `date.today()` must be patched in `test_riskguard.py` for the new-day reset test (RG-08).
- `positionmanager.datetime` must be patched for the max-hold-time test (PM-13).
- No test should write to disk or make network calls.
