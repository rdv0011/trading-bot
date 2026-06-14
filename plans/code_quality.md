# Code Quality & Standards Improvement Plan

Analysed: all 16 Python source files  
Branch: `feature/dual-ml-system`  
Date: 2026-05-20

Each issue is graded **P1** (correctness / safety risk), **P2** (maintainability / reliability), or **P3** (polish / convention).

---

## 1. Dead code — P1

### 1.1 `BUY_SLIPPAGE` / `SELL_SLIPPAGE` in `mlstrategy.py`
`mlstrategy.py:12-13` defines two constants that are never referenced anywhere in the file or the rest of the codebase. Slippage is now modelled inside `simulate_trades_core` via `TAKER_FEE` + `SLIPPAGE` in `mltrainingcore.py`. These constants are misleading — a reader might think slippage is applied in live trading when it is not.

**Fix:** Delete lines 12–13 from `mlstrategy.py`.

---

## 2. Typo in public function name — P1

### 2.1 `caculate_metrics` → `calculate_metrics`
`mltrainingcore.py:331` defines `caculate_metrics` (missing `l`). It is called at line 327 and in `mltraining.py`. The typo is in a public function that is part of the shared simulation engine.

**Fix:** Rename to `calculate_metrics` everywhere.

---

## 3. Exception handling — P1

### 3.1 Swallowed exceptions with silent `pass`
The following locations catch an exception and do nothing, hiding real failures:

| File | Line | Problem |
|------|------|---------|
| `mlio.py` | 49–51 | Metadata write failure silently ignored |
| `mlio.py` | 64–65 | Old model file deletion failure silently ignored |
| `mlio.py` | 83–85 | Metadata load failure silently ignored |
| `strategic/strategicml.py` | 143 | `FileNotFoundError` on hot-swap silently ignored |
| `mltraining.py` | 762–763 | Final simulation CSV save failure silently ignored |

**Fix:** At minimum log a warning. For `mlio.py:49` (metadata write) and `mlio.py:83` (metadata load), the current behaviour is intentional (best-effort) but should be logged at `WARNING` level so failures are visible in cron logs.

### 3.2 `import traceback` inside `except` blocks — P2
`basestrategy.py:112`, `mlstrategy.py:390`, `mlpredictor.py:167`, `dualmlstrategy.py:137` all do `import traceback` inside an exception handler. This is a deferred import that works but is unconventional and slightly slower on the hot path.

**Fix:** Move `import traceback` to the top of each file.

### 3.3 `except Exception` too broad in broker methods — P2
`binancefuturesbroker.py` and `binancespotbroker.py` catch `Exception` on every broker method. This masks `KeyboardInterrupt` propagation issues and makes it impossible to distinguish network errors from logic errors.

**Fix:** Catch specific Binance exceptions (`BinanceAPIException`, `BinanceRequestException`) where possible, and re-raise unexpected exceptions after logging.

---

## 4. Logging — P1

### 4.1 `print()` used throughout training and I/O pipelines
`mlio.py`, `strategic/strategictraining.py`, and `mltraining.py` use `print()` for all output. When run from cron, this output goes to the log file but has no timestamps, no severity levels, and no structured format. A failure at 2 AM is indistinguishable from normal output.

**Fix:** Replace all `print()` calls in `mlio.py` and `strategic/strategictraining.py` with a module-level `logger = logging.getLogger(__name__)`. Use `logger.info()` for progress, `logger.warning()` for non-fatal issues, `logger.error()` for failures. Configure a formatter with timestamps in the cron entry:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
```

### 4.2 Mixed emoji + structured logging in live strategy
`dualmlstrategy.py`, `mlstrategy.py`, and `positionmanager.py` use emoji-prefixed strings via `self.log_message()`. This is fine for human readability but makes log parsing (e.g. grep for errors) unreliable.

**Fix:** Keep emoji for human-readable output but add a consistent severity prefix so automated monitoring can filter: `[INFO]`, `[WARN]`, `[ERROR]` before the emoji.

---

## 5. Type annotations — P2

### 5.1 87 functions missing return type annotations
The AST scan found 87 functions with no return type. The most impactful missing annotations are on public API boundaries:

| File | Functions |
|------|-----------|
| `mltrainingcore.py` | `make_features`, `make_labels`, `simulate_trades_core`, `calculate_metrics`, `adaptive_thresholding` |
| `mlio.py` | `save_model`, `load_model`, `get_latest_model_paths`, `load_featured_df`, `load_labels` |
| `binancebasebroker.py` | `cancel_open_orders`, `close_position`, `_fetch_klines` |
| `positionmanager.py` | `on_signal`, `emergency_close`, `_open_position`, `_scale_up`, `_partial_close`, `_full_close` |

### 5.2 41 functions with unannotated parameters
Key examples: `simulate_trades_core(df, df_hist, signal_col, param_list)` — none of these have types. `df` could be any DataFrame; `param_list` could be a dict or list of dicts (this ambiguity is actually a source of a real bug, see §8).

**Fix:** Add `pd.DataFrame`, `list[dict]`, `str`, `float` annotations progressively, starting with the public simulation and I/O functions. Use `from __future__ import annotations` for forward references.

---

## 6. Configuration management — P2

### 6.1 Magic numbers scattered across source files
The scan found 63 magic number candidates. The most impactful ones that should be named constants:

| Location | Value | Should be |
|----------|-------|-----------|
| `mltrainingcore.py:48` | `(20, 100)` EMA spans hardcoded in `make_features` | `REGIME_EMA_SHORT`, `REGIME_EMA_LONG` |
| `mltrainingcore.py:61` | `48` in `vol_48` rolling window | `VOL_LONG_WINDOW` |
| `mlpredictor.py:122` | `50` minimum window | `MIN_WARMUP_CANDLES` |
| `mlpredictor.py:125` | `12` retrain interval | `RETRAIN_EVERY_N_CANDLES` |
| `basestrategy.py:43` | `50` safety margin | `HISTORY_SAFETY_MARGIN` |
| `basestrategy.py:114` | `60` error retry sleep | `ERROR_RETRY_SLEEP_SECONDS` |
| `positionmanager.py` | `MAX_SCALE_COUNT=3`, `SCALE_INCREMENT_FRAC=0.5`, `PARTIAL_CLOSE_FRAC=0.33` | Already named — good. Move to config. |

### 6.2 No central configuration file
Trading parameters (`stake_frac`, `stop_loss`, `leverage`, `model_params`) are spread across `main.py`, `mlstrategy.py`, `dualmlstrategy.py`, and `strategic/strategictraining.py`. There is no single place to tune the bot without editing source code.

**Fix:** Introduce a `config.py` (or `config.yaml` loaded at startup) that centralises all tuneable parameters. `main.py` reads from it; strategy classes receive a typed config object rather than a raw `dict`.

### 6.3 API keys default to empty string on missing env vars
`main.py:12-15` and `strategictraining.py:166-167` use `os.getenv("KEY", "")`. An empty string is silently passed to the Binance client, which then fails with a cryptic authentication error rather than a clear startup failure.

**Fix:** Use `os.getenv("KEY")` and raise `ValueError("BINANCE_TESTNET_FUTURES_API_KEY not set")` at startup if the value is `None` or empty.

---

## 7. Sleep-based main loop — P2

### 7.1 `time.sleep()` blocks the entire process
`basestrategy.py:105` calls `time.sleep(sleep_seconds)` (up to 300 seconds). During this sleep the process cannot respond to signals, cannot be gracefully shut down, and cannot be interrupted cleanly. `KeyboardInterrupt` during sleep works but leaves no opportunity for cleanup hooks.

**Fix:** Replace the monolithic sleep with a loop of short sleeps checking `self.is_running`:

```python
interval = 1  # seconds
elapsed = 0
while self.is_running and elapsed < sleep_seconds:
    time.sleep(interval)
    elapsed += interval
```

This allows the shutdown flag to be respected within 1 second of being set.

---

## 8. Logic issues — P1

### 8.1 `get_param_row` silently falls back to first param on out-of-range index
`mltrainingcore.py:131-140`: when `param_list` is a list and the index is out of range, the function returns `param_list[0]` silently. In a walk-forward simulation this means the last few candles silently use the first parameter set rather than raising an error. This is a hidden source of incorrect backtest results.

**Fix:** Log a warning when the fallback is triggered, or raise `IndexError` in strict mode.

### 8.2 `PositionManager` passes `self._symbol` to `get_cash()` instead of quote symbol
`positionmanager.py:130`: `self._broker.get_cash(self._symbol.replace(self._asset, ""))` — this string manipulation to extract the quote symbol from the trading pair is fragile. If the asset symbol appears in the quote symbol (e.g. a hypothetical `USDTUSDT` pair) it would produce wrong results. The quote symbol is already known at construction time.

**Fix:** Store `self._quote_symbol` directly in `__init__` and use it in `get_cash()`.

### 8.3 `simulate_trades_core` uses `regime` variable after loop ends
`mltrainingcore.py:300`: the `final_close` trade appended after the loop uses `regime` which holds the value from the last iteration. If the loop body `continue`d on the last candle (chop regime), `regime` is still set correctly, but this is fragile and undocumented.

**Fix:** Use `df_iter.iloc[-1]['regime']` explicitly for the final close trade.

---

## 9. Model hot-swap race condition — P2

### 9.1 No atomicity guarantee on model file write
`mlio.py:43`: `joblib.dump(model, model_fname)` writes directly to the final filename. If the training process is killed mid-write, the running bot's next hot-swap check will find a corrupt `.pkl` file and crash. The metadata `.json` is written separately, creating a window where the `.pkl` exists but the `.meta.json` does not.

**Fix:** Write to a temporary file first, then atomically rename:

```python
import tempfile, shutil
tmp = model_fname + ".tmp"
joblib.dump(model, tmp)
shutil.move(tmp, model_fname)  # atomic on same filesystem
```

---

## 10. `mlio.py` path resolution — P2

### 10.1 `MODEL_DIR` and `LABEL_DIR` resolve relative to `sys.argv[0]`
`mlio.py:13-15`: paths are computed from `os.path.abspath(sys.argv[0])`. When `mlio` is imported from a script in a subdirectory (e.g. `strategic/strategictraining.py`), `sys.argv[0]` is the training script's path, so `MODEL_DIR` resolves to `strategic/model/` instead of `model/`. This is currently papered over by `sys.path.insert(0, ...)` in `strategictraining.py` but is fragile.

**Fix:** Resolve paths relative to `__file__` (the `mlio.py` file itself), not `sys.argv[0]`:

```python
_HERE = Path(__file__).resolve().parent
MODEL_DIR = _HERE / "model"
LABEL_DIR = _HERE / "labeleddata"
```

---

## 11. `TimeframeConfig` class attribute vs instance attribute — P3

### 11.1 `ema_hours` is a class attribute, not a dataclass field
`timeframe_config.py:21`: `ema_hours = (0.5, 1.25, 5, 20)` is defined as a plain class attribute inside a `@dataclass(frozen=True)`. It is not a field, so it cannot be overridden per-instance and does not appear in `__repr__` or `__eq__`. This is inconsistent with all other attributes which are proper dataclass fields.

**Fix:** Convert to a proper field with a default:

```python
ema_hours: tuple = (0.5, 1.25, 5, 20)
```

---

## 12. `mltrainingcore.py` — `make_features` modifies caller's DataFrame — P2

### 12.1 `df = df.copy()` is present but `dropna()` returns a new object
`make_features` and `make_labels` both call `df.copy()` at the start (good), but the `dropna()` at the end silently drops rows. The caller receives a shorter DataFrame than they passed in, with no indication of how many rows were dropped. In a live trading context this is fine, but in a training pipeline it can silently reduce the dataset.

**Fix:** Log the number of rows dropped: `logger.debug(f"make_features: dropped {len(df_in) - len(df_out)} rows with NaN")`.

---

## Implementation order

| Priority | Item | Effort |
|----------|------|--------|
| P1 | §8.1 `get_param_row` silent fallback | Small |
| P1 | §2.1 Rename `caculate_metrics` | Small |
| P1 | §1.1 Delete dead `BUY_SLIPPAGE`/`SELL_SLIPPAGE` | Trivial |
| P1 | §6.3 Fail fast on missing API keys | Small |
| P1 | §9.1 Atomic model file write | Small |
| P1 | §10.1 Fix `MODEL_DIR`/`LABEL_DIR` path resolution | Small |
| P2 | §4.1 Replace `print()` with `logging` in `mlio.py` and training scripts | Medium |
| P2 | §7.1 Interruptible sleep loop in `BaseStrategy` | Small |
| P2 | §8.2 Fix `get_cash()` quote symbol extraction | Small |
| P2 | §6.2 Central `config.py` | Medium |
| P2 | §5.1/5.2 Type annotations on public API | Large |
| P3 | §11.1 `ema_hours` dataclass field | Trivial |
| P3 | §3.2 Move `import traceback` to top of files | Trivial |
| P3 | §4.2 Consistent log severity prefixes | Small |
