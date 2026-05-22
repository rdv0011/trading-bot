# Plan: Simulation-Driven Strategic Meta-Parameter Optimization

## Problem

`strategictraining.py` currently labels training data using hard-coded heuristics:

```python
stake_long_frac  = 0.2 if trend else 0.1
stop_loss_frac   = 0.03 if high_vol else 0.015 if trend else 0.02
max_hold_hours   = 8 if trend else 2 if high_vol else 4
```

StrategicML learns to replicate these rules — not to find parameters that actually maximize
trading performance. The legacy system solved this by labeling each window with the param set
that produced the best objective score in a walk-forward simulation. This plan applies the
same approach to the dual system.

## Goal

Replace rule-based label generation in `strategictraining.py` with a walk-forward simulation
loop that labels each 1h window with the param set that maximized the objective score when
applied to the tactical signal (`pred`) over that window. StrategicML then learns to predict
those simulation-grounded params from market features.

## Execution Order Dependency

The walk-forward optimizer needs the tactical `pred` column to already exist. The correct
pipeline order is:

```
Step A  dualmlsimulation.py (predictions only)
          → labeleddata/dual_BTCUSDT_5m_45d_predictions.csv

Step B  strategictraining.py (new walk-forward labeling + train)
          reads:  dual_*_predictions.csv  (5m tactical preds, resampled to 1h)
          writes: strategic_*_sim_labeled.csv
          writes: model/strategic_meta_model_<ts>.pkl

Step C  dualmlsimulation.py (full backtest)
          reads:  model/strategic_meta_model_<ts>.pkl
          writes: dual_*_final_test_sim.csv
```

---

## Tasks

### Task 1 — Extract tactical predictions at 1h resolution

**File:** `dualmlsimulation.py`

Add a new exported function `run_predictions_only(symbol, days, timeframe, model_dir)` that
runs Steps 1–2 of the existing pipeline (data fetch + walk-forward tactical predictions) and
returns early, without running the strategic query or simulation. This allows `strategictraining.py`
to call it as a library function to obtain the `pred` column before training.

The function must:
- Reuse the existing `USE_SAVED_FEATURED` / `USE_SAVED_PREDICTIONS` cache flags
- Return `(df_predictions, df_raw)` — the full predictions DataFrame and the raw 5m OHLCV
- Not duplicate any existing logic — call `_rolling_tactical_predictions` internally

The existing `run_simulation` function must remain unchanged and continue to work as before.

**Acceptance:** `from dualmlsimulation import run_predictions_only` works without error.

---

### Task 2 — Add walk-forward param optimizer to `mltraining.py`

**File:** `mltraining.py`

Extract `walkforward_label_forward_windows` into a standalone importable function with a
cleaner signature. The existing function already does what is needed but is tightly coupled
to the legacy pipeline. No changes to the function body are required — only verify it is
importable and that its signature accepts:

```python
walkforward_label_forward_windows(
    df,           # DataFrame with signal_col already present
    param_grid,   # list of dicts
    signal_col,   # str — 'pred'
    window_hours, # float
    step_hours,   # float
    tf_cfg,       # TimeframeConfig
) -> pd.DataFrame  # date-indexed, columns: best_param_idx, best_param, best_metric
```

No code changes needed if the function is already importable. Confirm with a dry-run import.

**Acceptance:** `from mltraining import walkforward_label_forward_windows` works without error.

---

### Task 3 — Replace `_build_strategic_labels` with simulation-driven labeling

**File:** `strategic/strategictraining.py`

Replace the body of `_build_strategic_labels` with a new function
`_build_strategic_labels_from_simulation` that:

1. **Receives** `df_5m_predictions: pd.DataFrame` (the output of `run_predictions_only`) in
   addition to the existing `df: pd.DataFrame` (1h strategic features) and `tf_cfg`.

2. **Resamples** the 5m tactical predictions to 1h by taking the last `pred` value in each
   1h bucket:
   ```python
   pred_1h = df_5m_predictions['pred'].resample('1h').last()
   ```
   Then aligns it onto the 1h strategic feature DataFrame by index.

3. **Builds a param_grid** equivalent to the legacy system:
   ```
   stake_long_frac:  [0.10, 0.15, 0.25]
   stake_short_frac: [0.05, 0.10, 0.15]
   stop_loss_frac:   [0.01, 0.015, 0.02, 0.03, 0.05]
   take_profit_frac: stop_loss * 2.0
   max_hold_hours:   [2, 4, 8, 12, 24]
   ```

4. **Calls** `walkforward_label_forward_windows` (imported from `mltraining`) on the aligned
   1h DataFrame (which now has the `pred` column as signal) to produce `best_param` per window.

5. **Merges** the `best_param` labels back onto the 1h strategic feature DataFrame using
   `pd.merge_asof` (same pattern as the legacy pipeline in `mltraining.py` lines 259–271).

6. **Expands** each `best_param` dict into individual columns:
   `stake_long_frac`, `stake_short_frac`, `stop_loss_frac`, `take_profit_frac`, `max_hold_hours`

7. **Preserves** the non-optimized columns that StrategicML still needs:
   `allow_trading`, `recommended_leverage`, `max_exposure_frac`
   These continue to be derived from vol_ratio and regime as before (they are not part of the
   param_grid — they are risk-gate signals, not trading execution params).

8. **Returns** the labeled DataFrame with all columns present, ready for `_train_strategic_model`.

Update `run_training` to:
- Accept an optional `df_5m_predictions: pd.DataFrame = None` parameter
- When `df_5m_predictions` is provided, call `_build_strategic_labels_from_simulation`
- When `df_5m_predictions` is None, fall back to the existing `_build_strategic_labels`
  (preserves backward compatibility — cron job with no tactical data still works)
- Add a new cache file name: `strategic_{symbol}_{timeframe}_{days}d_sim_labeled.csv`
  (separate from the existing `_labeled.csv` so both label modes can coexist)

**Acceptance:**
- `run_training(..., df_5m_predictions=None)` produces identical output to today
- `run_training(..., df_5m_predictions=df)` runs without error and produces a labeled
  DataFrame where `stake_long_frac` values are drawn from the param_grid (not 0.1/0.2)

---

### Task 4 — Wire the full pipeline in `main.py`

**File:** `main.py`

Update the `--train-strategic` CLI path to run the full two-step pipeline:

1. Call `run_predictions_only(symbol, days='45', timeframe='5m')` to get `df_5m_predictions`
2. Call `run_training(symbol, days, timeframe='1h', df_5m_predictions=df_5m_predictions)`

Add a new CLI flag `--tactical-days` (default: 45) to control how many days of 5m data are
used for the walk-forward param search. This is separate from `--days` which controls the
1h strategic training window.

The existing `--train-strategic` path without the new flag must continue to work (falls back
to rule-based labels).

**Acceptance:** `python main.py --train-strategic` completes without error.

---

### Task 5 — Add unit tests

**File:** `tests/test_strategic_param_optimization.py`

Write tests covering:

1. `test_param_grid_structure` — `build_param_grid` returns list of dicts, each containing
   exactly the five keys: `stake_long_frac`, `stake_short_frac`, `stop_loss_frac`,
   `take_profit_frac`, `max_hold_hours`. No network calls.

2. `test_pred_1h_alignment` — Given a synthetic 5m DataFrame with a `pred` column,
   resampling to 1h via `.resample('1h').last()` produces the correct number of rows and
   the last 5m value in each hour is preserved.

3. `test_build_strategic_labels_fallback` — When `df_5m_predictions=None`,
   `run_training` calls `_build_strategic_labels` (rule-based path). Mock the file I/O
   and assert the labeled DataFrame contains `stake_long_frac` values from {0.1, 0.2} only.

4. `test_build_strategic_labels_from_simulation` — Given a small synthetic 1h DataFrame
   with a `pred` column and a minimal param_grid (2 param sets), assert that the returned
   labeled DataFrame has `stake_long_frac` values drawn from the param_grid, not the
   hard-coded heuristic values.

5. `test_non_optimized_columns_preserved` — After simulation-driven labeling,
   `allow_trading`, `recommended_leverage`, and `max_exposure_frac` are still present
   and contain values consistent with the vol_ratio/regime rules.

All tests must be offline (no Binance calls, no disk I/O beyond tmp).

**Acceptance:** `pytest tests/test_strategic_param_optimization.py` — all 5 tests pass.

---

## Files Changed

| File | Change |
|------|--------|
| `dualmlsimulation.py` | Add `run_predictions_only()` function |
| `strategic/strategictraining.py` | Add `_build_strategic_labels_from_simulation()`, update `run_training` signature |
| `main.py` | Wire two-step pipeline under `--train-strategic`, add `--tactical-days` flag |
| `tests/test_strategic_param_optimization.py` | New file — 5 tests |

`mltraining.py` and `mltrainingcore.py` are **not modified** — `walkforward_label_forward_windows`
and `simulate_trades_core` are reused as-is.

## Files NOT Changed

| File | Reason |
|------|--------|
| `mltraining.py` | Reused as a library — no changes needed |
| `mltrainingcore.py` | Reused as a library — no changes needed |
| `strategic/strategicml.py` | Inference path unchanged — model format identical |
| `strategic/strategicfeatures.py` | Feature engineering unchanged |
| `tactical/tacticalml.py` | Unchanged |
| `positionmanager.py` | Unchanged |
| `riskguard.py` | Unchanged |

---

## Key Design Decisions

**Why resample `pred` to 1h instead of running walk-forward at 5m?**
StrategicML operates at 1h. Running the param search at 5m granularity would produce
~12x more windows, making training prohibitively slow and creating a timeframe mismatch
between the signal used for optimization and the timeframe the model is deployed at.

**Why keep `allow_trading`, `recommended_leverage`, `max_exposure_frac` rule-based?**
These are risk-gate signals, not execution parameters. They answer "should we trade at all
and with how much capital?" — a question that is better answered by regime/vol state rules
than by optimizing for past returns (which would risk overfitting the gate to historical
volatility patterns).

**Why fall back to rule-based labels when `df_5m_predictions` is None?**
The cron job runs `strategictraining.py` standalone. Requiring it to first run
`dualmlsimulation.py` would break the existing cron setup. The fallback preserves the
current behavior for unattended retraining.

**Why `pd.merge_asof` for label alignment?**
Walk-forward windows produce one label per step (not per candle). `merge_asof` propagates
the most recent `best_param` forward to all candles until the next window boundary —
the same approach used in the legacy pipeline.
