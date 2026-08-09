# Plan: Demo Trading vs. Simulation Comparison (Aligned Time Window)

**Date:** 2026-07-27
**Branch:** profitability-improvements
**Status:** Implemented (Tasks 1–6, 7). Audit & tuning tools live; see §8 How to run.
**Objective:** Compare 8 days of demo (live) trading against a simulation of the *same* 8 calendar days, and explain why live makes far fewer trades.

---

## 1. Problem Statement

The live bot (`DualMLStrategy`) makes **far fewer trades** than `dualmlsimulation.py` on the same market conditions. Both use the same ML components (TacticalML, StrategicML), but they are **not the same strategy** — the live path has additional gates and different decision cadence that the simulation does not model. Any comparison requires:

1. **Extracting** the live trade record from 8 days of demo logs.
2. **Re-running** the simulation restricted to the exact same 8 calendar days (UTC).
3. **Explaining** the divergence by gate-level attribution.

---

## 2. Root-Cause Analysis: Why Live Makes Fewer Trades

### 2.1 Confirmed divergences between live and simulation

| # | Divergence | Live (`dualmlstrategy.py` + `positionmanager.py`) | Simulation (`simulate_trades_core` in `mltrainingcore.py`) | Impact on trade count |
|---|---|---|---|---|
| **D1** | **Chop regime handling** | **HARD BLOCK** — `GATE: chop_regime blocked` (positionmanager.py:80, dualmlstrategy.py:346) | `regime_stake_mult = {'chop': 0.3}` → trades at 30% stake | **HIGH** — biggest single cause |
| **D2** | **Volume filter** | Blocks entry if `vol < 0.5×SMA20` (chop) or `< 0.8×SMA20` (else) (dualmlstrategy.py:251-262) | **Not present** | HIGH |
| **D3** | **HTF trend filter (EMA50)** | Blocks LONG below EMA50 / SHORT above EMA50 on 1h (dualmlstrategy.py:271-298) | **Not present** | HIGH |
| **D4** | **RiskGuard circuit breaker** | Halts all trading on 5% daily loss / 15% drawdown | **Not present** | MEDIUM |
| **D5** | **Regime source mismatch** | Uses **model-predicted** `strategic_decision.market_regime` for the chop gate | Uses **rule-based** `detect_regime()` from features | MEDIUM — chop classification diverges |
| **D6** | **Scaling & partial closes** | `MAX_SCALE_COUNT=5`, `SCALE_INCREMENT_FRAC=1.0`, `PARTIAL_CLOSE_FRAC=0.33` | Single position in/out, no scaling | LOW for count, HIGH for PnL fidelity |
| **D7** | **Tactical retraining cadence** | Retrains `fit_and_predict` every new 15m candle, `iterations=300` | Walk-forward, retrains every 12 candles, `iterations=100, thread_count=1` | LOW — predictions differ slightly |
| **D8** | **Signal caching between candles** | Reuses cached signal on 5m heartbeats between 15m candles | Evaluates every 15m candle | LOW — decision frequency aligned at 15m |

### 2.2 Quantifying the gap

From D1–D5 alone, the live bot suppresses:
- All chop-regime signals (D1) — depending on regime distribution this can be 30–50% of candles
- All low-volume signals (D2)
- All counter-trend signals (D3) — up to 50% of remaining
- All post-halt signals (D4)

**The gate counters added in commit 316cd20 are the key diagnostic** — they already count exactly how many signals each gate suppresses per day in live trading. The comparison harness must surface these counters next to the simulation's trade count.

### 2.3 Critical caveat (recent change)

`mltrainingcore.py:193` currently sets `{'chop': 0.3}` in the simulation — **but live still blocks chop entirely**. This means the simulation *over-trades relative to live by design* in chop regimes. The comparison must either:
- (a) set the simulation's chop multiplier to `0.0` to mirror live behavior, or
- (b) expose it as a CLI parameter so both modes can be tested.

---

## 3. Alignment Methodology

### 3.1 Core idea

```
Demo logs (8 days)                     Simulation (same 8 days)
────────────────────                   ────────────────────────────
Parse trade events              ───►   Windowed sim over [start, end]
Parse gate counters / regime           Warmup history BEFORE window
Parse daily summaries                   (adaptive thresholds, strategic 1h history)
        │                                        │
        ▼                                        ▼
   demo_trades.csv                        sim_trades.csv
   demo_daily_summary.csv                 sim_daily_summary.csv
        └───────────────────┬────────────────────┘
                           ▼
              compare_demo_vs_sim.py
              (per-day side-by-side + attribution)
```

### 3.2 Time alignment rules

- **Timeframe:** Both operate on 15m candles. Live decision points = new 15m candle timestamps. Simulation trades = 15m candle timestamps. Align on candle timestamp (UTC).
- **Window definition:** `[start_date 00:00 UTC, end_date 23:45 UTC]` inclusive of 15m candles.
- **Warmup requirement (simulation side):** The simulation's adaptive thresholding needs `adaptive_history_candles` (200 for 15m) of predictions **before** the window, plus the strategic model needs ~250+ 1h candles of history. Therefore:
  - Fetch `days = window_days + 25` days of raw data (e.g., 8 + 25 = 33 days for a safe margin).
  - Predictions generated over the full fetch period.
  - `df_hist` = predictions **strictly before** window start.
  - `df` = predictions **within** window.
  - This mirrors how the live bot had accumulated prediction history before the 8-day observation period.

### 3.3 What to compare

| Metric | Demo (from logs) | Simulation (from sim) |
|---|---|---|
| Trade count (entries) | `🟢 OPEN` lines | `trade_markers` entries |
| Direction split | LONG/SHORT | LONG/SHORT |
| Exit reasons | FULL CLOSE reason, TP/SL inferred | exit_reason in trades |
| Per-trade PnL | entry/exit prices from logs | `return` field |
| Win rate | computed | computed |
| Gate suppression counts | `ℹ️ Gate counter summary` | not applicable (unless gates added) |
| Regime distribution | `📊 Regime distribution` | df `regime` column |

### 3.4 The "apples-to-apples" question

The comparison is only meaningful if the simulation can optionally **mirror the live gates**. The plan therefore parameterizes `simulate_trades_core` so the harness can run it in two modes:

- **Mode "sim-raw"**: current behavior (no gates) — shows maximum possible trade count.
- **Mode "sim-live-faithful"**: with volume filter, HTF EMA50 filter, chop=0.0, RiskGuard — reproduces live behavior on historical data.

The gap between the two modes *is* the explanation for the trade-count difference.

---

## 4. Implementation Tasks

### Task 1 — Parameterize `simulate_trades_core` for live fidelity

**File:** `mltrainingcore.py`

Change signature to accept optional gate controls with defaults preserving current behavior:

```python
def simulate_trades_core(
    df, df_hist, signal_col, tf_cfg, param_list,
    close_col="close",
    regime_stake_mult=None,          # default {'trend': 1.0, 'high_vol': 0.5, 'chop': 0.3}
    volume_filter_threshold=None,    # None = off; 0.8 = live-like
    htf_ema_span=None,               # None = off; 50 = live-like (needs 1h series)
    max_daily_loss_frac=None,        # None = off; 0.05 = live-like RiskGuard
    max_drawdown_frac=None,          # None = off; 0.15
):
```

- `regime_stake_mult`: replace the hardcoded dict at line 193.
- `volume_filter_threshold`: before entry, check `row['volume'] < sma20(row['volume']) * threshold` → skip entry (mirrors dualmlstrategy.py:251-262). `df` already contains `volume`.
- `htf_ema_span`: requires a pre-computed 1h EMA column; the harness passes a merged `df` that already carries the 1h EMA50 value per 15m candle (see Task 3). If absent and filter enabled, fall back to "pass".
- `max_daily_loss_frac` / `max_drawdown_frac`: minimal in-loop equity tracking (peak equity, start-of-day equity) mirroring `riskguard.py`. Only needs to *skip entries* when halted — full RiskGuard parity is out of scope.

**Acceptance:** Existing tests in `tests/test_mltrainingcore.py` still pass with defaults. New tests for each optional gate.

### Task 2 — Demo log parser

**New file:** `demo_log_parser.py` (root, alongside dualmlsimulation.py)

```python
def parse_demo_logs(
    log_path: str,
    start_date: Optional[str] = None,   # "2026-07-19" UTC
    end_date: Optional[str] = None,
) -> DemoLogResult
```

`DemoLogResult` holds:
- `trades: List[dict]` — one per entry/exit pair
- `daily_gates: dict[date -> dict[str,int]]`
- `regime_distribution: dict[date -> dict[str,float]]`
- `heartbeat: List[dict]` — raw `Tactical |` / `Strategic |` signal lines (for signal-level comparison)

**Log line formats to parse** (from `positionmanager.py` / `dualmlstrategy.py`):

| Pattern | Regex | Meaning |
|---|---|---|
| `🟢 OPEN LONG @ 64200.0 qty=0.012` | `OPEN (LONG\|SHORT) @ ([0-9.]+) qty=([0-9.]+)` | Entry |
| `📈 SCALE UP LONG +0.006 (total=0.0180, scale#1)` | `SCALE UP (LONG\|SHORT) \+([0-9.]+) \(total=([0-9.]+)` | Scale-up |
| `🔽 PARTIAL CLOSE -0.004 (remaining=0.0080)` | `PARTIAL CLOSE -([0-9.]+) \(remaining=([0-9.]+)\)` | Partial exit |
| `🔵 FULL CLOSE (start) reason=MAX_HOLD_TIME — side=LONG amount=0.0120 entry=64200.0` | `FULL CLOSE \(start\) reason=(\S+) .*side=(LONG\|SHORT) amount=([0-9.]+)` | Full close metadata |
| `🔵 FULL CLOSE reason=MAX_HOLD_TIME — position closed` | `FULL CLOSE reason=(\S+) .*position closed` | Full close completed |
| `ℹ️ Gate counter summary \| day=2026-07-19 vol_flt=5 htf_trd=12 adapt_thr=34 riskguard=0 chop=8 veto=2 \| pred(...)` | `Gate counter summary \| day=(\S+) (.*)` | Gate counters |
| `📊 Regime distribution (2026-07-19): chop=45% \| high_vol=12% \| trend=43%` | `Regime distribution \((\S+)\): (.*)` | Regime % |

**Trade reconstruction logic:**
- On `OPEN`: start a position record (side, entry_price, qty, ts).
- On `SCALE UP`: add to qty, keep weighted avg entry price.
- On `PARTIAL CLOSE`: subtract qty, record partial PnL.
- On `FULL CLOSE (start)`: capture reason + amount.
- On `FULL CLOSE reason=... position closed`: close the position record, compute PnL from `entry_price` vs current price **if available**. If the exchange-side TP/SL filled instead (position went flat without a FULL CLOSE line — detected via `Position | FLAT` lines), estimate exit price as `entry * (1 + tp_frac)` or `entry * (1 - sl_frac)` using the strategic SL/TP fracs logged at entry (`🛡 TP=... SL=...`).
- **Timestamp handling:** broker log lines include timestamps (`2026-07-19 14:00:03,123`). Parse with `datetime.strptime(..., "%Y-%m-%d %H:%M:%S")` and treat as UTC.

**Outputs:**
- `demo_trades.csv` — schema below
- `demo_daily_summary.csv` — per day: entries, exits, win rate, pnl, gate counts

**Schema `demo_trades.csv`** (mirrors sim trades schema for easy diff):

```csv
entry_ts,exit_ts,side,entry_price,exit_price,qty,exit_reason,pnl_raw,regime
2026-07-19 08:00:00,2026-07-19 11:45:00,LONG,64200.0,64700.0,0.012,MAX_HOLD_TIME,0.0078,trend
```

**Schema `demo_daily_summary.csv`**:

```csv
date,entries,exits,win_rate,pnl_total,vol_flt,htf_trd,adapt_thr,riskguard,chop,veto,regime_trend_pct,regime_chop_pct,regime_highvol_pct
2026-07-19,2,2,0.5,0.012,5,12,34,0,8,2,43,45,12
```

**Acceptance:** Unit tests with a small fixture log file (10–20 lines covering each pattern).

### Task 3 — Windowed simulation mode

**File:** `dualmlsimulation.py`

Add:

```python
def run_windowed_simulation(
    symbol: str,
    days: int,                    # total fetch days (window + warmup margin)
    timeframe: str,
    start_date: str,              # "2026-07-19" UTC
    end_date: str,                # "2026-07-26" UTC
    model_dir=MODEL_DIR,
    live_faithful: bool = False,  # mirrors live gates (Task 1 params)
) -> tuple
```

Logic:
1. `df_predictions, df_raw = run_predictions_only(symbol, days, timeframe)` — reuse existing cache logic.
2. Compute `window_start = pd.Timestamp(start_date)`, `window_end = pd.Timestamp(end_date) + 1 day`.
3. `df_hist = df_predictions[df_predictions.index < window_start]` — take last `adaptive_history_candles`.
4. `df_test = df_predictions[(df_predictions.index >= window_start) & (df_predictions.index < window_end)]`.
5. `param_list = _build_strategic_param_list(df_test, df_raw, strategic, strategic_tf_cfg)` — note this already resamples `df_raw` to 1h for the strategic model; since `df_raw` spans the full fetch period, 1h history before the window is available.
6. If `live_faithful`: build the HTF EMA50 column on the 1h series, merge onto `df_test` (carry-forward per 15m candle), then call `simulate_trades_core` with the Task-1 gate parameters (`regime_stake_mult={'chop': 0.0}`, `volume_filter_threshold=0.5/0.8`, `htf_ema_span=50`, RiskGuard limits).
7. Else: call `simulate_trades_core` with defaults.
8. Return `(df_result, metrics, df_test)`.

**CLI:** extend `__main__` argparse with `--start-date`, `--end-date`, `--live-faithful`.

**Outputs:** `sim_trades_{start}_{end}.csv` + `sim_daily_summary_{start}_{end}.csv` using the *same schemas* as Task 2.

**Acceptance:** 
- `--start-date`/`--end-date` restrict trade markers strictly to the window.
- No trades generated in the warmup (pre-window) region.
- Offline test with synthetic df verifying window boundary behavior.

### Task 4 — Comparison report

**New file:** `compare_demo_vs_sim.py`

```python
def compare(
    demo_trades_csv: str,
    sim_trades_csv: str,
    demo_daily_csv: str,
    sim_daily_csv: str,
) -> None
```

Report (console + `comparison_report.md`):

1. **Per-day side-by-side table:**
   ```
   Date        | Demo entries | Sim entries | Delta | Demo win% | Sim win% | Demo PnL | Sim PnL
   2026-07-19  | 2            | 5           | -3    | 50%       | 60%      | +0.012   | +0.008
   ```

2. **Gate attribution block:** for each date, show which live gates suppressed signals (from `demo_daily_summary.csv`), and the count of sim trades that would have been blocked by each gate if live gates were applied (estimated in mode "sim-live-faithful" runs both sub-modes internally and reports the delta).

3. **Regime distribution comparison:** demo vs sim regime % per day — highlights D1/D5 contribution.

4. **Summary:** totals, win rates, PnL, and a ranked list of top contributors to the trade-count gap (gate counts × typical blocked rate).

### Task 5 — Unit tests

**New files:** `tests/test_demo_log_parser.py`, `tests/test_windowed_simulation.py`

Cover:
- Parser: each log pattern, trade reconstruction, partial/scale handling, TP/SL exit inference.
- Windowed sim: boundary filtering (no trades before start / after end), warmup-history seeding, `live_faithful` mode activates gates (mock `simulate_trades_core` to assert parameters passed).
- `simulate_trades_core` new params: `regime_stake_mult`, `volume_filter_threshold`, `htf_ema_span`, RiskGuard skip — with synthetic 200-row DataFrames, no network.

### Task 6 — Documentation

**File:** `plans/` — update or add `demo_vs_sim_comparison.md` (this plan) with a short "How to run" section after implementation:

```bash
# 1. Extract live trades from logs
python demo_log_parser.py --log logs/bot.log --start-date 2026-07-19 --end-date 2026-07-26

# 2. Run aligned simulation (33 days fetch, simulate only the 8-day window)
python dualmlsimulation.py --days 33 --start-date 2026-07-19 --end-date 2026-07-26 --live-faithful

# 3. Compare
python compare_demo_vs_sim.py --demo trades/demo_trades.csv --sim trades/sim_trades.csv ...
```

---

### Task 7 — Built-in rotating log file (10-day retention, max-size cap)

**Motivation:** tmux `capture-pane`/`pipe-pane` output includes ANSI escape codes and terminal-width line wrapping, making it unreliable for the demo-log parser. The bot must therefore write **clean machine-readable logs itself** as the primary source.

**File:** `binancebasebroker.py` — replace the `logging.basicConfig` in `setup_logging()` (lines 247-252) with a **daily-rotating file handler** plus a console handler.

```python
def setup_logging(self):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    # daily files: logs/trading_2026-07-19.log
    file_handler = TimedRotatingFileHandler(
        log_dir / "trading.log",
        when="midnight",
        backupCount=10,          # keep 10 days
        utc=True,
        encoding="utf-8",
    )
    # hard cap: if the current day's file exceeds MAX_LOG_BYTES (e.g. 5 MB),
    # the strategy loop truncates it (see below) — TimedRotatingFileHandler
    # alone has no byte cap, so the strategy enforces it.
    ...
```

**Retention rules (per user requirement: "max size limit to keep specific trading days log only"):**
- **Daily files** `logs/trading_YYYY-MM-DD.log` via `TimedRotatingFileHandler(when="midnight", backupCount=10, utc=True)` → keeps exactly the last 10 trading days, older files auto-deleted.
- **Max size cap:** each day's file capped at `MAX_LOG_BYTES = 5 * 1024 * 1024` (5 MB ≈ ~40k lines ≈ ~17 days of baseline logging at 2,300 lines/day — comfortably inside one day). Enforced in `BaseStrategy.run()` loop: after each iteration, if `(log_dir / f"trading_{today}.log").stat().st_size > MAX_LOG_BYTES`, truncate to the **last 10k lines** (rotate tail) and log a warning. This guarantees no single day's log grows unbounded (e.g. during error-backoff storms).
- **Timestamps:** the `logging.Formatter` already emits `%(asctime)s` (UTC if the machine is UTC — verify; the bot's candle logic already uses `datetime.utcnow()`, so the host should run UTC).

**Console output preserved:** keep a `StreamHandler` alongside the file handler so the tmux pane still shows live output (and the tmux scrollbuffer remains a secondary/human fallback).

**Config:** expose `LOG_DIR` and `MAX_LOG_BYTES` as module constants in `binancebasebroker.py` (or `mlio.py`), overridable via env vars `TBOT_LOG_DIR` / `TBOT_MAX_LOG_BYTES`.

**Acceptance:**
- Bot run for ≥2 simulated days (or with `sleeptime` temporarily reduced) produces `logs/trading_*.log` files split at UTC midnight.
- Files from 11+ days ago are deleted (backupCount rotation).
- A stress test writing >5 MB to one day's file triggers truncation to the last 10k lines.
- `demo_log_parser.py` (Task 2) parses the produced files with **no ANSI escapes** — line format `2026-07-19 14:00:03,123 - INFO - 🟢 OPEN LONG @ ...` parses directly.

**Update to Task 2:** the parser's primary input becomes `logs/trading_*.log` (glob across the 8-day window); tmux scrollbuffer export is demoted to a documented fallback for cases where built-in logging wasn't enabled retroactively.

---

## 5. Data-format alignment notes

- **Timestamps:** all comparisons in UTC. Demo log lines carry local broker timestamps → parse and convert to UTC (bot already uses `datetime.utcnow()` for `entry_time`; the log prefix from the broker logger uses local time — verify and normalize).
- **PnL units:** demo PnL computed as `(exit/entry - 1)` * direction (raw). Sim `return` field already is `perf_raw * stake * leverage`. To compare apples-to-apples, report **both** `pnl_raw` and `pnl_equity` columns in both pipelines; comparison defaults to `pnl_raw`.
- **Trade pairing:** a sim "trade" = one entry→exit. A demo trade may include scale-ups (combined into one position record) and partial closes (fragment PnL). Pairing rule: one demo entry event == one sim entry marker at the same candle timestamp; scale-ups/partials are recorded as sub-events in a separate CSV (`demo_position_events.csv`) for fidelity.

---

## 6. Risks & limitations

| Risk | Mitigation |
|---|---|
| Log timestamps not in UTC | Normalize in parser; verify against `Tactical \|` candle timestamps |
| TP/SL exchange-side exits not logged as close events | Infer via `Position \| FLAT` transitions + `🛡 TP/SL` values at entry |
| Model-trained regime (live) ≠ rule-based regime (sim) in chop gate | D5 documented; `live_faithful` mode uses rule-based regime column to approximate |
| Recent `chop: 0.3` sim change inflates sim trades vs live block | `live_faithful` mode forces `chop: 0.0`; report both raw and faithful numbers |
| Demo logs may not contain 8 full days (startup warmup gap) | Parser skips dates before first `Tactical \|` line with a warning; report effective window |
| tmux scrollbuffer export (fallback source) contains ANSI escapes + wrapped lines | Task 7 built-in rotating log becomes the primary source; tmux export only used when built-in logging wasn't enabled during the window |
| Strategic model absent (`is_ready=False`) | `_build_strategic_param_list` already falls back to `DEFAULT_PARAMS`; comparison notes this |

---

## 7. Estimated effort

| Task | Files | Effort |
|---|---|---|
| 1. Parameterize simulate_trades_core | `mltrainingcore.py` | Medium |
| 2. Demo log parser | `demo_log_parser.py` (new) | Medium |
| 3. Windowed simulation | `dualmlsimulation.py` | Medium |
| 4. Comparison report | `compare_demo_vs_sim.py` (new) | Medium |
| 5. Unit tests | `tests/` | Medium |
| 6. Documentation | `plans/` | Small |
| 7. Built-in rotating log file | `binancebasebroker.py` | Small |

**Dependency order:** Task 1 → (Task 2 ∥ Task 3) → Task 4 → Task 5 → Task 6.

Tasks 2 and 3 are independent and can be built in parallel once Task 1 lands.

**Task 7 ordering note:** Task 7 (rotating log file) is independent of Tasks 1-6 and can land at any time. It becomes the **primary input** for Task 2's parser, so ideally Task 7 ships *before* the next demo-trading run begins (so the 10-day window is captured in clean files). If the demo bot is already running with `logging.basicConfig` only, the tmux scrollbuffer export in the README is the fallback until Task 7 is deployed and the next 10 days accrue.

---

## 8. Implementation status & how to run

**Status:** Tasks 1–7 implemented. Task 1 gates are **off by default** in
`simulate_trades_core` (defaults preserve prior behavior); Task 7 rotating log
ships in `binancebasebroker.py` (see README "Cyclic logging").

| Task | File | Status |
|---|---|---|
| 1. Parameterize `simulate_trades_core` | `mltrainingcore.py` | Done — `regime_stake_mult`, `volume_filter_threshold`, `htf_ema_span`, `max_daily_loss_frac`, `max_drawdown_frac` |
| 2. Demo log parser | `demo_log_parser.py` | Done — `trades/demo_trades.csv` + `demo_daily_summary.csv` |
| 3. Windowed simulation | `dualmlsimulation.py` | Done — `run_windowed_simulation` + `--start-date/--end-date/--live-faithful` CLI |
| 4. Comparison report | `compare_demo_vs_sim.py` | Done — per-day table, gate attribution, regime diff, ranking |
| 5. Unit tests | `tests/test_demo_log_parser.py`, `tests/test_windowed_simulation.py` | Done — plus `tests/_verify_sim_core.py` gate checks |
| 6. Documentation | `plans/`, `README.md` | Done (this section + README "Recommended workflow") |
| 7. Rotating log file | `binancebasebroker.py` | Done (README "Cyclic logging") |

### Working commands

```bash
# 1. Extract live trades from the accumulated daily logs (UTC window)
python demo_log_parser.py --log logs --start-date 2026-07-19 --end-date 2026-07-26

# 2. Aligned simulation: fetch 35 days, simulate only the 8-day window, live-faithful gates
python dualmlsimulation.py --symbol BTCUSDT --days 35 --timeframe 15m \
    --start-date 2026-07-19 --end-date 2026-07-26 --live-faithful

# 3. Compare demo vs sim (per-day side-by-side + gate attribution + regime diff)
python compare_demo_vs_sim.py \
    --demo trades/demo_trades.csv \
    --sim trades/sim_trades_2026-07-19_2026-07-26.csv \
    --demo-daily trades/demo_daily_summary.csv \
    --sim-daily trades/sim_daily_summary_2026-07-19_2026-07-26.csv \
    --out comparison_report.md
```

Notes on drift from the original plan text:
- `run_windowed_simulation(df_test)` filters to `[start_date, end_date]` **inclusive
  (end + 1 day)**, warmup history = the `adaptive_history_candles` rows strictly
  before the window.
- `--live-faithful` sets `regime_stake_mult={'trend':1.0,'high_vol':0.5,'chop':0.0}`,
  `volume_filter_threshold=0.8`, `htf_ema_span=50`, `max_daily_loss_frac=0.05`,
  `max_drawdown_frac=0.15`.
- Gate parameterization lives entirely in `mltrainingcore.py:simulate_trades_core`;
  the tests in `tests/_verify_sim_core.py` third-party-check each gate preserves
  prior behavior with defaults.
