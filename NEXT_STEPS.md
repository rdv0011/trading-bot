# Next Steps Plan

## Immediate (Post-Trade Analysis)

### 1. Fix Trailing SL Bug ✅ DONE
- **Commit `fd24a78`**: Fixed `atr_pct` undefined error in `_sync_trailing_orders`
- Trailing SL now functional from iteration 1

### 2. Relax Time-Weighted SL Tightening ✅ DONE
- **Commit `221b753`**: Trigger moved from 50% → 75% of `max_hold_hours`
- **Added**: Progressive tightening option (TIME_WEIGHTED_SL_PROGRESSIVE)
- **Config**: `TIME_WEIGHTED_SL_TRIGGER=0.75`, `TIME_WEIGHTED_SL_PROGRESSIVE=True`
- **File**: `strategy.py` `_check_exits()` method, `config.py`
- **Rationale**: Last trade hit tightened SL exactly at 50% mark (3.87h / 7.55h = 51%)

### 3. Add Volatility Filter for Entries ✅ DONE
- **Commit `b1fb2f4`**: Added abs_vol entry gate (GATE_ABS_VOL_ENABLED)
- Blocks entries when `abs_vol_ratio < 0.10` (10% of median volume)
- Added config: `GATE_ABS_VOL_ENABLED`, `ABS_VOL_RATIO_CRITICAL`
- Added `abs_vol_blocked` to decision logging

---

## Medium Term: Initialization Enhancements

### 4. Seed Volatility Regime History ✅ DONE
- **Commit `8ee481d`**: Added `seed_vol_history()` in `strategy.py` using `compute_vol_ratio_from_klines()` on startup klines
- **Config**: `ADAPTIVE_VOL_ENABLED`, `ADAPTIVE_VOL_LOOKBACK=200`, `ADAPTIVE_VOL_QUANTILE=0.95`, `ADAPTIVE_VOL_MIN=1.2`, `ADAPTIVE_VOL_MAX=3.0`
- **Integration**: Extreme vol gate now uses adaptive threshold (95th percentile of recent vol_ratio) when history available
- **Log**: `[VOL INIT] Seeded vol_ratio_history with 50 samples median=0.985`

### 5. Seed Spread/Depth Metrics ✅ DONE
- **Commit `8ee481d`**: Added `seed_spread_depth_history()` in `strategy.py` fetching orderbook snapshots via REST
- **Config**: `ADAPTIVE_BOOK_ENABLED`, `ADAPTIVE_BOOK_LOOKBACK=200`, `ADAPTIVE_SPREAD_QUANTILE=0.75`, `ADAPTIVE_SPREAD_MIN=2.0`, `ADAPTIVE_SPREAD_MAX=20.0`, `ADAPTIVE_DEPTH_QUANTILE=0.25`, `ADAPTIVE_DEPTH_MIN=20000`, `ADAPTIVE_DEPTH_MAX=500000`
- **Integration**: Liquidity entry/exit gates now use adaptive spread/depth thresholds when history available
- **Log**: `[BOOK INIT] Seeded spread_history (20 samples, median=0.02 bps) and depth_history (20 samples, median=15355747 USD)`

### 6. Prediction History Pre-seeding 📋 BLOCKED
- **Status**: Cannot pre-seed — requires model inference
- **Workaround**: Warmup period is acceptable (200 preds = ~3.3h)
- **Alternative**: Cache predictions across restarts (complex)

---

## Longer Term: Risk Management

### 7. Dynamic Position Sizing Based on Volatility ✅ DONE (86897e5)
- ✅ **Done**: Scale stake fraction by `1 / abs_vol` when abs_vol > threshold (DYNAMIC_SIZING_ABS_VOL_THRESHOLD=1.5)
- ✅ **Done**: Min/max stake fraction caps (DYNAMIC_SIZING_MIN_FRAC=0.25, MAX_FRAC=2.0)
- ✅ **Done**: Applied to both initial entry and scale-ins
- Config: DYNAMIC_SIZING_ENABLED, DYNAMIC_SIZING_ABS_VOL_THRESHOLD, MIN_FRAC, MAX_FRAC

### 8. Trailing SL Improvements ✅ DONE (86897e5)
- ✅ **Done**: Trail from entry immediately (TRAILING_START_IMMEDIATE=True)
- ✅ **Done**: Floor at initial SL until 1R, then breakeven
- ✅ **Done**: Minimum trailing distance (TRAILING_MIN_DISTANCE_ATR_MULT=0.5)
- ✅ **Done**: Chandelier exit variant (TRAILING_CHANDELIER_ENABLED, TRAILING_CHANDELIER_MULT=3.0)
- Applied in both _check_exits (live) and _sync_trailing_orders (exchange sync)

### 9. Liquidity Gate Refinements ✅ DONE (86897e5)
- ✅ **Done**: Separate entry vs exit thresholds (LIQ_ENTRY_SPREAD_BPS=15, LIQ_ENTRY_MIN_DEPTH_USD=75000, LIQ_ENTRY_MAX_VANISH_PCT=50, LIQ_ENTRY_MIN_INTENSITY_USD=30000)
- ✅ **Done**: Orderbook imbalance metric (LIQ_IMBALANCE_ENABLED, LIQ_IMBALANCE_THRESHOLD=2.0)
- ✅ **Done**: Funding rate check (LIQ_FUNDING_ENABLED, LIQ_FUNDING_THRESHOLD_BPS=5.0)
- Applied to both entry and exit gates with adaptive thresholds

### 10. Cost-to-Edge Optimization ✅ DONE (86897e5)
- ✅ **Done**: Track realized vs expected costs per trade (COST_TRACKING_ENABLED)
- ✅ **Done**: Adjust confidence stake power based on recent cost_ratio (COST_STAKE_POWER_ADJUST, MIN=0.3, MAX=0.7)
- ✅ **Done**: Time-of-day slippage model (COST_SLIPPAGE_TOD_ENABLED with COST_SLIPPAGE_TOD_WINDOWS)
- Integrated into _confidence_scaled_stake and _exit_position

### 11. Profit Taking / Partial Exits ✅ DONE (221b753)
- ✅ **Done**: R-multiple partial exits at 1R/2R/3R (33%/33%/34%)
- ✅ **Done**: 5% buffer triggers at 0.95R/1.95R/2.95R
- ✅ **Done**: Remaining position protected by trailing SL

---

## Infrastructure

### 11. Monitoring & Alerting
- Add health check endpoint
- Alert on: LIQ_EXIT frequency, error rate, prediction latency
- Daily PnL summary email/Telegram

### 12. Automated Retraining Pipeline
- Schedule weekly retraining (cron or GitLab CI)
- A/B test new models before deployment
- Rollback mechanism if performance degrades

---

## Completed (Reference)

| Date | Item | Commit |
|------|------|--------|
| Nov 17 | MIN_TREES guard (5 trees minimum) | `9f2009d` |
| Nov 17 | Trailing SL fix (breakeven threshold removed) | `1162939` |
| Nov 17 | Testnet simulation for SL/TP | `1162939` |
| Nov 18 | Lower LIQ_EXIT threshold (50k→25k) | `11eca40` |
| Nov 18 | Lower TRAILING_SL_UPDATE_THRESHOLD (0.5%→0.2%) | `11eca40` |
| Nov 18 | Scale-in preserves trailing progress | `11eca40` |
| Nov 18 | Adaptive liquidity threshold | `bde80da` |
| Nov 18 | Adaptive trailing update threshold | `bde80da` |
| Nov 19 | Trade intensity initialization | `ff2acab` |
| Nov 19 | ATR history initialization | `3bc3212` |
| Nov 19 | Fix 7-day klines fetch (672 candles) | `ed2e6d8` |
| Nov 19 | Fix adaptive trailing: pass price parameter | `25b155f` |
| Nov 20 | Fix atr_pct undefined bug | `fd24a78` |
| Nov 20 | Fix missing imports (is_testnet_env, ADAPTIVE_LIQ_COOLDOWN_MIN) | `74f08b0` |
| Nov 20 | Fix SL fraction mutation bug | `b1fb2f4` |
| Nov 20 | Add abs_vol entry gate | `b1fb2f4` |
| Nov 20 | Add error logging for bare except blocks | `b1fb2f4` |
| Nov 20 | Fix f-string format specifier bug in SCALE-IN logging | `8106e28` |
| Nov 23 | **Partial exit buffer** (5% at 0.95R/1.95R/2.95R) | `221b753` |
| Nov 23 | **Scale-in minimum profit** (0.5R required) | `221b753` |
| Nov 23 | **Scale-in stake reduced** (33% from 50%) | `221b753` |
| Nov 23 | **Trailing SL from entry** (immediate, floor at init SL → breakeven at 1R) | `221b753` |
| Nov 23 | **Time-weighted SL** (trigger 75%, progressive tightening) | `221b753` |
| Nov 23 | **Fix trailing SL logic** (correct floor behavior) | `221b753` |
| Nov 23 | **Seed vol_ratio history** (adaptive extreme vol threshold) | `8ee481d` |
| Nov 23 | **Seed spread/depth history** (adaptive liquidity gates) | `8ee481d` |
| Nov 23 | **Dynamic position sizing** (1/abs_vol scaling, cost-adjusted power, TOD slippage) | `86897e5` |
| Nov 23 | **Trailing SL improvements** (min distance, chandelier exit) | `86897e5` |
| Nov 23 | **Liquidity gate refinements** (separate entry/exit, imbalance, funding) | `86897e5` |
| Nov 23 | **Cost-to-edge optimization** (cost tracking, stake power adjust, TOD model) | `86897e5` |
| Nov 23 | **Circuit breaker** (daily loss %, consecutive losses, drawdown, cooldown) | `362b33f` |
| Nov 23 | **Position sync improvements** (existing SL/TP detection, dust validation, trailing sync) | `362b33f` |
| Nov 24 | **Enable Short Signals** (ABSOLUTE_THRESHOLD_SHORT=0.003, separate threshold) | `e678584` |
| Nov 24 | **Reduce Trailing SL Sensitivity** (ATR_MULT=2.0, MIN_DIST=1.0, CHANDELIER=on) | `e678584` |
| Nov 24 | **Relax LIQ_EXIT for Trending Markets** (quantile=0.10, min=3000, grace=20min, trend relax) | `e678584` |
| Nov 24 | **Lower Entry Threshold for Shorts** (THRESHOLD_SHORT=0.003, ADAPTIVE_MIN=0.001) | `e678584` |
| Nov 24 | **Add Trend-Following Filter** (EMA50, block counter-trend signals) | `e678584` |
| Nov 24 | **Retrain tactical model with balanced labels** (180 days, sample weight balancing) | `d414d39` |
| Nov 24 | **Fix walk-forward retraining** to use sample weight balancing, add disable option | `87cdbe5` |
| Nov 26 | **Switch tactical model to 1h timeframe** (retrained on 180 days 1h data, 500 trees) | `3c33a36` |
| Nov 26 | **Widen trailing SL for 1h** (CHANDELIER_MULT=5.0, ATR_MULT=3.0) | `3c33a36` |

---

## Code Review Findings (Nov 20)

### Critical Issues

| # | Issue | Location | Severity | Status |
|---|-------|----------|----------|--------|
| 1 | **Missing import**: `is_testnet_env()` not imported in `strategy.py` | `strategy.py` lines 1319, 1377 | **CRITICAL** | ✅ Fixed in `74f08b0` |
| 2 | **Variable mutation bug**: `sl_frac` modified in-place affects trailing stop | `strategy.py` line 1108 | **HIGH** | ✅ Fixed in `b1fb2f4` |

### High Severity Issues

| # | Issue | Location | Severity | Status |
|---|-------|----------|----------|--------|
| 3 | **SL fraction mutation**: Time-weighted SL tightening modifies `sl_frac` which is then reused for trailing stop calculations, causing incorrect base_sl | `strategy.py` `_check_exits()` lines 1103-1108, 1131 | **HIGH** | ✅ Fixed in `b1fb2f4` (separate `sl_frac_tightened` variable) |
| 4 | **Missing error logging**: Several bare `except Exception:` blocks without logging | `strategy.py` lines 850, 887, 892 | **MEDIUM** | ✅ Fixed in `b1fb2f4` (added log_debug) |

### Medium Severity Issues

| # | Issue | Location | Severity | Status |
|---|-------|----------|----------|--------|
| 5 | **Testnet vs Mainnet logic**: Multiple `if is_testnet_env():` branches with different behavior | `strategy.py`, `broker.py`, `main.py` | **MEDIUM** | ✅ Reviewed - testnet simulates orders due to API limitations |
| 6 | **Position sync on startup**: `sync_exchange_position()` may not handle all edge cases | `strategy.py` | **MEDIUM** | ✅ Fixed in `362b33f` - detects existing SL/TP, validates size, syncs trailing state |
| 7 | **No circuit breaker**: No max daily loss limit or consecutive loss halt | `strategy.py` | **MEDIUM** | ✅ Fixed in `362b33f` - daily loss %, consecutive losses, drawdown, cooldown |

### Low Severity / Code Quality

| # | Issue | Location | Severity | Status |
|---|-------|----------|----------|--------|
| 8 | **Magic numbers**: Hardcoded values (0.5 for SL tightening, 1.2 for ATR mult) | `strategy.py`, `config.py` | **LOW** | 🟢 Configurable - most moved to config.py |
| 9 | **Large functions**: `_check_exits()` >100 lines, `_sync_trailing_orders()` >100 lines | `strategy.py` | **LOW** | 🟢 Functional - refactor optional |
| 10 | **Inconsistent naming**: Mix of `snake_case` and `camelCase` in some variables | Various | **LOW** | 🟢 Style only - no functional impact |

---

## Remaining Items (Optional / Nice-to-Have)

### Critical (Required for Profitability) - ✅ COMPLETE
- **Retrain tactical model with balanced labels**: ✅ Done (180 days, sample weight balancing)
  - Added class weight balancing in training (`TACTICAL_BALANCE_SAMPLES=True`)
  - Trained on 180 days of data (includes bear market periods)
  - Model now produces both positive and negative predictions
  - Fixed walk-forward retraining to preserve balanced training
- **Switch tactical model to 1h timeframe**: ✅ Done (Nov 26)
  - Retrained on 180 days of 1h data (4320 candles)
  - 500 trees (early stopped to 30), balanced labels (49.9%/50.1%)
  - Wider trailing SL for 1h: CHANDELIER_MULT=5.0, ATR_MULT=3.0
  - Bot restarted and running on 1h timeframe
- **Remove Strategic Model (Phase 1)**: ✅ Done (Nov 26)
  - Deleted strategic ML model training/loading/prediction code
  - Replaced with heuristic meta-params in strategy.py (_update_meta_params_heuristic)
  - Updated train_mode, simulate_mode, live_mode to use tactical model only
  - Removed STRATEGIC_TARGET_COLS, STRATEGIC_MODEL_PARAMS, STRATEGIC_TF from config.py
  - Removed strategic features from data.py
  - Updated compare_live_vs_sim.py and all tests
  - Simplifies codebase: single tactical model (1h) + heuristic meta-params

### Phase 2: In-Memory Tactical Model Training (✅ DONE)
- **Implemented** in `live_mode` (main.py): trains the tactical model on 180 days of 1h data via `run_full_pipeline(symbol=SYMBOL, whole_days=HISTORY_DAYS, timeframe=TACTICAL_TF, train_frac=TRAIN_FRACTION)`, stores only in memory (`save=False`), no disk persistence.
- **Disabled periodic walk-forward retraining**: set `WALKFORWARD_RETRAIN_EVERY = 0` in config.py so the live loop no longer retrains on cadence.

### Phase 3: Hybrid 15m Entry Confirmation Filter (✅ DONE)
- **Added** config flags: `CONFIRM_15M_ENABLED`, `CONFIRM_15M_TF="15m"`, `CONFIRM_15M_LIMIT=200`, RSI(14)/volume-zscore/momentum thresholds.
- **Added** helpers in `data.py`: `compute_rsi`, `compute_momentum`, `compute_volume_zscore`.
- **Added** `_phase3_15m_confirm()` in `strategy.py`: fetches recent 15m candles via broker, computes RSI(14), momentum, volume z-score; blocks entry when RSI is oversold/overbought for longs / conflicting for shorts, or momentum/volume z-score below threshold.
- **Hooked** in `_live_iteration` right before `_enter_position`.

### Code Quality Improvements
- **Refactor large functions**: Split `_check_exits()` and `_sync_trailing_orders()` into smaller helpers
- **Naming consistency**: Standardize all variables to `snake_case`
- **Type hints**: Add more comprehensive type annotations

### Infrastructure
- **Monitoring & Alerting**: Health check endpoint, LIQ_EXIT frequency alerts, daily PnL summary
- **Automated Retraining Pipeline**: Weekly retraining schedule, A/B testing, rollback mechanism

### Testing
- **Integration tests**: End-to-end test coverage for all exit paths
- **Load testing**: Verify performance under high-frequency market data