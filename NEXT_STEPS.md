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

### 7. Dynamic Position Sizing Based on Volatility
- Scale stake fraction by `1 / abs_vol` when abs_vol > threshold
- Reduces exposure during elevated volatility

### 8. Trailing SL Improvements ✅ PARTIAL (221b753)
- ✅ **Done**: Trail from entry immediately (TRAILING_START_IMMEDIATE=True)
- ✅ **Done**: Floor at initial SL until 1R, then breakeven
- ⏳ Minimum trailing distance (prevent too-tight trails)
- ⏳ Chandelier exit variant

### 9. Liquidity Gate Refinements
- Separate entry vs exit thresholds
- Add orderbook imbalance metric (bid/ask depth ratio)
- Consider funding rate in liquidity assessment

### 10. Cost-to-Edge Optimization
- Track realized vs expected costs per trade
- Adjust confidence stake power based on recent cost_ratio
- Add slippage model per time-of-day

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
| 5 | **Testnet vs Mainnet logic**: Multiple `if is_testnet_env():` branches with different behavior | `strategy.py`, `broker.py`, `main.py` | **MEDIUM** | 🟡 Reviewed |
| 6 | **Position sync on startup**: `sync_exchange_position()` may not handle all edge cases | `strategy.py` | **MEDIUM** | 🟡 Reviewed |
| 7 | **No circuit breaker**: No max daily loss limit or consecutive loss halt | `strategy.py` | **MEDIUM** | 🟡 Planned |

### Low Severity / Code Quality

| # | Issue | Location | Severity | Status |
|---|-------|----------|----------|--------|
| 8 | **Magic numbers**: Hardcoded values (0.5 for SL tightening, 1.2 for ATR mult) | `strategy.py`, `config.py` | **LOW** | 🟢 Configurable |
| 9 | **Large functions**: `_check_exits()` >100 lines, `_sync_trailing_orders()` >100 lines | `strategy.py` | **LOW** | 🟢 Refactor later |
| 10 | **Inconsistent naming**: Mix of `snake_case` and `camelCase` in some variables | Various | **LOW** | 🟢 Style only |