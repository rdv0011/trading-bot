# Next Steps Plan

## Immediate (Post-Trade Analysis)

### 1. Fix Trailing SL Bug ✅ DONE
- **Commit `fd24a78`**: Fixed `atr_pct` undefined error in `_sync_trailing_orders`
- Trailing SL now functional from iteration 1

### 2. Relax Time-Weighted SL Tightening 🔄 PENDING
- **Current**: SL tightens to 50% at 50% of `max_hold_hours`
- **Proposed**: Move trigger from 50% → 75% of `max_hold_hours`
- **File**: `strategy.py` `_check_exits()` method
- **Rationale**: Last trade hit tightened SL exactly at 50% mark (3.87h / 7.55h = 51%)

### 3. Add Volatility Filter for Entries ✅ DONE
- **Commit `b1fb2f4`**: Added abs_vol entry gate (GATE_ABS_VOL_ENABLED)
- Blocks entries when `abs_vol_ratio < 0.10` (10% of median volume)
- Added config: `GATE_ABS_VOL_ENABLED`, `ABS_VOL_RATIO_CRITICAL`
- Added `abs_vol_blocked` to decision logging

---

## Medium Term: Initialization Enhancements

### 4. Seed Volatility Regime History 📋 PLANNED
- **Function exists**: `compute_vol_ratio_from_klines()` in `data.py`
- **Use for**: `vol_ratio` (extreme vol gate) adaptive threshold
- **Source**: 7-day klines already fetched at startup
- **Integration**: Add `seed_vol_history()` method in `strategy.py`

### 5. Seed Spread/Depth Metrics 📋 PLANNED
- **Source**: REST depth endpoint (`https://fapi.binance.com/fapi/v1/depth`)
- **Function exists**: `_fetch_snapshot_rest()` in `orderbook.py`
- **Use for**: Adaptive spread/depth thresholds in liquidity gates
- **Integration**: Add `estimate_initial_spread_depth()` in `orderbook.py`

### 6. Prediction History Pre-seeding 📋 BLOCKED
- **Status**: Cannot pre-seed — requires model inference
- **Workaround**: Warmup period is acceptable (200 preds = ~3.3h)
- **Alternative**: Cache predictions across restarts (complex)

---

## Longer Term: Risk Management

### 7. Dynamic Position Sizing Based on Volatility
- Scale stake fraction by `1 / abs_vol` when abs_vol > threshold
- Reduces exposure during elevated volatility

### 8. Trailing SL Improvements
- Add minimum trailing distance (prevent too-tight trails)
- Add "profit lock" at 1R/2R/3R milestones
- Consider chandelier exit variant

### 9. Liquidity Gate Refinements
- Separate entry vs exit thresholds
- Add orderbook imbalance metric (bid/ask depth ratio)
- Consider funding rate in liquidity assessment

### 10. Cost-to-Edge Optimization
- Track realized vs expected costs per trade
- Adjust confidence stake power based on recent cost_ratio
- Add slippage model per time-of-day

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