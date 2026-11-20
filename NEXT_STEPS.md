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

### 3. Add Volatility Filter for Entries 🔄 PENDING
- **Current**: No abs_vol filter on entries
- **Proposed**: Block entries when `abs_vol > 0.8` (elevated volatility)
- **File**: `strategy.py` `_check_entries()` or `_liquidity_entry_gate()`
- **Rationale**: Last trade entered with `abs_vol=0.84` in declining market

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