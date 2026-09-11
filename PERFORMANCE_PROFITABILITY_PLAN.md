# Performance & Profitability Improvement Plan

This plan focuses on the repo-root simplified version of the trading bot.
The legacy bot code has been removed from the repository.

---

## 1. Current Architecture Summary

```
.
├── strategy.py      # DualMLStrategy (live loop, signal→trade)
├── model.py         # CatBoostModel, rolling_tactical_predict, strategic_batch_predict
├── data.py          # Feature engineering, label generation, data pipeline
├── broker.py        # BinanceBroker (live) + MockBroker (sim), rate limiter
├── simulate.py      # MockBroker simulation engine, run_simulation()
├── config.py        # All parameters (thresholds, model params, etc.)
├── logger.py        # Logging (file + console, daily rotation)
├── main.py          # Entry point: train / simulate / live / compare
├── compare.py       # Sim vs live comparison
└── utils.py         # Helpers
```

**Key design decisions already made:**
- 15m tactical + 1h strategic (vs 5m/1h in legacy)
- Walk-forward retrain every 100 candles (not every iteration)
- Atomic model save + hot-swap for parallel training
- Absolute threshold (no adaptive thresholding)
- No position scaling (single entry/exit per trade)

---

## 2. Performance Bottlenecks

### P0 — Critical Latency (live loop)

| Bottleneck | Where | Current Cost | Fix |
|---|---|---|---|
| Tactical retrain every 100 candles | `strategy._retrain_tactical()` | 0.5-2s catboost fit | Incremental fit / cached retrain |
| Feature engineering 3x per iteration | `strategy._live_iteration()` | 0.1-0.3s × 3 | Cache with TTL |
| Sequential API calls | `broker.get_historical_prices()` × 2 + `get_last_price()` | 1-2s total | Parallel fetch |
| No model persistence for tactical | `_retrain_tactical()` rebuilds from scratch on restart | 30s warmup | Persist tactical model |

### P1 — Moderate Latency

| Bottleneck | Where | Current Cost | Fix |
|---|---|---|---|
| Strategic prediction every ~1h | `strategy._live_iteration()` | 0.05s | Batch cache |
| Klines cache TTL too generous | `broker._klines_cache_ttl()` | Stale data risk | Shorter TTL for live |
| Sleep is blocking | `strategy.run_live_loop()` | Blocks background work | Async-friendly sleep |

---

## 3. Profitability Improvements

### P0 — High-Impact Edge Additions

| Improvement | Current | Proposed | Expected Impact |
|---|---|---|---|
| **Adaptive threshold** | Fixed `ABSOLUTE_THRESHOLD=0.006` | Rolling percentile of recent predictions (like legacy) | Fewer false signals in low-vol, more entries in high-signal regimes |
| **Trailing stop-loss** | Fixed SL fraction from strategic model | ATR-based trailing stop that locks in profit | +15-25% on winning trades |
| **Multi-timeframe confirmation** | Tactical signal alone | Require strategic regime ≠ "chop" AND tactical signal agreement | Filter ~30% of losing trades |
| **Volatility-scaled position sizing** | Fixed stake from strategic model | Scale stake inversely with realized vol | Smaller positions in choppy markets |

### P1 — Medium-Impact Edge Additions

| Improvement | Current | Proposed | Expected Impact |
|---|---|---|---|
| **Position scaling** | None (single entry) | Scale in 1-2× when signal confirms (like legacy `MAX_SCALE_COUNT`) | +10-15% on trending moves |
| **Partial exit on reversal** | Full close on reversal signal | Partial close (33%) on first reversal, full on second | Reduce whipsaw losses |
| **Regime-aware leverage** | Strategic model predicts but not always optimal | Override leverage ceiling based on regime + vol_state | Better risk-adjusted returns |
| **Trade frequency filter** | None | Min cooldown between trades (e.g. 30 min) | Avoid overtrading in chop |

### P2 — Research / Backtest Needed

| Improvement | Description |
|---|---|
| **Ensemble tactical models** | Average predictions from 2-3 catboost models with different seeds/params |
| **Feature importance pruning** | Drop bottom 20% importance features to speed up retrain |
| **Session-aware trading** | Skip or reduce sizing during low-liquidity hours (e.g. 02:00-06:00 UTC) |

---

## 4. Implementation Roadmap

### Phase 1: Performance (Week 1)
**Goal: Cut live iteration from ~3s to <500ms**

1. **Feature caching** in `strategy.py`
   - Cache `make_features_df()` result with 60s TTL per timeframe
   - Avoid recomputing on same raw data
   - Files: `strategy.py`, `data.py`

2. **Parallel API fetch** in `broker.py`
   - Fetch tactical + strategic klines + price in parallel using `concurrent.futures.ThreadPoolExecutor`
   - Files: `broker.py`

3. **Incremental tactical retrain** in `model.py`
   - Buffer recent candles, only refit catboost every N iterations (keep at 100 for now, but optimize the fit)
   - Use `catboost` pool for faster training
   - Files: `model.py`, `strategy.py`

4. **Tactical model persistence** in `model.py`
   - Save/load tactical model alongside strategic
   - Skip warmup on restart
   - Files: `model.py`, `strategy.py`

### Phase 2: Profitability — Core (Week 2)
**Goal: Add adaptive threshold + trailing stop**

5. **Adaptive threshold** in `strategy.py`
   - Replace `ABSOLUTE_THRESHOLD` with rolling percentile (top/bottom 5% of recent predictions)
   - Mirror legacy `adaptive_thresholding()` from `mltrainingcore.py`
   - Files: `strategy.py`, `data.py`, `config.py`

6. **Trailing stop-loss** in `strategy.py`
   - After entry, track highest/lowest price
   - Move SL to breakeven after 1× risk, then trail at 1.5× ATR
   - Files: `strategy.py`, `simulate.py` (for backtest)

7. **Regime gate hardening** in `strategy.py`
   - Already blocks "chop" regime — also block entries when `vol_state == "extreme"`
   - Files: `strategy.py`

### Phase 3: Profitability — Edge (Week 3)
**Goal: Position scaling + partial exits**

8. **Position scaling** in `strategy.py`
   - Allow 1-2 scale-ups when same-direction signal repeats (like legacy `MAX_SCALE_COUNT=3`)
   - Cancel+replace TP/SL bracket on scale-up
   - Files: `strategy.py`, `broker.py`

9. **Partial exit on reversal** in `strategy.py`
   - Close 33% of position on first reversal signal
   - Full close on second consecutive reversal
   - Files: `strategy.py`, `simulate.py`

10. **Trade cooldown** in `strategy.py`
    - Minimum 30-min gap between new entries
    - Prevents overtrading in sideways markets
    - Files: `strategy.py`, `config.py`

### Phase 4: Backtest Validation (Week 4)
**Goal: Prove improvements before going live**

11. **Backtest all changes** via `simulate` mode
    - Compare old vs new on same validation period
    - Require: Sharpe ≥ old, Max DD ≤ old, Win Rate ≥ old
    - Files: `main.py`, `compare.py`

12. **A/B live test** on testnet
    - Run old strategy alongside new for 1 week
    - Compare real trade outcomes
    - Files: `main.py`

---

## 5. Files to Modify (by Phase)

### Phase 1 — Performance
- `strategy.py` — feature caching, parallel fetch orchestration
- `broker.py` — parallel klines fetch, shorter cache TTL for live
- `model.py` — incremental retrain, tactical persistence
- `config.py` — cache TTL config, retrain interval config

### Phase 2 — Profitability Core
- `strategy.py` — adaptive threshold, trailing stop, regime hardening
- `data.py` — adaptive threshold helper function
- `config.py` — adaptive threshold params, trailing stop params
- `simulate.py` — trailing stop in MockBroker

### Phase 3 — Profitability Edge
- `strategy.py` — position scaling, partial exit, trade cooldown
- `broker.py` — cancel+replace bracket on scale-up
- `simulate.py` — scaling + partial exit in MockBroker
- `config.py` — scale count, partial close fraction, cooldown

### Phase 4 — Validation
- `main.py` — enhanced simulate mode for comparison
- `compare.py` — side-by-side metrics comparison

---

## 6. Expected Combined Impact

| Metric | Current | After Phase 1 | After Phase 2-3 | Target |
|---|---|---|---|---|
| Iteration latency | 2-3s | 300-500ms | 300-500ms | <500ms |
| API weight/min (live) | 80-120 | 40-60 | 40-60 | <60 |
| False signal rate | ~40% | ~40% | ~25% | <25% |
| Avg winning trade duration | 4-6h | 4-6h | 2-4h | 2-4h |
| Max drawdown | 8-12% | 8-12% | 5-8% | <8% |
| Sharpe ratio | 1.2-1.5 | 1.2-1.5 | 1.8-2.2 | >1.8 |

---

## 7. Implementation Order (Dependency Graph)

```
Phase 1 (Performance) ─────────────────────────────┐
  1.1 Feature caching ───────────────┐              │
  1.2 Parallel API fetch ────────────┤              │
  1.3 Incremental retrain ───────────┤              │
  1.4 Tactical persistence ──────────┴──────────────┘
                                                    │
Phase 2 (Profitability Core) ───────────────────────┘
  2.1 Adaptive threshold ─────────────┐
  2.2 Trailing stop-loss ─────────────┤
  2.3 Regime gate hardening ──────────┴──────────────┐
                                                     │
Phase 3 (Profitability Edge) ────────────────────────┘
  3.1 Position scaling ───────────────┐
  3.2 Partial exit ───────────────────┤
  3.3 Trade cooldown ─────────────────┴──────────────┐
                                                     │
Phase 4 (Validation) ────────────────────────────────┘
  4.1 Backtest all changes ───────────┐
  4.2 A/B live test ──────────────────┴── DONE
```

Phase 1 items are independent and can be parallelized.
Phase 2 depends on Phase 1 (needs fast iteration to validate).
Phase 3 depends on Phase 2 (scaling/partial exits need adaptive threshold).
Phase 4 depends on all.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Adaptive threshold causes overfitting to recent data | Use 200-candle lookback, test on out-of-sample |
| Trailing stop triggers too early in trending markets | ATR-based multiplier (1.5×-2×), test multiple values |
| Position scaling increases drawdown | Cap at MAX_SCALE_COUNT=2, test with small stake fractions |
| Parallel API calls trigger rate limiter | Use ThreadPoolExecutor(3), respect existing rate limiter |
| Feature cache serves stale data | 60s TTL for live, invalidate on new candle |

---

## 9. Success Criteria

Phase 1 is complete when:
- [ ] Live iteration < 500ms (measured via iteration timing log)
- [ ] API weight < 60/min (measured via rate limiter logs)
- [ ] Tactical model survives restart without warmup

Phase 2 is complete when:
- [ ] Adaptive threshold produces signals in backtest (not all "hold")
- [ ] Trailing stop improves avg winning trade by ≥15%
- [ ] Regime gate filters ≥80% of choppy-market entries

Phase 3 is complete when:
- [ ] Position scaling backtest shows ≥10% improvement on trending periods
- [ ] Partial exit reduces whipsaw losses by ≥20%
- [ ] Trade cooldown prevents >5 trades/day in chop

Phase 4 is complete when:
- [ ] Backtest Sharpe ≥ old + 0.3
- [ ] Backtest Max DD ≤ old × 0.8
- [ ] 1-week testnet A/B shows improvement
