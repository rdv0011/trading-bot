# Trade Frequency & Profitability Analysis

**Date:** 2026-07-27
**Bot:** Dual-ML Trading System (primary: `DualMLStrategy`, fallback: `MLStrategy`)
**Focus:** Why the bot makes too few trades, and how to increase trade count and/or per-trade profitability.

---

## Executive Summary

The bot uses a cascade of filters, each of which can independently block a trade. Analysis reveals **10 distinct gates** that suppress entry signals. The net effect is that the bot enters trades only under a narrow confluence of conditions: non-chop regime, extreme prediction values (top/bottom ~6% of recent distribution), adequate volume, favorable higher-timeframe trend, and StrategicML approval. This conservative design produces high-certainty entries at the cost of low trade frequency.

**Primary bottleneck:** Adaptive thresholding regime filter. Secondary: HTF trend filter and StrategicML volatility veto.

---

## 1. Complete Signal Path (DualMLStrategy)

```
15m OHLCV 
  → TacticalML (CatBoost): predict future return
    → Adaptive thresholding: signal = LONG/SHORT/HOLD
      → Volume filter: blocks if vol < 80% SMA20
        → HTF trend filter: blocks if LONG below EMA50 / SHORT above EMA50
          → StrategicML (CatBoost): predicts meta-params, allow_trading, regim
            → Regime check: blocks in "chop"
              → RiskGuard: blocks if daily loss ≥5% or drawdown ≥15%
                → PositionManager (entry): requires CONSECUTIVE_SIGNALS_REQUIRED=2 for scale-up
                  → Open bracket order on exchange
```

Each arrow is a potential trade suppression point.

---

## 2. Gate-by-Gate Analysis of Trade Suppression

### Gate 1: Adaptive Thresholding (TacticalML.tacticalml.py:68-81)

**Mechanism:** `fit_and_predict()` maintains a rolling deque (`_pred_history`) of recent predictions. On each new 15m candle, it computes `adaptive_max` (mean of top ~6% of recent predictions) and `adaptive_min` (mean of bottom ~6%). A signal is generated only when the new prediction exceeds `adaptive_max` (LONG) or falls below `adaptive_min` (SHORT).

```python
# timeframe_config.py — 15m preset
adaptive_history_candles = int(50h * 4 candles/h) = 200 candles
label_window_candles = int(16.7h * 4) = 66.8 ≈ 66 candles
# adaptive_thresholding sorts 200 most recent predictions,
# then averages the top (200/66) ≈ 3 predictions → MAX
# and bottom 3 predictions → MIN
```

**Impact:** Only ~3 out of every 200 predictions (1.5%) will naturally exceed these extreme thresholds. This is the #1 reason for too few entries.

**Suggestion:** Increase `label_window_candles` (use fewer, more extreme values) directionally to **widen** the gap (fewer entries but higher conviction), OR decrease it to tighten thresholds and generate **more entries**. Currently at 16.7h → defines top/bottom 3 values. Reducing to e.g. 8h would use top/bottom ~6 values → narrower thresholds → more signals.

### Gate 2: Regime Detection → Chop Block (mltrainingcore.py:125-134)

**Mechanism:** Every candle is classified as `trend`, `chop`, or `high_vol`. In `chop` regime, `stake_mult = 0.0`, which blocks trading entirely in both `MLStrategy` and `DualMLStrategy`.

```python
def detect_regime(row) -> str:
    trend_strength = abs(ema_20 - ema_100) / atr14
    if trend_strength < 0.6: return "chop"
```

**Impact:** During ranging/consolidation (which is >50% of crypto market time), all trading is blocked. This is intentional — chop is where trend-following strategies lose money — but it dramatically reduces total trade count.

**Suggestion:** Consider a "low-conviction" mode in chop with smaller stakes (e.g., 0.3× instead of 0.0×) and tighter SL. Or use mean-reversion logic in chop (different from trend).

### Gate 3: Volume Filter (dualmlstrategy.py:153-163)

**Mechanism:** When a non-HOLD signal arrives, it is blocked if `current_volume < 0.8 * SMA20(volume)`.

```python
if vol_sma20 > 0 and current_vol < vol_sma20 * 0.8:
    override to HOLD
```

**Impact:** Low-volume periods (e.g., weekend, holidays, low-vol regimes) block trades. This filter is sensible for avoiding illiquid entries, but it further reduces trade count.

**Suggestion:** Lower threshold from 0.8 to 0.5, or make it regime-dependent (tighter in trend, looser in chop).

### Gate 4: HTF Trend Filter (dualmlstrategy.py:165-183)

**Mechanism:** LONG signals require `close > EMA50` on 1h; SHORT signals require `close < EMA50`.  
This is a **strict higher-timeframe trend filter** that blocks counter-trend entries.

```python
if signal == LONG and not above_ema50: → HOLD
if signal == SHORT and above_ema50: → HOLD
```

**Impact:** In a strong uptrend, all SHORT signals are blocked. In a downtrend, all LONG signals are blocked. This eliminates ~50% of potential signals. But for a trend-following system this is standard. The problem is when both conditions fail simultaneously (price crossing EMA50 frequently) — the bot oscillates between LONG and SHORT blocks.

**Suggestion:** Allow counter-trend entries with reduced stake (e.g., 0.5×) instead of hard-blocking. Or use EMA200 instead of EMA50 for a looser filter.

### Gate 5: StrategicML `allow_trading` Veto (strategic/strategicml.py:18-22)

**Mechanism:** StrategicML predicts `vol_state` {0: low, 1: normal, 2: high, 3: extreme}. When `vol_state == 3.0` (extreme volatility), trading is completely blocked:

```python
_VOL_STATE_ALLOW = {0.0: True, 1.0: True, 2.0: True, 3.0: False}
```

**Impact:** During extreme volatility events (e.g., flash crashes, major news), the bot goes flat. This is sensible for risk but reduces trade count. How often this activates depends on market conditions during the testing window.

**Suggestion:** The vol_state 3.0 threshold (`EXTREME_VOL_RATIO = 2.5`) is high — short-term volatility must be 2.5× the long-term average. If this is activating frequently, consider raising to 3.0.

### Gate 6: RiskGuard Circuit Breaker (riskguard.py)

**Mechanism:** Tracks daily loss (max 5%) and max drawdown (max 15%). When either is breached, all trading halts until the next day.

**Impact:** If the bot hits a losing streak, it stops trading entirely for the remainder of the day. This is a hard cap on daily trade count.

**Suggestion:** These limits are reasonable. If they trigger too often, consider reducing position sizing instead of loosening the limits.

### Gate 7: Consecutive Signals Requirement (positionmanager.py:32)

**Mechanism:** Scaling up an existing position requires `CONSECUTIVE_SIGNALS_REQUIRED = 2` same-direction signals in a row. The history is tracked via a deque of the last 10 signals.

```python
CONSECUTIVE_SIGNALS_REQUIRED = 2
```

**Impact:** This delays scale-ups and means not every valid signal results in increased exposure.

**Suggestion:** Reduce to 1 for faster scaling, or increase to 3 for higher confidence.

### Gate 8: PositionManager Entry Stake Limits (positionmanager.py:97-102)

**Mechanism:** When opening a position, the stake is computed as:

```python
qty = (cash * stake_frac * max_exposure_frac) / price
```

Default values from `DEFAULT_PARAMS` show `stake_long_frac=0.10`, `stake_short_frac=0.05`, `max_exposure_frac` ranges from 0.3 to 1.0.

**Effective leverage exposure per trade:** 3%–10% of cash in LONG, 1.5%–5% in SHORT.

**Impact:** Small stakes mean small PnL per trade. Even with 10 wins, the wallet impact is marginal.

**Suggestion:** The existing `plans/improvements.md` already covers increasing these (Item 2). Key: expand the walk-forward param grid to include higher stake values.

### Gate 9: Max Hold Time Exits (positionmanager.py:68-72)

**Mechanism:** Positions are automatically closed after `max_hold_hours` (typically 2–8h depending on regime). This caps the duration and prevents riding extended trends.

```python
if elapsed_hours >= max_hold_hours:
    self._full_close("MAX_HOLD_TIME")
```

**Impact:** In the 1h timeframe, `max_hold_hours=4` means only 4 candles max. For the 15m tactical model, this means trades that would have become highly profitable if held longer are cut short.

**Suggestion:** Increase the upper bound of `max_hold_hours` in the param grid from 24h to 48h or 72h. Let the optimizer find the optimal value.

### Gate 10: Prediction History Warmup (tacticalml.py:32-60)

**Mechanism:** On startup, the tactical model needs `max_history_candles` (200 for 15m) + `min_feature_candles` (80 for 15m) = 280 candles of history (70 hours at 15m) before adaptive thresholding is functional. During warmup, `max_th/min_th` are `NaN`, producing only HOLD signals.

**Impact:** No trades for the first ~3 days after startup.

**Suggestion:** Warmup could use model confidence instead of adaptive thresholds — enter when prediction magnitude > some threshold (e.g., |pred| > 0.005).

---

## 3. Parameter Sensitivity: Impact of Threshold Tuning

Current vs. proposed ranges for the key tuning knobs:

| Parameter | Current Value | Tighter → Fewer Trades | Looser → More Trades |
|---|---|---|---|
| `adaptive_history_hours` | 50h | 100h (more selective) | 25h (more frequent) |
| `label_window_hours` | 16.7h | 24h (fewer, more extreme) | 8h (more signals) |
| `chop threshold` (trend_strength) | 0.6 | 0.8 (more chop detection) | 0.4 (less chop) |
| Volume filter | 80% SMA20 | 90% (stricter) | 50% (looser) |
| HTF EMA | EMA50 | EMA100 (looser filter) | EMA20 (tighter filter) |
| `stake_long_frac` | 0.10–0.25 | 0.05 (smaller) | 0.15–0.50 (larger) |
| `max_hold_hours` | 2–24h | 1–12h (shorter) | 4–72h (longer) |
| `max_daily_loss` | 5% | 3% (stricter) | 10% (looser) |
| `CONSECUTIVE_SIGNALS_REQUIRED` | 2 | 3 (fewer scales) | 1 (faster scaling) |

---

## 4. Model Architecture Observations

### 4.1 TacticalML Retrain Frequency

The tactical model is retrained **every 15m candle** (when the last timestamp changes in `dualmlstrategy.py:130`). Between 15m candles (5m heartbeat), it **reuses the cached signal**. This means:
- At most 4 retraining runs per hour
- The signal cannot change between 15m candles regardless of price action

**Suggestion:** For higher trade frequency, consider:
- Running tactical predictions on 5m instead of 15m (but feature quality may degrade)
- Adding a "confidence decay" — if the signal was `HOLD` for several consecutive iterations, allow re-evaluation mid-candle

### 4.2 Walk-Forward Labeling

The current simulation-based labeling (`dualmlsimulation.py`) uses `iterations=100` for the tactical model and a param grid with `stake_long` capped at 0.25. The strategic model (`strategictraining.py`) uses rule-based labels by default unless `--optimize-params` is passed.

The existing `plans/improvements.md` already identifies:
- **Item 2:** Increase position sizing via expanded param grid
- **Item 3:** Add leverage to walk-forward param grid
- **Item 8:** Make simulation-driven training the default

---

## 5. Recommended Actions (Prioritized)

### Tier 1: Quick Wins (Low Risk, Immediate Impact)

1. **Reduce adaptive threshold strictness** — Change `label_window_hours` from 16.7 to **8.0** in `timeframe_config.py` for the 15m preset. This doubles the number of expected signals (top/bottom 6 values instead of 3).

2. **Lower volume filter threshold** — Change 0.8 to **0.5** in `dualmlstrategy.py:157`. Reduces volume-based trade suppression during normal trading hours.

3. **Reduce chop threshold** — Change `trend_strength < 0.6` to **`< 0.4`** in `mltrainingcore.py:129`. Reduces chop detection to allow more trades in mildly ranging markets.

4. **Lower CONSECUTIVE_SIGNALS_REQUIRED** — Change from 2 to **1** in `positionmanager.py:32`. Enables immediate scale-up on any same-direction signal.

### Tier 2: Profitability Boost (Medium Risk)

5. **Expand walk-forward param grid** — Add higher stake values: `stake_long=[0.10, 0.15, 0.25, 0.35, 0.50]`, `stake_short=[0.05, 0.10, 0.15, 0.20, 0.30]` in `strategic/strategictraining.py` (and `mltraining.py` for legacy).

6. **Add leverage to param grid** — Add `recommended_leverage=[1, 2, 3, 5]` to the walk-forward search. This is already partially supported but not wired into the training pipeline's param search.

7. **Increase max_hold_hours upper bound** — Change from 24h to **48h** to let winning trades run longer.

### Tier 3: Structural Changes (Higher Risk)

8. **Regime-dependent-volume filter** — Make the volume threshold depend on regime: 0.5× in chop, 0.8× in trend. This preserves the safety filter during low-liquidity chop but doesn't block trend entries.

9. **Allow low-conviction chop entries** — Instead of `stake_mult = 0.0` in chop, use 0.3× with tighter SL (`stop_loss * 1.5`). This lets the bot participate in ranging markets with reduced risk.

10. **Warmup with absolute thresholds** — During the first `adaptive_history_candles` (before adaptive thresholds are available), use a fixed threshold (e.g., `|pred| > 0.003`) as an entry gate instead of holding indefinitely.

### Tier 4: System-Level Changes

11. **Config-driven thresholds** — Extract all tuning knobs (label_window_hours, adaptive_history_hours, chop threshold, volume filter, EMA span) into a centralized config file (YAML or JSON). This makes experimentation systematic rather than requiring code changes.

12. **Multi-timeframe entry confirmation** — For higher entry quality, require confirmation across two tactical timeframes (e.g., 15m and 5m both signaling the same direction) before entering. This would REDUCE trade frequency but INCREASE win rate.

13. **Dynamic position sizing based on prediction strength** — Scale position size proportionally to `|prediction - threshold|` rather than using a binary in/out with fixed stake. Stronger convictions get larger positions.

---

## 6. Estimated Impact

| Action | Trade Frequency | Per-Trade Profit | Risk | Effort |
|---|---|---|---|---|
| 1. Loosen adaptive thresholds | +2× | − | Low | 1 line |
| 2. Lower volume threshold | +1.2× | − | Low | 1 line |
| 3. Reduce chop threshold | +1.5× | −10% win rate | Low | 1 line |
| 4. Faster scaling | +1.3× | +1.2× | Low | 1 constant |
| 5. Bigger stakes | − | +2–3× | Medium | ~5 lines |
| 6. Leverage in grid | − | +1.5–2× | Medium | ~10 lines |
| 7. Longer max hold | − | +1.2× | Low | 1 constant |
| 8. Regime volume filter | +1.2× | − | Medium | ~10 lines |
| 9. Chop low-conviction | +2× | − | Medium | ~5 lines |
| 10. Warmup thresholds | +1.1× | − | Low | ~10 lines |

**Combined potential (Tier 1 + Tier 2):** 2–4× more trades AND 2–3× higher profit per trade.

---

## 7. Diagnostic Recommendations

To decide which gates are actually suppressing trades in the current test:

1. **Add entry gate logging** — Log WHICH gate blocked each potential signal (e.g., `GATE: volume_filter`, `GATE: htf_trend`, `GATE: chop_regime`). Currently only the final signal is logged, not the intermediate blocks.

2. **Run a counter** — Log the count of signals at each gate per day:
   ```
   15m candles processed: 96
   Non-HOLD signals: 24
   After volume filter: 18
   After HTF filter: 12
   After regime check: 8
   After RiskGuard: 8
   Actual entries: 4
   Scale-ups: 2
   ```

3. **Test regime distribution** — Log `% of candles in each regime` during the test period. If >50% is chop, the chop threshold is the dominant bottleneck.

4. **Plot prediction distribution** — TacticalML predictions are logged (`pred` values). Collect them and check if they cluster near zero (indicating model uncertainty) or if they're well-distributed but the adaptive thresholds are simply too wide.
