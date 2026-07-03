# Deferred Profitability Improvements

> Items from the profitability analysis that carry **Medium** or **Medium–High** risk.
> Deferred — not implemented in the `profitability-improvements` branch.
> Each item includes the rationale, proposed change, and risk assessment so it can be
> picked up independently.

---

## Item 2 — Increase Position Sizing (Stake Fractions)

**Risk:** Medium  
**Expected profit boost:** 2–3× return  
**Trade-off:** Higher drawdown; drawdown is already bounded by RiskGuard (15% max).

### Current values

| Parameter | Current | Regime adjustment |
|---|---|---|
| `stake_long_frac` | 0.10 (10%) | trend: 0.20, others: 0.10 |
| `stake_short_frac` | 0.05 (5%) | trend: 0.10, others: 0.05 |
| `max_exposure_frac` | 0.3–1.0 (via vol_ratio) | |

Effective exposure per trade is 5–10% of capital, which is conservative for a
futures account with leverage.

### Proposed change

Expand the walk-forward param grid so the optimizer can select larger sizes:

```python
# In strategictraining.py _build_strategic_labels_from_simulation()
param_grid = build_param_grid(
    stake_short=[0.05, 0.10, 0.15, 0.20, 0.30],
    stake_long=[0.10, 0.15, 0.25, 0.35, 0.50],
    ...,
)
```

Also raise the rule-based defaults for `stake_long_frac` / `stake_short_frac`
in `_build_strategic_labels()` when the market regime is favourable.

### Files touched

- `strategic/strategictraining.py` — rule-based labels
- `strategic/strategicfeatures.py` — `max_exposure_frac` thresholds

---

## Item 3 — Add Leverage to the Walk-Forward Param Grid

**Risk:** Medium  
**Expected profit boost:** 1.5–2× return  
**Trade-off:** Leverage amplifies losses; needs careful validation.

### Current state

The walk-forward search (`--optimize-params`) optimises `stake_*_frac`,
`stop_loss`, `max_hold_hours` — but **not** leverage. Leverage is hardcoded
per regime in `_REGIME_TO_LEVERAGE`:
- trend → 5×, high_vol → 2×, chop → 1×

### Proposed change

1. Add `leverage` to the `build_param_grid()` product:
   ```python
   leverage=[1, 2, 3, 5],
   ```
2. Add `recommended_leverage` as a prediction target in the strategic model
   (it is already a field in `StrategicDecision`).
3. Ensure `simulate_trades_core()` reads `recommended_leverage` from param
   dicts and applies it to PnL. **(Item 1 already did this for the
   simulation engine — the remaining work is only in the training pipeline.)**

### Files touched

- `mltraining.py` — `build_param_grid()` signature
- `strategic/strategictraining.py` — param grid passed to
  `_build_strategic_labels_from_simulation()`
- `positionmanager.py` — `StrategicDecision` already has
  `recommended_leverage`; no change needed

---

## Item 4 — Relax the RiskGuard Leverage Cap

**Risk:** Medium–High  
**Expected profit boost:** 1–2× return  
**Trade-off:** Higher potential loss per trade; liquidation risk increases.

### Current state

`RiskGuard` initialises with `max_leverage=10.0` in `dualmlstrategy.py:83`.
The strategic model maps regimes to 1×–5×, so the cap rarely binds, but it
cannot go above 5× even when market conditions warrant it.

### Proposed change

Increase `max_leverage` to **20×** in the RiskGuard config:

```python
self.risk_guard = RiskGuard(
    max_leverage=20.0,
    ...
)
```

The strategic model remains the actual decision-maker — RiskGuard is a
circuit breaker. With walk-forward optimisation selecting optimal leverage
(Item 3), the cap should be high enough not to constrain the search.
Daily loss (5%) and drawdown (15%) limits remain as hard stops.

### Files touched

- `dualmlstrategy.py` — RiskGuard instantiation

---

## Item 7 — Boost Consecutive Signal Scaling

**Risk:** Medium  
**Expected profit boost:** 1.2–1.5× return  
**Trade-off:** Larger positions during streaks amplify both gains and losses.

### Current state

| Constant | Value | Effect |
|---|---|---|
| `CONSECUTIVE_SIGNALS_REQUIRED` | 2 | Scale on every 2nd same-direction signal |
| `SCALE_INCREMENT_FRAC` | 0.5 | Each scale adds 50% of original size |
| `MAX_SCALE_COUNT` | 3 | Up to 2.5× original size |

### Proposed change

Make scaling more aggressive to capture extended trends:

- `SCALE_INCREMENT_FRAC` 0.5 → **1.0** (double down on each scale)
- `MAX_SCALE_COUNT` 3 → **5** (up to 6× original size with 5 doubles)
- `CONSECUTIVE_SIGNALS_REQUIRED` 2 → **1** (scale on every signal, not
  every other)

These are tunable constants in `positionmanager.py`.

### Files touched

- `positionmanager.py` — module-level constants

---

## Item 8 — Always Use Simulation-Driven Training (`--optimize-params`)

**Risk:** Low (operational, no code change)  
**Expected profit boost:** 1.5–3× return vs rule-based training  
**Trade-off:** Higher compute cost (~12,960 CatBoost fits for 45d of 5m data).

### Current state

The strategic model can be trained with either:
- **Rule-based labels** (`_build_strategic_labels`) — heuristics based on
  regime, volatility ratio, and EMA cross. Fast but simplistic.
- **Simulation-driven labels** (`_build_strategic_labels_from_simulation`) —
  walk-forward search over a param grid labels each 24h window with the
  parameter set that maximised trading PnL. Much more realistic.

The flag `--optimize-params` selects the simulation-driven path. Without it,
the rule-based path is used.

### Proposed change

Make simulation-driven labels the **default** when 5m/tactical predictions
are available. This could be done by:

1. Always running `run_predictions_only()` inside `run_training()` when
   `df_5m_predictions` is `None` — removing the need for the separate
   `--optimize-params` step.
2. Or simply documenting that `--optimize-params` should always be used in
   production and cron schedules.

### Recommended workflow

```bash
python main.py --train-strategic --optimize-params --strategic-days 180 --tactical-days 45
```

### Files touched

- None (flag already exists) — or `strategic/strategictraining.py` to
  change the default behaviour.
