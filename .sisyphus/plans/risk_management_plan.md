# 📊 BTC Futures Risk & Positioning System (ML + Deterministic Hybrid)

This document describes a full framework for:

1. **Deriving risk & leverage regimes from long-term BTC history (10+ years)**
2. **Calculating position size dynamically**
3. **Designing ML architecture for mode selection**
4. **Comparing ML vs deterministic approaches**

---

# 🧠 1. Long-Term Risk & Leverage Modeling (Cycle-Aware)

## Goal
Extract **market regimes** (low/high volatility, bull/bear, crash risk) from long-term BTC data.

---

## 📥 Data Required (10+ years)
- OHLCV (daily + intraday)
- Funding rates (if futures)
- Open interest
- Volatility metrics- Drawdowns

---

## 📊 Core Features to Compute

### Volatility
- Rolling std (7d, 30d, 90d)
- ATR (Average True Range)
- Realized volatility

### Drawdowns
- Max drawdown (rolling windows)
- Tail events (e.g., worst 1% moves)

### Trend
- Moving averages (50D / 200D)
- Market phase (bull / bear)

### Liquidity / Structure
- Volume spikes- Wicks (high-low vs close-open)

---

## 🔁 Regime Classification (Deterministic)

Cluster historical data into regimes:

| Regime | Description | Example |
|--------|------------|--------|
| Low Vol | Stable market | 2019 sideways |
| Bull Trend | Uptrend + moderate vol | 2020–2021 |
| High Vol | Large swings | 2022 |
| Crash Risk | Extreme tails | March 2020 |

---

## 📉 Risk & Leverage Tables

For each regime compute:

| Metric | Meaning |
|--------|--------|
| Max daily drop (p99) | Tail risk |
| Avg volatility | Baseline movement |
| Max wick size | Liquidation risk |
| Safe leverage | 1 / max drawdown |

---

### Example Table

| Regime | Max Drop | Safe Leverage | Risk % |
|--------|----------|---------------|--------|
| Low Vol | 5% | 10× | 2% |
| Bull | 10% | 5× | 1.5% |
| High Vol | 20% | 2.5× | 1% |
| Crash | 40% | 1–2× | 0.5% |

---

## ⚠️ Key Insight

> Risk & leverage should be derived from **historical worst-case scenarios**, not guessed.

---

# 🧮 2. Position Size Calculation

## Inputs
- `repo` (account size)
- `entry_price`
- `stop_price`
- `risk_percent`
- `leverage`

---

## Core Formularisk_amount = repo * risk_percent
price_risk = abs(entry_price - stop_price)
position_size_btc = risk_amount / price_risk
position_value = position_size_btc * entry_price
implied_leverage = position_value / repo

---

## Modes

| Mode | Risk | Leverage Cap |
|------|------|--------------|
| Conservative | 0.5% | 2× |
| Normal | 1% | 3× |
| Aggressive | 2% | 5× |

---

## Adjustment Rule
final_leverage = min(implied_leverage, leverage_cap)

---

# 🤖 3. ML Architecture

## 🎯 Objective
Predict **trading mode**:
- Conservative
- Normal- Aggressive

---

## 🏗️ Option A: Single Model (Recommended Start)

### Input Features
- Volatility metrics
- Drawdowns
- Trend indicators- Volume/liquidity
- Funding rates

### Output
Mode ∈ {Conservative, Normal, Aggressive}

---

## 🧪 Labeling Strategy

Labels derived from history:

- High volatility → Conservative
- Medium → Normal- Low volatility → Aggressive

---

## ⚠️ Important

> Model predicts **mode**, NOT raw leverage/risk numbers.

---

# 🤖 4. Dual-Model Architecture (Advanced)

## Model 1: Risk/Leverage Predictor
Predicts:
- Expected volatility
- Tail risk (max drop)
- Safe leverage

## Model 2: Mode Classifier (e.g. CatBoost)
Uses:
- Market features
- Model 1 outputs

Outputs:
Mode (C / N / A)

---

## Pros
- More adaptive
- Captures nonlinear relationships

## Cons
- Harder to train
- Risk of overfitting
- Less interpretable

---

# 🔄 5. Training Pipeline## Step 1: Feature Engineering
Compute all indicators from historical data

## Step 2: Compute Risk Tables
- Rolling windows (e.g. 30d, 90d)
- Extract worst-case drops

## Step 3: Label Data
Assign mode per day## Step 4: Train Model
- CatBoost / XGBoost
- Classification

---

# ⚡ 6. Inference Pipeline (Live Trading)

## Daily Process

1. Compute latest features
2. Compute deterministic risk metrics
3. Feed into ML model
4. Get mode

---

## Intraday Process- Recompute features (short timeframe)
- Re-run model
- Adjust mode dynamically

---

## Final Execution Flow
Market Data → Features → Risk Metrics → ML Model → Mode
↓
Position Size Calculation
↓
Order Execution

---

# ❓ Should Risk & Leverage Be Predicted?

## ❌ Not recommended (initially)

Reasons:
- Hard to generalize
- Tail events are rare
- ML struggles with extremes

---

## ✅ Better Approach

> Use **deterministic risk calculations** + ML for **mode switching**

---

# 🧠 7. Deterministic Alternative (Highly Recommended)

## Pure Rule-Based System

### Step 1: Compute volatility

### Step 2: Map to mode
if volatility > high_threshold:
mode = "Conservative"
elif volatility > mid_threshold:
mode = "Normal"
else:
mode = "Aggressive"

---

## Advantages

- Transparent
- Robust to regime shifts- No training required
- Easier debugging

---

## Hybrid (Best Option)

| Component | Method |
|----------|--------|
| Risk calculation | Deterministic |
| Mode selection | ML or rules |
| Position sizing | Deterministic |

---

# 🔥 Final Recommendation

## Start with:

✅ Deterministic system:
- Volatility → Mode
- Mode → Risk & Leverage
- Position sizing formula

---

## Then add ML:

- Predict **mode only**
- Keep risk math deterministic

---

## Avoid:

❌ Predicting leverage directly  
❌ Fully ML-driven risk system  

---

# 🧩 Summary

- Use **10-year data** to derive risk regimes  
- Compute **worst-case scenarios (tail risk)**  
- Use **fixed formulas for position sizing**  
- Use ML only for **mode switching**  
- Prefer **hybrid system for robustness**  

---

This structure gives you:
- Stability (deterministic core)
- Adaptability (ML layer)
- Safety (tail-risk awareness)

---