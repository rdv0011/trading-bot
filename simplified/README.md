# Dual-ML Bitcoin Trading Bot (Simplified)

Minimal dual-ML trading system for BTCUSDT futures with simulation vs demo comparison.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Dual-ML Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐      ┌─────────────────┐              │
│  │ Tactical ML     │      │ Strategic ML    │              │
│  │ (15m)           │      │ (1h)            │              │
│  │                 │      │                 │              │
│  │ - Walk-forward  │      │ - Batch predict │              │
│  │ - Retrain every │──────│ - Meta-params:  │              │
│  │   N candles     │      │   stake, SL, TP,│              │
│  │ - Predicts:     │      │   leverage,     │              │
│  │   future_ret    │      │   regime        │              │
│  └────────┬────────┘      └────────┬────────┘              │
│           │                        │                        │
│           ▼                        ▼                        │
│  ┌─────────────────┐      ┌─────────────────┐              │
│  │ Signal Logic    │      │ Position Mgmt   │              │
│  │ pred > thr →    │      │ stake × equity  │              │
│  │   LONG          │      │ SL/TP brackets  │              │
│  │ pred < -thr →   │      │ Max hold time   │              │
│  │   SHORT         │      │                 │              │
│  └─────────────────┘      └─────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point: train, simulate, live, compare |
| `config.py` | All hyperparameters in one place |
| `data.py` | Download, features, labels, train/val split |
| `model.py` | CatBoost models (tactical + strategic) |
| `simulate.py` | MockBroker + validation backtest |
| `broker.py` | BinanceBroker (live) + MockBroker (sim) |
| `strategy.py` | Dual-ML strategy logic |
| `logging.py` | Console + file + trade CSV logging |
| `compare.py` | Sim vs Demo comparison + HTML report |
| `utils.py` | Shared utilities |

## Quick Start

```bash
# 1. Install dependencies
pip install catboost pandas numpy scikit-learn python-dotenv

# 2. Create .env file from template
cp .env.example .env
# Edit .env and add your Binance testnet API credentials

# 3. Train both models
python main.py train

# 4. Run simulation on validation data
python main.py simulate

# 5. Compare simulation vs demo logs
python main.py compare
```

## Environment Variables (.env)

The bot reads all configuration from a `.env` file in the repo root. Copy `.env.example` to `.env` and fill in your credentials:

```bash
# Required for live/testnet trading
BINANCE_TESTNET_FUTURES_API_KEY=your_testnet_futures_api_key
BINANCE_TESTNET_FUTURES_API_SECRET=your_testnet_futures_api_secret

# Optional: for spot trading
# BINANCE_TESTNET_SPOT_API_KEY=
# BINANCE_TESTNET_SPOT_API_SECRET=

# Optional: override defaults
# SYMBOL=BTCUSDT
# MARKET_TYPE=futures
# TESTNET=true
# SLEEPTIME=5m
```

Get testnet API keys from: https://testnet.binance.vision/

## Modes

### Train Mode

Downloads historical data, engineers features, and trains both models:

```bash
python main.py train --symbol BTCUSDT --days 90
```

**What it does:**
1. Downloads 90 days of 15m + 1h candles from Binance testnet
2. Engineers features: returns, EMAs, ATR, volatility, time encoding, regime
3. Generates labels: future_ret (4 candles ahead for 15m)
4. Splits chronologically: 80% train, 20% validation
5. Trains tactical model (CatBoost, iterations=100)
6. Trains strategic model (CatBoost, iterations=300)
7. Saves to `models/` directory

**Output:**
```
models/
├── model_tactical.cbm      # Tactical CatBoost model
├── model_tactical_meta.json # Tactical metadata
├── model_strategic.cbm      # Strategic CatBoost model
└── model_strategic_meta.json # Strategic metadata
```

### Simulate Mode

Runs backtest on validation data with MockBroker:

```bash
python main.py simulate --model-dir models/
```

**What it does:**
1. Loads validation data
2. Loads both models
3. Runs walk-forward tactical predictions (retrains every 100 candles)
4. Gets strategic meta-parameters (batch prediction)
5. Simulates trades with MockBroker (fees, slippage, brackets)
6. Calculates metrics (return, Sharpe, drawdown, win rate)
7. Saves trades CSV + equity curve

**Output:**
```
logs/
├── trades_sim_20260831_223000.csv  # All simulated trades
└── equity_sim_20260831_223000.csv  # Equity curve
```

**Metrics:**
- Total return %
- Sharpe ratio
- Max drawdown %
- Win rate
- Profit factor
- Average trade PnL

### Live Mode

Runs live trading on Binance testnet (reads API keys from .env):

```bash
python main.py live \
  --sleep 60 \
  --model-dir models/
```

**Features:**
- Rate limiter (1000 weight/min)
- Circuit breaker on -1003 errors
- Position caching (2s TTL)
- Bracket orders (TP/SL)
- Daily trade logging

### Compare Mode

Compares simulation vs demo trading logs:

```bash
python main.py compare \
  --log-dir logs/ \
  --demo-pattern "trading_*.log" \
  --sim-pattern "trades_sim_*.csv" \
  --output logs/comparison_report.html
```

**What it does:**
1. Parses demo log files (extracts trades)
2. Loads simulation trade CSVs
3. Aligns trades by timestamp
4. Calculates aggregate metrics comparison
5. Generates HTML report with charts

**Report includes:**
- Summary metrics table
- Win rate comparison
- PnL by regime (trend, chop, high_vol)
- Trade-by-trade comparison
- Entry/exit price differences

## Configuration

All parameters in `config.py` with `.env` overrides:

Configuration is loaded from `config.py` which reads `.env` file at startup. All values below can be overridden via environment variables:

```python
# Data (overridable via .env)
SYMBOL = "BTCUSDT"                    # SYMBOL
TACTICAL_TF = "15m"                   # TACTICAL_TIMEFRAME
STRATEGIC_TF = "1h"                   # STRATEGIC_TIMEFRAME
HISTORY_DAYS = 90                     # STRATEGIC_DAYS
TRAIN_FRACTION = 0.8

# Features
FEATURE_LAGS = [1, 2, 3, 5, 10, 20, 50]
EMA_SPANS = [5, 10, 20, 50, 100]
ATR_PERIOD = 14
LABEL_HORIZON = 4  # 4 candles = 1 hour ahead

# Tactical Model
TACTICAL_MODEL_PARAMS = {
    "iterations": 100,                # TACTICAL_ITERATIONS
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
}

# Strategic Model
STRATEGIC_MODEL_PARAMS = {
    "iterations": 300,                # STRATEGIC_ITERATIONS
    "depth": 8,
    "learning_rate": 0.03,
    "loss_function": "RMSE",
}

# Trading Defaults
STAKE_LONG_FRAC_DEFAULT = 0.10
STAKE_SHORT_FRAC_DEFAULT = 0.05
STOP_LOSS_FRAC_DEFAULT = 0.02
TAKE_PROFIT_FRAC_DEFAULT = 0.04
MAX_HOLD_HOURS_DEFAULT = 4.0
LEVERAGE_DEFAULT = 1.0                # LEVERAGE

# Signal Threshold
ABSOLUTE_THRESHOLD = 0.003

# Simulation
INITIAL_EQUITY = 1.0
FEE = 0.0004                          # SIMULATION_FEE
SLIPPAGE = 0.0003                     # SIMULATION_SLIPPAGE
WALKFORWARD_RETRAIN_EVERY = 100

# API Credentials (from .env only, NOT hardcoded)
# BINANCE_TESTNET_FUTURES_API_KEY
# BINANCE_TESTNET_FUTURES_API_SECRET
# BINANCE_TESTNET_SPOT_API_KEY
# BINANCE_TESTNET_SPOT_API_SECRET
```

**Environment variable overrides** (set in `.env`):
- `SYMBOL` - Trading pair (default: BTCUSDT)
- `MARKET_TYPE` - futures or spot (default: futures)
- `TESTNET` - true/false (default: true)
- `TACTICAL_TIMEFRAME` - Tactical timeframe (default: 5m)
- `STRATEGIC_TIMEFRAME` - Strategic timeframe (default: 1h)
- `STRATEGIC_DAYS` - Strategic training days (default: 365)
- `TACTICAL_DAYS` - Tactical training days (default: 45)
- `TACTICAL_ITERATIONS` - Tactical model iterations (default: 300)
- `STRATEGIC_ITERATIONS` - Strategic model iterations (default: 300)
- `LEVERAGE` - Default leverage (default: 1.0)
- `SIMULATION_FEE` - Fee per side (default: 0.0004)
- `SIMULATION_SLIPPAGE` - Fixed slippage (default: 0.0003)
- `WALKFORWARD_RETRAIN_EVERY` - Retrain interval (default: 100)

**Required for live trading** (in `.env`):
- `BINANCE_TESTNET_FUTURES_API_KEY` / `BINANCE_TESTNET_FUTURES_API_SECRET`
- `BINANCE_TESTNET_SPOT_API_KEY` / `BINANCE_TESTNET_SPOT_API_SECRET` (for spot)

## Features

| Feature | Description |
|---------|-------------|
| Walk-forward | Tactical model retrains every 100 candles |
| Regime detection | trend / chop / high_vol classification |
| Bracket orders | Automatic TP/SL on entry |
| Rate limiting | 1000 weight/min with circuit breaker |
| Position caching | 2s TTL to reduce API calls |
| Daily logging | Rotating log files with 10-day retention |
| Trade CSV | Detailed trade log for analysis |

## Logs

### Daily Log File
```
logs/trading_2026-08-31.log
```
Contains all DEBUG/INFO/WARNING messages with timestamps.

### Trade CSV
```
logs/trades_2026-08-31.csv
```
One row per completed trade with:
- Timestamp, symbol, side, entry/exit prices
- PnL, regime, exit reason
- Tactical prediction, strategic parameters

### Equity Curve
```
logs/equity_2026-08-31.csv
```
Timestamped equity values for plotting.

## Regime Logic

| Regime | Condition | Stake Multiplier |
|--------|-----------|------------------|
| trend | trend_strength > 0.4, vol_ratio ≤ 1.4 | 1.0 |
| high_vol | vol_ratio > 1.4 | 0.5 |
| chop | trend_strength < 0.4 | 0.3 |

## Exit Conditions

1. **Stop Loss**: PnL ≤ -stop_loss_frac (default 2%)
2. **Take Profit**: PnL ≥ take_profit_frac (default 4%)
3. **Max Hold**: Time in position ≥ max_hold_hours (default 4h)
4. **Signal Reversal**: Opposite signal while in position

## Troubleshooting

### "No validation data found"
Run `python main.py train` first to create train/val splits.

### "Model not loaded"
Ensure `models/` directory exists with trained models.

### "Rate-limit cooldown"
Normal behavior. Wait for cooldown period (exponential backoff up to 5min).

### API connection failed
Check API keys are set in `.env`:
```bash
cat .env | grep BINANCE
```
Or verify the config module loads them:
```bash
python -c "from config import validate_credentials; validate_credentials('futures', True)"
```