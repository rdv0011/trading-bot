# Dual-ML Bitcoin Trading Bot (Simplified)

Minimal trading system for BTCUSDT futures with in-memory tactical model, heuristic meta-params, and 15m entry confirmation filter. No strategic model.

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Simplified Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Tactical ML (1h)                                      │ │
│  │ - In-memory training at startup (Phase 2)              │ │
│  │ - No disk persistence, no walk-forward retraining     │ │
│  │ - WALKFORWARD_RETRAIN_EVERY=0                        │ │
│  │ - 180 days of 1h data, balanced labels, early-stopped │ │
│  │ - Tree count: 30 (early stopped)                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Position Mgmt & Filters                               │ │
│  │ - 15m entry confirmation (Phase 3): RSI, volume,      │ │
│  │   momentum gates before entering                      │ │
│  │ - Heuristic meta-params replace strategic model       │ │
│  │ - Adaptive thresholds (ATR, vol ratio, liquidity)    │ │
│  │ - Trailing SL, circuit breaker, regime gates         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point: train, simulate, live, compare |
| `config.py` | All hyperparameters in one place |
| `data.py` | Download, features, labels, train/val split |
| `model.py` | CatBoost tactical model (in-memory training) |
| `simulate.py` | MockBroker + validation backtest |
| `broker.py` | BinanceBroker (live) + MockBroker (sim) |
| `strategy.py` | Simplified strategy logic with 15m filter |
| `logger.py` | Console + file + trade CSV logging |
| `compare.py` | Sim vs Demo comparison + HTML report |
| `utils.py` | Shared utilities |
| `fancontrol/` | GPIO PWM fan controller (systemd + libgpiod v2) — runs independently of the bot |
| `keepawake/` | macOS LaunchAgent to keep the host awake 24/7 |

## Quick Start

```bash
# 1. Install dependencies
pip install catboost pandas numpy scikit-learn python-dotenv

# 2. Create .env file from template
cp .env.example .env
# Edit .env and add your Binance testnet API credentials

# 3. Train models (tactical model is now trained in-memory during live startup)
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

Downloads historical data, engineers features, and trains models:

```bash
python main.py train --symbol BTCUSDT --days 180
```

**What it does:**
1. Downloads 180 days of 1h candles from Binance
2. Engineers features: returns, EMAs, ATR, volatility, time encoding, regime
3. Generates labels: future_ret (lookahead candles for 1h timeframe)
4. Splits chronologically: 80% train, 20% validation
5. Trains tactical model (CatBoost, iterations=300, early-stopped to 30)
6. Saves to `models/` directory (optional for live - model trained in-memory)

**Output:**
```
models/
├── model_tactical.cbm      # Tactical CatBoost model
├── model_tactical_meta.json # Tactical metadata
```

### Simulate Mode

Runs backtest on validation data with MockBroker:

```bash
python main.py simulate --model-dir models/
```

**What it does:**
1. Loads validation data
2. Loads tactical model
3. Runs walk-forward tactical predictions (retrain_every=100 candles)
4. Gets meta-parameters from heuristic (replaces strategic model)
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
- **Phase 2**: Tactical model trained in-memory on 180 days of 1h data at startup (no disk persistence)
- **Phase 3**: 15m entry confirmation filter - validates 1h signals using 15m data (RSI, volume z-score, momentum)
- No periodic retraining (WALKFORWARD_RETRAIN_EVERY=0)
- Rate limiter (1000 weight/min)
- Circuit breaker on -1003 errors
- Position caching (2s TTL)
- Bracket orders (TP/SL)
- Daily trade logging

**Startup sequence:**
1. Train tactical model in-memory on 180 days of 1h data
2. Validate tree count (min 10 trees)
3. Create strategy with feature_cols from training data
4. Start live loop with 15m confirmation filter

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
TIMEFRAME = "1h"                      # TACTICAL_TF (was 15m)
TACTICAL_TF = "1h"                    # Alias for compatibility
HISTORY_DAYS = 180                    # STRATEGIC_DAYS
TRAIN_FRACTION = 0.8

# Features
FEATURE_LAGS = [1, 2, 3, 5, 10, 20, 50]
EMA_SPANS = [5, 10, 20, 50, 100]
ATR_PERIOD = 14
LABEL_HORIZON = 4  # 4 candles for 1h timeframe

# Tactical Model
TACTICAL_MODEL_PARAMS = {
    "iterations": 300,                # TACTICAL_ITERATIONS
    "depth": 6,
    "learning_rate": 0.05,
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
WALKFORWARD_RETRAIN_EVERY = 0         # Disabled (Phase 2)

# Phase 3: 15m Confirmation Filter
CONFIRM_15M_ENABLED = True            # Enable 15m entry confirmation
CONFIRM_15M_TF = "15m"                # Confirmation timeframe
CONFIRM_15M_LIMIT = 200               # Candles to fetch (~50h)
CONFIRM_15M_RSI_PERIOD = 14           # RSI lookback
CONFIRM_15M_RSI_MIN_LONG = 30.0       # RSI must be above this (not oversold) for long
CONFIRM_15M_RSI_MAX_LONG = 70.0       # RSI must be below this (not overbought) for long
CONFIRM_15M_RSI_MIN_SHORT = 30.0      # RSI must be above this for short
CONFIRM_15M_RSI_MAX_SHORT = 70.0      # RSI must be below this (not overbought) for short
CONFIRM_15M_VOLUME_MIN_ZSCORE = 1.0   # Require volume spike above this z-score
CONFIRM_15M_MOMENTUM_MIN = 0.0005     # Min abs momentum (|pct_change|) required

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
- `TACTICAL_TIMEFRAME` - Tactical timeframe (default: 1h)
- `STRATEGIC_DAYS` - Tactical training days (default: 180)
- `TACTICAL_ITERATIONS` - Tactical model iterations (default: 300)
- `LEVERAGE` - Default leverage (default: 1.0)
- `SIMULATION_FEE` - Fee per side (default: 0.0004)
- `SIMULATION_SLIPPAGE` - Fixed slippage (default: 0.0003)
- `WALKFORWARD_RETRAIN_EVERY` - Retrain interval (default: 0, disabled)
- `CONFIRM_15M_ENABLED` - Enable 15m entry confirmation (default: True)

**Required for live trading** (in `.env`):
- `BINANCE_TESTNET_FUTURES_API_KEY` / `BINANCE_TESTNET_FUTURES_API_SECRET`
- `BINANCE_TESTNET_SPOT_API_KEY` / `BINANCE_TESTNET_SPOT_API_SECRET` (for spot)

## Features

| Feature | Description |
|---------|-------------|
| Phase 2: In-memory training | Tactical model trained on 180 days of 1h data at startup, no disk persistence |
| Phase 3: 15m confirmation | RSI(14), volume z-score, momentum filters validate 1h signals before entry |
| Walk-forward disabled | No periodic retraining (WALKFORWARD_RETRAIN_EVERY=0) |
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
- Tactical prediction, meta-parameters

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

## Running 24/7 on a Mac

To prevent the machine from sleeping while the bot runs:

```bash
keepawake/keepawake.sh
```

To cancel:

```bash
keepawake/cancelkeepawake.sh
```

The `keepawake/` folder installs a macOS LaunchAgent
(`com.keepawake.caffeinate.plist`) that runs `/usr/bin/caffeinate -dimsu`.
This is host-level maintenance — it runs independently of the bot code.

---

## Fan Control (CPU Cooling)

During heavy training the CPU runs at 100% for extended periods. The
`fancontrol/` folder contains a libgpiod v2 PWM fan controller that
ramps fan speed based on CPU temperature, managed as a **separate
systemd service** — it runs on its own and never interacts with the bot code.

### Quick Start (Radxa Zero 3W)

```bash
# 1. Deploy the script and service
sudo cp fancontrol/fan_pwm.py /usr/local/bin/fan_pwm.py
sudo chmod +x /usr/local/bin/fan_pwm.py
sudo cp fancontrol/fan-pwm.service /etc/systemd/system/fan-pwm.service

# 2. Enable and start
sudo systemctl daemon-reload
sudo systemctl enable fan-pwm.service
sudo systemctl start fan-pwm.service

# 3. Verify
sudo systemctl status fan-pwm.service
```

### Manual Test

```bash
# Verify GPIO (gpiochip3 line 20 = physical pin 7)
sudo gpioset -c gpiochip3 20=1   # fan ON
sudo gpioset -c gpiochip3 20=0   # fan OFF

# Run the controller directly
sudo python3 /usr/local/bin/fan_pwm.py
# Ctrl+C to stop
```

### Full Documentation

- **[`fancontrol/README.md`](fancontrol/README.md)** — hardware wiring, circuit diagram, prerequisites
- **[`fancontrol/UPDATE_FAN_CONTROL.md`](fancontrol/UPDATE_FAN_CONTROL.md)** — full setup guide: hardware verification, temperature curve tuning, stress testing with `stress-ng`, service management, logs via `journalctl`

---

## Troubleshooting

### "No validation data found"
Run `python main.py train` first to create train/val splits.

### "Model not loaded"
Ensure `models/` directory exists with trained models (optional for live mode - model trained in-memory).

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