# Trading Bot

A modular crypto trading bot powered by **CatBoost** and **pandas**, built around a two-tier machine learning system for live futures and spot trading on Binance.

Inspired by:
* [Real-time head-to-head adaptive modeling](https://emergentmethods.medium.com/real-time-head-to-head-adaptive-modeling-of-financial-market-data-using-xgboost-and-catboost-995a115a7495)
* [FreqAI: from price to prediction](https://emergentmethods.medium.com/freqai-from-price-to-prediction-6fadac18b665)

---

## How it works

The bot runs two ML models simultaneously:

**Tier 1 — TacticalML (5m, ephemeral)**
Retrained from scratch on every candle using a rolling window of recent 5m data. Never saved to disk. Produces `LONG`, `SHORT`, or `HOLD` signals using adaptive min/max thresholds computed from prediction history.

**Tier 2 — StrategicML (1h, persisted)**
Trained offline on historical data and saved to `model/`. Loaded at startup and hot-swapped at runtime when a new model file appears. Controls leverage, position sizing, stop-loss, take-profit, and max hold time. Acts as a gatekeeper: if it detects extreme volatility or a choppy regime, no trades are opened regardless of what TacticalML signals.

By default the strategic model is trained using rule-based labels derived from market regime and volatility. With `--optimize-params` it instead uses **simulation-driven labels**: a walk-forward search evaluates a grid of trading parameters against the actual tactical signal and labels each window with the combination that maximised the objective score. This makes StrategicML learn parameters that performed best historically rather than parameters that replicate hand-coded rules.

**PositionManager**
Sits between the two models and the broker. Tracks open positions and implements:
- *Consecutive signal scaling*: two consecutive signals in the same direction increase the position size
- *Gradual close*: each signal in the opposite direction closes a fraction of the position
- *Veto enforcement*: immediately closes any open position when StrategicML blocks trading

**RiskGuard**
A circuit breaker that enforces hard limits on daily loss, drawdown, and leverage. Configured via `parameters` in the strategy and checked before every trade.

---

## Installation

```bash
gh repo clone rdv0011/trading-bot
cd trading-bot
conda env create -f environment.yml
conda activate tradingbot
```

---

## Usage

### Run the bot

```bash
python main.py
```

Runs the dual-ML strategy on Binance Futures testnet by default.

```bash
python main.py --market-type futures --strategy dual
```

Available flags:

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--market-type` | `futures`, `spot` | `futures` | Binance market |
| `--strategy` | `dual`, `legacy` | `dual` | `dual` = two-tier ML, `legacy` = original single-ML |

---

### Train the strategic model

The strategic model must be trained before the first run. Training fetches historical data from Binance (public endpoint — no API credentials required) and saves the model to `model/`.

#### Rule-based training (fast, default)

Labels are derived from market regime and volatility heuristics.

```bash
python main.py --train-strategic --strategic-days 25
```

Or directly via the training script:

```bash
python strategic/strategictraining.py --symbol BTCUSDT --days 25 --timeframe 1h
```

#### Simulation-driven training (recommended)

Labels are produced by a walk-forward parameter search: for each 24h window the parameter combination that maximised the trading objective score against the tactical signal is selected as the training target. Requires generating 5m tactical predictions first (controlled by `--tactical-days`).

```bash
python main.py --train-strategic --optimize-params --strategic-days 25 --tactical-days 25
```

Available flags for `--train-strategic`:

| Flag | Default | Description |
|------|---------|-------------|
| `--strategic-days` | `365` | Days of 1h historical data for strategic model training |
| `--strategic-timeframe` | `1h` | Candle interval for strategic features |
| `--optimize-params` | off | Use simulation-driven parameter optimisation |
| `--tactical-days` | `45` | Days of 5m data for walk-forward param search (requires `--optimize-params`) |

The trained model is saved with a UTC timestamp (e.g. `strategic_meta_model_20260101T020000Z.pkl`). The running bot detects the new file and hot-swaps it automatically — no restart needed.

---

### Run a backtest

`dualmlsimulation.py` runs a full walk-forward backtest of the two-tier system over historical data without touching the live broker.

```bash
python dualmlsimulation.py --symbol BTCUSDT --days 25 --timeframe 5m
```

Available flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--symbol` | `BTCUSDT` | Trading pair |
| `--days` | `45` | Days of historical data to simulate |
| `--timeframe` | `5m` | Candle interval for tactical predictions |
| `--model-dir` | `model/` | Directory containing the trained strategic model |

The script produces three files in `labeleddata/`:

| File | Contents |
|------|----------|
| `dual_*_featured.csv` | 5m OHLCV with tactical features and labels |
| `dual_*_predictions.csv` | Walk-forward tactical predictions over the full window |
| `dual_*_final_test_sim.csv` | Simulated trades with wallet, PnL, entry/exit prices, and fees |

Set `USE_SAVED_FEATURED = True` or `USE_SAVED_PREDICTIONS = True` at the top of the script to skip re-computation on repeated runs.

#### Data breakdown

For `--days 180 --timeframe 5m` the 180-day window is used as follows:

| Phase | Rows | Calendar time |
|---|---|---|
| Raw fetch from Binance | ~51,840 | 180 days |
| Feature engineering warmup (dropped) | ~68 | ~6 hours |
| TacticalML training window (first predictions skipped) | 600 | 50 hours |
| Walk-forward training — 80% of predictions, not traded | ~40,938 | ~142 days |
| **Simulation / backtest — 20% of predictions, traded** | **~10,234** | **~36 days** |

The backtest trades only the last ~36 days. The preceding ~144 days exist so TacticalML has enough history to produce well-calibrated predictions before the test period begins — the model never sees the test period during training.

To read the results from the terminal:

```bash
python -c "
import pandas as pd
df = pd.read_csv('labeleddata/dual_BTCUSDT_5m_25d_final_test_sim.csv', index_col=0)
trades = (df['wallet'].diff().abs() > 0).sum()
start, end = df['wallet'].iloc[0], df['wallet'].iloc[-1]
print(f'Trades: {trades}   Wallet: {start:.5f} → {end:.5f}   PnL: {(end-1)*100:+.3f}%')
"
```

---

### Recommended workflow for 25 days

Run these three commands in order. Each step caches its output so re-runs are fast.

```bash
# 1. Train the strategic model with simulation-driven parameter optimisation
python main.py --train-strategic --optimize-params --strategic-days 25 --tactical-days 25

# 2. Run the full dual-ML backtest
python dualmlsimulation.py --symbol BTCUSDT --days 25 --timeframe 5m

# 3. Start the live bot (uses the model trained in step 1)
python main.py
```

---

### Schedule periodic retraining with cron

The strategic model should be retrained daily to stay current with market conditions. To set this up:

1. Find your conda environment path:
```bash
conda env list
```

2. Open crontab:
```bash
crontab -e
```

3. Add this line (replace `<user>` and `<repo_path>`):
```bash
0 2 * * * /bin/bash -i -c "source /Users/<user>/miniconda3/etc/profile.d/conda.sh && conda activate tradingbot && python /Users/<repo_path>/strategic/strategictraining.py >> /Users/<repo_path>/strategictraining.log 2>&1"
```

This runs rule-based training at 2:00 AM daily. The bot picks up the new model automatically on the next prediction cycle. To use simulation-driven training in cron, replace the script call with:

```bash
python /Users/<repo_path>/main.py --train-strategic --optimize-params
```

---

## Running 24/7 on Mac Mini

To prevent the machine from sleeping:

```bash
keepawake/keepawake.sh
```

To cancel:

```bash
keepawake/cancelkeepawake.sh
```

---

## Fan Control (CPU Cooling)

During simulation-driven training the CPU runs at 100% for extended
periods.  The `fancontrol/` module turns a GPIO-controlled fan on
before training and off after — even if the process crashes.

**Board-agnostic**: auto-detects the GPIO interface (`gpioset`,
`raspi-gpio`, `pinctrl`, sysfs, or Python `libgpiod` bindings) and
works on any Linux SBC.

#### Config file (recommended)

```bash
# Copy the template to the project root (found automatically)
cp fancontrol/fanctl.toml.example fanctl.toml
# edit: chip = "gpiochip3", line = 20   (for Radxa Zero 3W)
python main.py --train-strategic --fan-control
```

No env vars needed — the file is auto-discovered by searching the
project root (alongside ``main.py``), then the current directory.

#### Radxa Zero 3W — full setup & heavy training

**Step 1 — One-time GPIO setup**

```bash
# Verify the GPIO chip and line (gpiochip3 line 20 = physical pin 7)
sudo gpioset -c gpiochip3 20=1   # fan should spin up
sudo gpioset -c gpiochip3 20=0   # fan stops

# Configure — copy template and edit chip/line
cp fancontrol/fanctl.toml.example fanctl.toml
# edit fanctl.toml: chip = "gpiochip3", line = 20
```

**Step 2 — Passwordless GPIO (recommended, one-time)**

```bash
sudo groupadd gpio 2>/dev/null
sudo usermod -aG gpio $USER
echo 'SUBSYSTEM=="gpio", KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"' \
  | sudo tee /etc/udev/rules.d/99-gpio.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
echo "Log out and back in for group changes to take effect."
```

After re-login, test without sudo:

```bash
python -c "
from fancontrol.fanctl import FanController
c = FanController()
c.on(); print('Fan on:', c.is_on)
c.off(); print('Fan off:', not c.is_on)
"
```

**Step 3 — Launch heavy training in a tmux session**

Simulation-driven training (`--optimize-params`) runs the CPU at 100%
for hours.  Use tmux so the process survives SSH disconnects:

```bash
# Install tmux if not present
sudo apt install tmux

# Create a session named "tradingbot" and start training
tmux new-session -s tradingbot
```

Inside the tmux session:

```bash
cd /home/armbian/trading-bot
conda activate tradingbot
python main.py --train-strategic --optimize-params --strategic-days 365 --tactical-days 45 --fan-control
```

Detach with **Ctrl+B D**, re-attach anytime with:

```bash
tmux attach -t tradingbot
```

Quick test (short run with low threshold to verify everything works):

```bash
FAN_TEMP_THRESHOLD=30 python main.py --train-strategic --fan-control
```

For other boards (Raspberry Pi, Orange Pi, Jetson, etc.) see the full
documentation and board reference table:

➡️ **[`fancontrol/README.md`](fancontrol/README.md)**

---

## Project structure

```
main.py                        Entry point and CLI
basestrategy.py                Abstract strategy loop (initialize / on_trading_iteration)
dualmlstrategy.py              Two-tier ML strategy (default)
mlstrategy.py                  Legacy single-ML strategy

tactical/
  tacticalml.py                Ephemeral 5m predictor, retrained every candle

strategic/
  strategicml.py               Persisted strategic model with hot-swap
  strategicfeatures.py         Multi-timeframe feature engineering for strategic model
  strategictraining.py         CLI training script (rule-based labels)

positionmanager.py             Position state, scaling, partial close, veto logic
riskguard.py                   Daily loss / drawdown / leverage circuit breaker

dualmlsimulation.py            Walk-forward backtest of the full two-tier system

mltrainingcore.py              Shared feature engineering, label generation, simulation
mltraining.py                  Walk-forward param optimisation (used by strategictraining)
mlio.py                        Model and data I/O utilities
timeframe_config.py            Timeframe presets (5m / 15m / 1h / 4h)

fancontrol/                    Board-agnostic GPIO fan control for CPU cooling
  fanctl.py                    Context manager, signal handlers, temp monitoring
  config.py                    TOML + env var configuration
  backends/                    5 GPIO backends (auto-detected)
  fanctl.toml.example          Board reference table + config template

binancebasebroker.py           Abstract broker interface
binancefuturesbroker.py        Binance Futures implementation
binancespotbroker.py           Binance Spot implementation
binancebrokerfactory.py        Broker factory

model/                         Trained strategic model (one file, timestamped)
labeleddata/                   Cached feature datasets and simulation outputs
tests/                         Unit test suite (99 tests)
```
