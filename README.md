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
Trained offline on 6–12 months of historical data and saved to `model/`. Loaded at startup and hot-swapped at runtime when a new model file appears. Controls leverage, position sizing, stop-loss, take-profit, and max hold time. Acts as a gatekeeper: if it detects extreme volatility or a choppy regime, no trades are opened regardless of what TacticalML signals.

**PositionManager**
Sits between the two models and the broker. Tracks open positions and implements:
- *Consecutive signal scaling*: two consecutive signals in the same direction increase the position size
- *Gradual close*: each signal in the opposite direction closes a fraction of the position
- *Veto enforcement*: immediately closes any open position when StrategicML blocks trading

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
| `--model-type` | `cat` | `cat` | ML model backend |

### Train the strategic model

The strategic model must be trained before the first run. Training fetches historical data from Binance and saves the model to `model/`.

```bash
python main.py --train-strategic
```

Or directly via the training script:

```bash
python strategic/strategictraining.py --symbol BTCUSDT --days 365 --timeframe 1h
```

Available flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--symbol` | `BTCUSDT` | Trading pair |
| `--days` | `365` | Days of historical data to train on |
| `--timeframe` | `1h` | Candle interval for strategic features |
| `--model-dir` | `model/` | Where to save the trained model |

The trained model is saved with a UTC timestamp in the filename (e.g. `strategic_meta_model_20260101T020000Z.pkl`). The running bot detects the new file and hot-swaps it automatically — no restart needed.

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

This runs training at 2:00 AM daily. The bot picks up the new model automatically on the next prediction cycle.

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
  strategictraining.py         CLI training script

positionmanager.py             Position state, scaling, partial close, veto logic

mltrainingcore.py              Shared feature engineering, label generation, simulation
mlio.py                        Model and data I/O utilities
timeframe_config.py            Timeframe presets (5m / 15m / 1h / 4h)

binancebasebroker.py           Abstract broker interface
binancefuturesbroker.py        Binance Futures implementation
binancespotbroker.py           Binance Spot implementation
binancebrokerfactory.py        Broker factory

model/                         Trained strategic model (one file, timestamped)
```
