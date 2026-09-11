# Trading Bot Simplification Plan (REVISED - Dual-ML + Sim vs Demo Comparison)

## Current System Analysis

The current codebase is a **dual-ML architecture** with:
- **Tactical ML** (15m): Ephemeral CatBoost retrained every candle, produces LONG/SHORT/HOLD signals via adaptive thresholding
- **Strategic ML** (1h): Persisted CatBoost predicting meta-parameters (stake sizes, SL, TP, max hold, leverage, regime)
- Complex training pipeline with walk-forward optimization, regime detection, meta-parameter prediction
- Live trading with Binance testnet, bracket orders, position management
- Simulation with dynamic slippage, latency, multiple gates (volume, HTF EMA, RiskGuard)
- Extensive logging to daily rotating files + console

**File count**: ~25 Python files across multiple modules (strategic/, tactical/, mlio/, etc.)

---

## Minimal System Requirements (User-Defined)

| Requirement | Status |
|-------------|--------|
| Trade Bitcoin (BTCUSDT) | ✅ Essential |
| ML models trained on historical data | ✅ Essential |
| Train/validation split | ✅ Essential |
| Simulation with mocked broker on validation data | ✅ Essential |
| Console logging (brief) + detailed trade logs to `logs/` | ✅ Essential |
| **KEEP dual-ML architecture (tactical + strategic)** | ✅ **MANDATORY** |
| **Compare simulation vs real demo trading from logs** | ✅ **MANDATORY** |
| Everything else | ❌ Optional - REMOVE |

---

## Simplification Strategy: **Minimal Dual-ML Architecture**

**Keep dual-ML but strip to bare essentials:**
- Tactical ML (15m): Walk-forward retraining on validation data, produces signals
- Strategic ML (1h): Persisted model predicting meta-parameters (stake, SL, TP, max_hold, leverage)
- Single training pipeline per model type
- Simple simulation with MockBroker
- **NEW**: Comparison module to compare sim trades vs demo trades from logs

---

## Minimal File Structure (Target: ~10-12 files)

```
trading-bot/
├── main.py                    # Entry point: train | simulate | live | compare
├── config.py                  # All hyperparameters in one place
├── data.py                    # Data download, features, labels, train/val split
├── model.py                   # CatBoost training/prediction for BOTH models
├── simulate.py                # Backtest on validation data with MockBroker
├── broker.py                  # Binance broker (live) + MockBroker (simulation)
├── strategy.py                # Dual-ML strategy logic (tactical + strategic)
├── logging.py                 # Console + file + trade CSV logging
├── compare.py                 # NEW: Sim vs Demo comparison from logs
└── utils.py                   # Shared utilities (metrics, helpers)
```

**Files to DELETE** (15+ files):
- `mltraining.py`, `mltrainingcore.py`, `mlpredictor.py`, `mlstrategy.py`, `dualmlstrategy.py`, `dualmlsimulation.py`
- `basestrategy.py`, `binancebrokerfactory.py`, `binancebasebroker.py`, `binancefuturesbroker.py`, `binancespotbroker.py`
- `positionmanager.py`, `riskguard.py`, `timeframe_config.py`, `mlio.py`
- `compare_demo_vs_sim.py`, `auto_compare_demo_vs_sim.py`, `demo_log_parser.py`, `displayresults.py`
- Strategic/ tactical/ directories entirely (logic moved to model.py + strategy.py)

---

## Component Specifications

### 1. `config.py` - Single Source of Truth
```python
# Data
SYMBOL = "BTCUSDT"
TACTICAL_TF = "15m"
STRATEGIC_TF = "1h"
HISTORY_DAYS = 90           # ~8640 candles 15m, ~360 candles 1h
TRAIN_FRACTION = 0.8        # Chronological split

# Features (shared)
FEATURE_LAGS = [1, 2, 3, 5, 10, 20, 50]
EMA_SPANS = [5, 10, 20, 50, 100]
ATR_PERIOD = 14
LABEL_HORIZON = 4           # Predict 4 candles ahead (1h for 15m tf)

# Tactical Model (15m, retrained every N candles)
TACTICAL_MODEL_PARAMS = {
    "iterations": 100,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "verbose": False,
}
WALKFORWARD_RETRAIN_EVERY = 100  # Candles between retraining

# Strategic Model (1h, persisted)
STRATEGIC_MODEL_PARAMS = {
    "iterations": 300,
    "depth": 8,
    "learning_rate": 0.03,
    "loss_function": "RMSE",
    "verbose": False,
}

# Trading (defaults, overridden by strategic ML predictions)
STAKE_LONG_FRAC_DEFAULT = 0.10
STAKE_SHORT_FRAC_DEFAULT = 0.05
STOP_LOSS_FRAC_DEFAULT = 0.02
TAKE_PROFIT_FRAC_DEFAULT = 0.04
MAX_HOLD_HOURS_DEFAULT = 4.0
LEVERAGE_DEFAULT = 1.0

# Signal Thresholds
ABSOLUTE_THRESHOLD = 0.003   # Minimum prediction for LONG/SHORT

# Simulation
INITIAL_EQUITY = 1.0
FEE = 0.0004
SLIPPAGE = 0.0003

# Logging
LOG_DIR = "logs"
TRADE_LOG_CSV = "logs/trades_demo.csv"
CONSOLE_LOG_LEVEL = "INFO"
FILE_LOG_LEVEL = "DEBUG"

# Comparison
COMPARE_SIM_DEMO = True
DEMO_LOG_PATTERN = "logs/trading_*.log"
```

### 2. `data.py` - Data Pipeline
```python
def download_data(symbol, timeframe, days) -> pd.DataFrame
def make_features(df, timeframe) -> pd.DataFrame    # Returns, EMAs, ATR, vol, time, regime
def make_labels(df, horizon) -> pd.DataFrame        # future_ret column
def train_val_split(df, train_frac=0.8) -> (train, val)  # Chronological
def get_feature_cols(df) -> List[str]               # Exclude targets, regime
def run_full_pipeline() -> (df_train, df_val)       # End-to-end
```

### 3. `model.py` - Dual-ML Model Pipeline
```python
class CatBoostModel:
    def __init__(self, model_type: str, params: dict):  # "tactical" or "strategic"
    def train(self, df, feature_cols, target_col) -> self
    def save(self, path_prefix) -> (model_path, meta_path)
    def load(self, model_path, meta_path) -> self
    def predict(self, df, feature_cols) -> pd.Series

# Tactical: walk-forward on validation set
def rolling_tactical_predict(df_val, model, feature_cols, tf_cfg, retrain_every) -> pd.Series

# Strategic: batch prediction for meta-params
def strategic_batch_predict(df_val, model, feature_cols) -> pd.Series
def predict_strategic_meta_params(df_val, model, feature_cols) -> List[dict]
```

### 4. `simulate.py` - Validation Backtest with MockBroker
```python
class MockBroker:
    def __init__(self, df_val, fee, slippage, initial_equity=1.0):
        self.df = df_val
        self.fee = fee
        self.slippage = slippage
        self.equity = initial_equity
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self.trades = []  # List of trade dicts

    def step(self, idx, signal, meta_params: dict):
        # signal: "long" | "short" | "hold"
        # meta_params: {stake_long, stake_short, sl, tp, max_hold, leverage, regime}
        # Execute at df.iloc[idx]['close'] with fee + slippage
        # Track position, equity, trades
        pass

def run_simulation(
    df_val: pd.DataFrame,
    tactical_preds: pd.Series,
    strategic_meta_params: List[dict],
    config
) -> (trades_df, metrics_dict, equity_curve):
    # Iterate validation candles
    # At each step: tactical pred → signal → strategic meta → mock broker
    # Return: trades DataFrame, metrics dict, equity curve Series
```

### 5. `broker.py` - Broker Interface
```python
class BinanceBroker:
    # Minimal live: get_historical, get_price, open_bracket, cancel, close
    pass

class MockBroker:  # Re-exported from simulate.py
    pass
```

### 6. `strategy.py` - Dual-ML Strategy Logic
```python
def prediction_to_signal(pred, threshold=0.003):
    if pred > threshold: return "long"
    elif pred < -threshold: return "short"
    else: return "hold"

class DualMLStrategy:
    def __init__(self, broker, config):
        self.broker = broker
        self.config = config

    def run_simulation_step(self, idx, tactical_pred, strategic_params):
        # Convert tactical pred to signal
        # Get meta-params from strategic
        # Check exits (SL, TP, max_hold, signal reversal)
        # Execute via broker
        pass

    def run_live_loop(self, tactical_model, strategic_model, feature_cols):
        # Live trading loop: fetch data → features → predict → execute
        pass
```

### 7. `logging.py` - Unified Logging
```python
def setup_logging():
    # Console: INFO, brief
    # File: DEBUG, logs/trading_YYYY-MM-DD.log (daily rotation)
    # Trade CSV: logs/trades_YYYY-MM-DD.csv (entry, exit, pnl, regime, meta_params)
    pass

def log_trade(trade_dict):  # Append to daily CSV
    pass

def log_info(msg):  # Console + file
    pass

def log_debug(msg):  # File only
    pass
```

### 8. `compare.py` - **NEW: Sim vs Demo Comparison**
```python
def parse_demo_logs(log_dir: str, pattern: str = "trading_*.log") -> pd.DataFrame:
    """
    Parse demo trading logs into trades DataFrame.
    Extracts: timestamp, symbol, side, entry_price, exit_price, qty, pnl, regime, meta_params
    """
    pass

def parse_sim_logs(log_dir: str, pattern: str = "trades_sim_*.csv") -> pd.DataFrame:
    """Parse simulation trade CSV logs."""
    pass

def align_trades(demo_trades: pd.DataFrame, sim_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Align trades by timestamp/symbol for comparison.
    Returns DataFrame with both demo and sim metrics per trade.
    """
    pass

def compare_metrics(demo_trades, sim_trades) -> dict:
    """
    Compare aggregate metrics:
    - Total return, Sharpe, Max DD, Win rate, # trades
    - Per-regime breakdown
    - Trade-by-trade diff (entry/exit price, PnL)
    """
    pass

def generate_comparison_report(demo_trades, sim_trades, output_path: str):
    """
    Generate HTML/text report with:
    - Summary metrics table
    - Equity curves overlay
    - Trade-by-trade comparison
    - Regime breakdown
    - Slippage/fee analysis
    """
    pass

def run_comparison(config):
    """Main entry point for comparison mode."""
    demo = parse_demo_logs(config.LOG_DIR, config.DEMO_LOG_PATTERN)
    sim = parse_sim_logs(config.LOG_DIR, "trades_sim_*.csv")
    aligned = align_trades(demo, sim)
    metrics = compare_metrics(demo, sim)
    generate_comparison_report(demo, sim, "logs/comparison_report.html")
    return metrics
```

### 9. `utils.py` - Shared Utilities
```python
def calculate_metrics(trades_df, equity_curve) -> dict:
    # Total return, Sharpe, Max DD, Win rate, Profit factor, etc.
    pass

def save_trades_csv(trades, path):
    pass

def load_trades_csv(path) -> pd.DataFrame:
    pass
```

### 10. `main.py` - CLI Entry Point
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "simulate", "live", "compare"])
    parser.add_argument("--tactical-model", default="models/tactical_model.cbm")
    parser.add_argument("--strategic-model", default="models/strategic_model.cbm")
    args = parser.parse_args()

    if args.mode == "train":
        # Download data for both timeframes
        # Build features, labels
        # Train tactical model on train set
        # Train strategic model on best params from walk-forward
        # Save both models
        pass

    elif args.mode == "simulate":
        # Load validation data
        # Load both models
        # Run walk-forward tactical predictions
        # Run strategic meta-param predictions
        # Run simulation with MockBroker
        # Save trades CSV + metrics
        pass

    elif args.mode == "live":
        # Load both models
        # Run live loop with BinanceBroker
        pass

    elif args.mode == "compare":
        # Run comparison between sim and demo logs
        pass
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Files: config.py, logging.py, data.py)
- [ ] Create `config.py` with all hyperparameters for dual-ML
- [ ] Create `logging.py` with console + file + trade CSV logging
- [ ] Create `data.py` with download, features, labels, split (both timeframes)

### Phase 2: Dual-ML Model Pipeline (Files: model.py)
- [ ] Create `model.py` with CatBoostModel class for both types
- [ ] Implement rolling_tactical_predict (walk-forward)
- [ ] Implement strategic_batch_predict + meta-param decoding

### Phase 3: Simulation (Files: simulate.py, strategy.py)
- [ ] Create `simulate.py` with MockBroker and run_simulation
- [ ] Create `strategy.py` with DualMLStrategy
- [ ] Verify train → simulate pipeline works end-to-end

### Phase 4: Comparison Module (Files: compare.py, utils.py)
- [ ] Create `compare.py` with demo log parser, sim log parser
- [ ] Implement alignment, metrics comparison, report generation
- [ ] Test with existing logs in `logs/`

### Phase 5: Live Trading (Files: broker.py, main.py)
- [ ] Create `broker.py` with BinanceBroker (minimal)
- [ ] Create `main.py` CLI with 4 modes: train/simulate/live/compare

### Phase 6: Cleanup & Validation
- [ ] Delete all obsolete files/directories
- [ ] Run full pipeline: train → simulate → compare → verify
- [ ] Test live mode on testnet (optional)

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Keep dual-ML | User requirement; tactical=signals, strategic=meta-params |
| Walk-forward tactical retraining | Matches live behavior; avoids lookahead |
| Strategic model predicts meta-params | Keeps adaptive sizing/SL/TP without complex pipeline |
| Fixed tactical threshold | Simplicity; strategic ML handles regime adaptation |
| MockBroker in simulation | No external deps; deterministic backtest |
| Daily log rotation + trade CSV | Meets logging requirements exactly |
| Comparison module reads logs | Works with existing demo logs; no live connection needed |

---

## Comparison Module Design (NEW)

### Input Sources
- **Demo logs**: `logs/trading_YYYY-MM-DD.log` (existing format from live trading)
- **Sim logs**: `logs/trades_sim_YYYY-MM-DD.csv` (generated by simulation)

### Output
- Console summary: key metric differences
- HTML report: `logs/comparison_report.html` with charts
- CSV export: `logs/comparison_trades_YYYY-MM-DD.csv` (aligned trade pairs)

### Metrics Compared
| Metric | Demo | Sim | Diff |
|--------|------|-----|------|
| Total Return | X% | Y% | Z% |
| Sharpe Ratio | A | B | C |
| Max Drawdown | D% | E% | F% |
| Win Rate | G% | H% | I% |
| # Trades | J | K | L |
| Avg Slippage | M | N | O |

### Per-Regime Breakdown
- Trend / Chop / High_vol regime comparison
- Trade count and PnL per regime

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Dual-ML adds complexity vs single model | Acceptable per user requirement; modular design isolates each model |
| Walk-forward too slow | Configurable retrain frequency; can use smaller window for testing |
| Comparison needs matching timestamps | Align by nearest timestamp; tolerate small offsets |
| Live broker API changes | Minimal broker interface, easy to adapt |
| Data leakage | Strict chronological split, no future data in features |

---

## Success Criteria

1. **Train**: `python main.py train --tactical-model models/tactical.cbm --strategic-model models/strategic.cbm` completes in <10 min
2. **Simulate**: `python main.py simulate --tactical-model models/tactical.cbm --strategic-model models/strategic.cbm` runs on validation set, outputs metrics + trade CSV
3. **Compare**: `python main.py compare` reads demo logs + sim logs, outputs comparison report
4. **Live**: `python main.py live --tactical-model models/tactical.cbm --strategic-model models/strategic.cbm` connects to testnet
5. **Logs**: Console shows brief iteration logs; `logs/` contains daily `.log` + trade `.csv` files
6. **Code size**: < 1500 lines total across 10 files (vs current ~5000+ lines)