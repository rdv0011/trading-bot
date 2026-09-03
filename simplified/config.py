"""
Configuration for Dual-ML Bitcoin Trading Bot.
KEEPING dual-ML architecture: Tactical (15m) + Strategic (1h).
"""

# ── Data ───────────────────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"          # Tactical timeframe
TACTICAL_TF = TIMEFRAME     # Alias for compatibility
STRATEGIC_TF = "1h"        # Strategic timeframe
HISTORY_DAYS = 50          # ~4800 candles 15m, ~1200 candles 1h (live Binance 90d+)
TRAIN_FRACTION = 0.8       # Chronological split: 80% train, 20% validation

# ── Feature Engineering ────────────────────────────────────────────────
LABEL_HORIZON = 4          # Predict 4 candles ahead = 1 hour ahead
FEATURE_LAGS = [1, 2, 3, 5, 10, 20, 50]  # Return lags
EMA_SPANS = [5, 10, 20, 50, 100]          # EMAs for trend detection
ATR_PERIOD = 14                                          # Average True Range

# ── Model Parameters ───────────────────────────────────────────────────
# Tactical ML (15m, retrained every candle)
TACTICAL_MODEL_PARAMS = {
    "iterations": 100,      # Fast for walk-forward
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "verbose": False,
}

# Strategic ML (1h, persisted model) - multi-output trade-parameter model
STRATEGIC_MODEL_PARAMS = {
    "iterations": 500,      # More thorough training
    "depth": 8,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "verbose": False,
}

# Strategic targets predicted by the multi-output strategic model
STRATEGIC_TARGET_COLS = [
    "recommended_leverage",
    "max_exposure_frac",
    "stake_long_frac",
    "stake_short_frac",
    "stop_loss_frac",
    "take_profit_frac",
    "max_hold_hours",
]

# Heuristic map from regime -> recommended leverage (matches legacy)
REGIME_LEVERAGE = {"trend": 5.0, "high_vol": 2.0, "chop": 1.0}
# Stake fractions by regime
REGIME_STAKE_LONG = {"trend": 0.2, "high_vol": 0.1, "chop": 0.1}
REGIME_STAKE_SHORT = {"trend": 0.1, "high_vol": 0.05, "chop": 0.05}
# Stop loss / take profit by regime
REGIME_STOP_LOSS = {"trend": 0.015, "high_vol": 0.03, "chop": 0.02}
TAKE_PROFIT_MULT = 2.0
# Max hold hours by regime
REGIME_MAX_HOLD = {"trend": 8.0, "high_vol": 2.0, "chop": 4.0}

# ── Trading Parameters ─────────────────────────────────────────────────
# These are predicted by strategic ML, but defaults for simulation/defaults:
STAKE_LONG_FRAC_DEFAULT = 0.10    # 10% of equity for long
STAKE_SHORT_FRAC_DEFAULT = 0.05   # 5% of equity for short
STOP_LOSS_FRAC_DEFAULT = 0.02     # 2% stop loss
TAKE_PROFIT_FRAC_DEFAULT = 0.04   # 4% take profit
MAX_HOLD_HOURS_DEFAULT = 4.0      # Maximum hold time in hours
LEVERAGE_DEFAULT = 1.0            # Default leverage (1x = spot)

# ── Simulation / Backtest ──────────────────────────────────────────────
INITIAL_EQUITY = 1.0          # Start with 1.0 (100%)
FEE = 0.0004                  # Binance taker fee per side
SLIPPAGE = 0.0003             # Fixed slippage per trade

# Walk-forward settings (simulation retrains tactical every N candles)
WALKFORWARD_RETRAIN_EVERY = 100  # Candles between tactical model retraining

# ── Signal Thresholds ──────────────────────────────────────────────────
# Tactical signal thresholds (absolute, since no adaptive threshold in minimal version)
ABSOLUTE_THRESHOLD = 0.003      # Minimum prediction to trigger LONG/SHORT

# ── Logging ────────────────────────────────────────────────────────────
LOG_DIR = "logs"                         # Daily log files go here
TRADE_LOG_CSV = "logs/trades_demo.csv"   # Trade details CSV for comparison
CONSOLE_LOG_LEVEL = "INFO"               # Brief info to console
FILE_LOG_LEVEL = "DEBUG"                 # Detailed to file
MODEL_DIR = "models"                     # Model directory

# ── Comparison ─────────────────────────────────────────────────────────
# These settings control the sim vs demo comparison
COMPARE_SIM_DEMO = True                # Enable sim vs demo log comparison
DEMO_LOG_PATTERN = "logs/trading_*.log"  # Pattern for demo log files