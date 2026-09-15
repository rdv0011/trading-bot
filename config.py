"""
Configuration for Dual-ML Bitcoin Trading Bot.
KEEPING dual-ML architecture: Tactical (15m) + Strategic (1h).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load broker credentials from .env in repo root
_REPO_ROOT = Path(__file__).parent
load_dotenv(_REPO_ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    """Get environment variable with optional default."""
    return os.getenv(key, default)


def _get_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable with optional default."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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

# Walk-forward settings.
# LIVE cadence: the live loop retrains tactical every N live ITERATIONS
# (~4s each, so 100 iterations ~= 6-10 min at observed pacing), on the recent
# `_tactical_window` (200) candles folded to ~145 usable after labeling.
# SIM cadence: rolling_tactical_predict counts CANDLES, not iterations. 100
# candles = 25h of 15m bars, far older than the live model -- the stale sim
# model compresses predictions to ~0 and never crosses the threshold, while
# the live model retrained every ~10 min DOES signal (observed: live saw ~28
# entry episodes over 2 days, default sim saw 0). The sim therefore must
# retrain far more often than 100 candles to mirror live's behavior.
WALKFORWARD_RETRAIN_EVERY = 100    # how often LIVE retrains, in live iterations
WALKFORWARD_RETRAIN_EVERY_CANDLES = 1   # how often SIM retrains, in 15m candles (mirrors live)

# Live model hot-swap: reload tactical/strategic models from disk this often
# (in iterations) so newly trained models are picked up without a restart.
MODEL_REFRESH_EVERY = 50

# ── Performance (Phase 1) ──────────────────────────────────────────────
FEATURE_CACHE_TTL = 60                # feature-frame cache TTL (seconds)
KLINE_CACHE_TTL_LIVE = 60             # max klines-cache TTL in live mode (seconds)
PARALLEL_FETCH_WORKERS = 3            # ThreadPoolExecutor workers for parallel API fetch

# ── Adaptive Threshold (Phase 2) ───────────────────────────────────────
ADAPTIVE_THRESHOLD_ENABLED = _get_bool("ADAPTIVE_THRESHOLD_ENABLED", True)
ADAPTIVE_LOOKBACK = 200               # rolling window of recent predictions
ADAPTIVE_QUANTILE = 0.95              # threshold = this percentile of |preds|
ADAPTIVE_MIN_THRESHOLD = 0.002        # floor so threshold never goes too low
ADAPTIVE_MAX_THRESHOLD = 0.02         # ceiling so threshold never goes too high

# ── Trailing Stop (Phase 2) ────────────────────────────────────────────
TRAILING_STOP_ENABLED = _get_bool("TRAILING_STOP_ENABLED", True)
TRAILING_ATR_MULT = 1.5               # trail distance = mult * ATR14
TRAILING_BREAKEVEN_MULT = 1.0         # move SL to breakeven after 1x initial risk

# ── Regime Gate Hardening (Phase 2) ────────────────────────────────────
GATE_EXTREME_VOL = _get_bool("GATE_EXTREME_VOL", True)               # block entries when vol_state = extreme
EXTREME_VOL_RATIO = 1.8               # vol_12 / vol_48 above this = extreme

# ── Liquidity Recorder + Gates (Phase 1-2, Option B) ───────────────────
LIQUIDITY_RECORDER_ENABLED = _get_bool("LIQUIDITY_RECORDER_ENABLED", True)
LIQUIDITY_STALE_AFTER_S = 15.0        # WS stream older than this = stale
LIQUIDITY_DEPTH_SPAN_BPS = 50.0       # depth measured within +-50 bps of mid
LIQUIDITY_WINDOW_SEC = 60.0           # aggTrade tape rolling window (s)
LIQUIDITY_TOP_N_VANISH = 25           # top-N levels compared for vanish ratio
LIQUIDITY_VANISH_INTERVAL = 10.0      # re-marker interval for vanish calc (s)

GATE_LIQUIDITY = _get_bool("GATE_LIQUIDITY", True)        # block entries on thin book
LIQ_MAX_SPREAD_BPS = 10.0             # entry blocked when spread wider
LIQ_MIN_DEPTH_USD = 50000.0           # entry blocked when side depth below
LIQ_MAX_VANISH_PCT = 60.0             # entry blocked when >60% of levels vanish
LIQ_MIN_AGGRESSOR_RATIO = 0.35        # taker-buy share below this = sell-dominated
LIQ_MAX_AGGRESSOR_RATIO = 0.65        # taker-buy share above this = buy-dominated
LIQ_MIN_TRADE_INTENSITY_USD = 50000.0 # min notional traded per second
GATE_LIQUIDITY_EXIT = _get_bool("GATE_LIQUIDITY_EXIT", True)  # liq_exit on thinning book
LIQ_EXIT_SPREAD_BPS = 20.0            # exit when spread widens beyond this
LIQ_EXIT_MIN_DEPTH_USD = 25000.0      # exit when total depth collapses below this
LIQUIDITY_FAIL_OPEN_ON_STALE = _get_bool("LIQUIDITY_FAIL_OPEN_ON_STALE", True)

# ── Position Scaling (Phase 3) ─────────────────────────────────────────
SCALING_ENABLED = _get_bool("SCALING_ENABLED", True)
MAX_SCALE_COUNT = 2                   # scale-ins allowed after initial entry
SCALE_CONFIRM_BARS = 2                # same-direction bars to confirm a scale-up
SCALE_STAKE_FRAC = 0.5                # scale-in stake = fraction of current position

# ── Partial Exit (Phase 3) ─────────────────────────────────────────────
PARTIAL_EXIT_ENABLED = _get_bool("PARTIAL_EXIT_ENABLED", True)
PARTIAL_EXIT_FRACTION = 0.33          # fraction closed on first reversal signal
REVERSAL_FULL_CLOSE_STREAK = 2        # reversal signals before full close

# ── Trade Cooldown (Phase 3) ───────────────────────────────────────────
TRADE_COOLDOWN_ENABLED = _get_bool("TRADE_COOLDOWN_ENABLED", True)
TRADE_COOLDOWN_MINUTES = 30           # minimum gap between new entries

# ── Startup Position Adoption ──────────────────────────────────────────
# Adopt an exchange position left open by a previous process (crash / killed
# session) so exits manage it instead of trading blindly beside it. When
# False, startup is refused if an open position exists.
ADOPT_EXISTING_POSITION = _get_bool("ADOPT_EXISTING_POSITION", True)

# ── Signal Thresholds ──────────────────────────────────────────────────
# Tactical signal threshold (absolute; used when ADAPTIVE_THRESHOLD_ENABLED=False)
ABSOLUTE_THRESHOLD = 0.006      # Minimum prediction to trigger LONG/SHORT

# ── Logging ────────────────────────────────────────────────────────────
LOG_DIR = "logs"                         # Daily log files go here
TRADE_LOG_CSV = "logs/trades_demo.csv"   # Trade details CSV for comparison
CONSOLE_LOG_LEVEL = "INFO"               # Brief info to console
FILE_LOG_LEVEL = "DEBUG"                 # Detailed to file
MODEL_DIR = "models"                     # Model directory
DECISION_LOG_INTERVAL = 300              # live-loop state summary every N iterations

# ── Comparison ─────────────────────────────────────────────────────────
# These settings control the sim vs demo comparison
COMPARE_SIM_DEMO = True                # Enable sim vs demo log comparison
DEMO_LOG_PATTERN = "logs/trading_*.log"  # Pattern for demo log files


# ── Trading Environment (Phase 0.5) ────────────────────────────────────
# Controls where the bot executes: "testnet" (debug/simulation) or "mainnet"
# (production, real funds). Switch via .env: TRADING_ENV=mainnet
#
# Rationale: the bot is developed/tuned against testnet where issues can be
# reproduced and fixed without burning real money. When a mainnet incident
# occurs, set TRADING_ENV=testnet to debug the same code path on testnet
# before going live again.
TRADING_ENV = _get("TRADING_ENV", "testnet").strip().lower()
assert TRADING_ENV in ("testnet", "mainnet"), \
    f"TRADING_ENV must be 'testnet' or 'mainnet', got: {TRADING_ENV!r}"


def is_testnet_env() -> bool:
    """True when the bot should trade on Binance testnet (debug mode)."""
    return TRADING_ENV == "testnet"


# ── Binance API Credentials (.env) ─────────────────────────────────────
BINANCE_TESTNET_FUTURES_API_KEY: str = _get("BINANCE_TESTNET_FUTURES_API_KEY")
BINANCE_TESTNET_FUTURES_API_SECRET: str = _get("BINANCE_TESTNET_FUTURES_API_SECRET")
BINANCE_TESTNET_SPOT_API_KEY: str = _get("BINANCE_TESTNET_SPOT_API_KEY")
BINANCE_TESTNET_SPOT_API_SECRET: str = _get("BINANCE_TESTNET_SPOT_API_SECRET")
BINANCE_LIVE_FUTURES_API_KEY: str = _get("BINANCE_LIVE_FUTURES_API_KEY")
BINANCE_LIVE_FUTURES_API_SECRET: str = _get("BINANCE_LIVE_FUTURES_API_SECRET")
BINANCE_LIVE_SPOT_API_KEY: str = _get("BINANCE_LIVE_SPOT_API_KEY")
BINANCE_LIVE_SPOT_API_SECRET: str = _get("BINANCE_LIVE_SPOT_API_SECRET")


def validate_credentials(market_type: str = "futures", testnet: bool = True) -> tuple[str, str]:
    """
    Validate and return API credentials for the given market type and testnet setting.

    Raises:
        ValueError: If required credentials are not set.
    """
    if testnet:
        if market_type == "futures":
            api_key = BINANCE_TESTNET_FUTURES_API_KEY
            api_secret = BINANCE_TESTNET_FUTURES_API_SECRET
        else:
            api_key = BINANCE_TESTNET_SPOT_API_KEY
            api_secret = BINANCE_TESTNET_SPOT_API_SECRET
    else:
        if market_type == "futures":
            api_key = BINANCE_LIVE_FUTURES_API_KEY
            api_secret = BINANCE_LIVE_FUTURES_API_SECRET
        else:
            api_key = BINANCE_LIVE_SPOT_API_KEY
            api_secret = BINANCE_LIVE_SPOT_API_SECRET

    if not api_key or not api_secret:
        env_prefix = "TESTNET" if testnet else "LIVE"
        raise ValueError(
            f"BINANCE_{env_prefix}_{market_type.upper()}_API_KEY and "
            f"BINANCE_{env_prefix}_{market_type.upper()}_API_SECRET must be set in .env"
        )

    return api_key, api_secret


def get_broker_config(market_type: str = "futures", testnet: bool = True) -> dict:
    """Get broker configuration dict for the given market type."""
    api_key, api_secret = validate_credentials(market_type, testnet)
    return {
        "api_key": api_key,
        "api_secret": api_secret,
        "market_type": market_type,
        "testnet": testnet,
    }