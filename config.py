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
# Minimum trees a CatBoost model must have to be considered non-degenerate.
# Early stopping can stop at iteration 0 (1 tree) when the validation metric
# is noisy or the dataset is small.  Models below this threshold predict
# ~constant values and never cross the signal threshold.
MIN_TREES = 5

# Tactical ML (15m, retrained every candle)
TACTICAL_MODEL_PARAMS = {
    "iterations": 500,      # Fast for walk-forward
    "depth": 6,
    "learning_rate": 0.03,
    "loss_function": "RMSE",
    "verbose": False,
}

# Tactical model balancing
TACTICAL_BALANCE_SAMPLES = _get_bool("TACTICAL_BALANCE_SAMPLES", True)    # enable sample weight balancing
TACTICAL_MIN_CLASS_WEIGHT = 0.1   # minimum weight for minority class
TACTICAL_MAX_CLASS_WEIGHT = 5.0   # maximum weight for minority class

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

# ── Trading Parameters (broker/market specific) ────────────────────────
# These are used to log and compare actual costs across live/demo/backtest.
# When a broker supports per-symbol rates, the broker overrides these at init.

# Commission / fee per side (fraction of notional)
FEE = 0.0004                  # Binance taker fee per side (0.04%)

# Slippage model (fraction of price)
SLIPPAGE = 0.0003             # Fixed slippage per trade (0.03%)

# Spread (fraction of price) — midpoint-to-worst-fill spread
SPREAD = 0.0001               # Default mid-spread estimate (0.01%)

# Funding rate / swap (per-hour fraction, applied at funding intervals)
# Binance perpetuals fund every 8 hours; this is the expected hourly rate.
FUNDING_RATE = 0.0            # Expected hourly funding rate (fraction)
FUNDING_INTERVAL_HOURS = 8    # Funding payment interval in hours

# ── Simulation / Backtest ──────────────────────────────────────────────
INITIAL_EQUITY = 1.0          # Start with 1.0 (100%)

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
WALKFORWARD_RETRAIN_EVERY = 10    # how often LIVE retrains, in live iterations
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

# ── Adaptive Liquidity Threshold (Phase 2b) ──────────────────────────────
ADAPTIVE_LIQ_ENABLED = _get_bool("ADAPTIVE_LIQ_ENABLED", True)
ADAPTIVE_LIQ_LOOKBACK = 100           # rolling window of recent trade_intensity (reduced for faster adaptation)
ADAPTIVE_LIQ_QUANTILE = 0.10          # use 10th percentile as floor (more conservative, allows more time)
ADAPTIVE_LIQ_MIN_USD = 5000.0         # absolute floor USD/s
ADAPTIVE_LIQ_MAX_USD = 25000.0        # absolute ceiling USD/s (lowered from 50k - market is quieter)
ADAPTIVE_LIQ_COOLDOWN_MIN = 30        # minutes to wait before allowing another LIQ_EXIT (increased from 15)

# ── Adaptive Trailing SL Update Threshold (Phase 2b) ─────────────────────
ADAPTIVE_TRAIL_UPDATE_ENABLED = _get_bool("ADAPTIVE_TRAIL_UPDATE_ENABLED", True)
ADAPTIVE_TRAIL_UPDATE_LOOKBACK = 50   # rolling window of recent ATR values
ADAPTIVE_TRAIL_UPDATE_QUANTILE = 0.5  # median ATR
ADAPTIVE_TRAIL_UPDATE_MULT = 0.1      # fraction of ATR for update threshold (10% of ATR)
ADAPTIVE_TRAIL_UPDATE_MIN = 0.001     # 0.1% floor
ADAPTIVE_TRAIL_UPDATE_MAX = 0.01      # 1% ceiling

# ── Adaptive Extreme Volatility Threshold (Phase 2b) ─────────────────────
ADAPTIVE_VOL_ENABLED = _get_bool("ADAPTIVE_VOL_ENABLED", True)
ADAPTIVE_VOL_LOOKBACK = 200           # rolling window of recent vol_ratio values
ADAPTIVE_VOL_QUANTILE = 0.95          # use 95th percentile as extreme threshold
ADAPTIVE_VOL_MIN = 1.2                # absolute floor
ADAPTIVE_VOL_MAX = 3.0                # absolute ceiling

# ── Adaptive Spread/Depth Thresholds (Phase 2b) ──────────────────────────
ADAPTIVE_BOOK_ENABLED = _get_bool("ADAPTIVE_BOOK_ENABLED", True)
ADAPTIVE_BOOK_LOOKBACK = 200          # rolling window of recent spread/depth values
ADAPTIVE_SPREAD_QUANTILE = 0.75       # 75th percentile as max spread
ADAPTIVE_SPREAD_MIN = 2.0             # absolute floor (bps)
ADAPTIVE_SPREAD_MAX = 20.0            # absolute ceiling (bps)
ADAPTIVE_DEPTH_QUANTILE = 0.25        # 25th percentile as min depth
ADAPTIVE_DEPTH_MIN = 20000.0          # absolute floor (USD)
ADAPTIVE_DEPTH_MAX = 500000.0         # absolute ceiling (USD)

# ── Trailing Stop (Phase 2) ────────────────────────────────────────────
TRAILING_STOP_ENABLED = _get_bool("TRAILING_STOP_ENABLED", True)
TRAILING_BREAKEVEN_MULT = 1.0         # move SL to breakeven after 1x initial risk
TRAILING_SL_UPDATE_THRESHOLD = 0.002  # 0.2% min move to trigger exchange SL/TP update (hybrid) — fallback when adaptive disabled

# ── Regime Gate Hardening (Phase 2) ────────────────────────────────────
GATE_EXTREME_VOL = _get_bool("GATE_EXTREME_VOL", True)               # block entries when vol_state = extreme
EXTREME_VOL_RATIO = 1.8               # vol_12 / vol_48 above this = extreme (fallback when adaptive disabled)
GATE_ABS_VOL_ENABLED = _get_bool("GATE_ABS_VOL_ENABLED", True)       # block entries when absolute volume drops
ABS_VOL_RATIO_CRITICAL = 0.10             # ratio below this = absolute volume drop (10% of median)
ABS_VOL_RATIO_ROLLING_WINDOW = 48          # rolling window for median (candles)

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
LIQ_MIN_TRADE_INTENSITY_USD = 25000.0 # min notional traded per second (entry gate, fallback)
GATE_LIQUIDITY_EXIT = _get_bool("GATE_LIQUIDITY_EXIT", True)  # liq_exit on thinning book
LIQ_EXIT_SPREAD_BPS = 100.0            # exit when spread widens beyond this
LIQ_EXIT_MIN_DEPTH_USD = 500000.0      # exit when total depth collapses below this
LIQ_EXIT_MIN_TRADE_INTENSITY_USD = 25000.0      # exit when trade intensity collapses (fallback when adaptive disabled)
LIQUIDITY_FAIL_OPEN_ON_STALE = _get_bool("LIQUIDITY_FAIL_OPEN_ON_STALE", True)

# ── Cost-to-Edge Optimizations ─────────────────────────────────────────
# Confidence-weighted position sizing: stake scales by |pred|/threshold^power
CONFIDENCE_STAKE_POWER = 0.5                # sqrt scaling (diminishing returns)
CONFIDENCE_STAKE_CAP = 3.0                  # max stake multiplier (3x base)

# Liquidity exit grace period: suppress liq_exit for N minutes after entry
LIQ_EXIT_GRACE_PERIOD_MIN = 10               # minutes to suppress liq_exit after entry

# Spread-aware entry gate: block entries when spread exceeds this threshold
SPREAD_ENTRY_GATE_BPS = 20.0                # bps threshold for entry gate

# ── Position Scaling (Phase 3) ─────────────────────────────────────────
SCALING_ENABLED = _get_bool("SCALING_ENABLED", True)
MAX_SCALE_COUNT = 2                   # scale-ins allowed after initial entry
SCALE_CONFIRM_BARS = 2                # same-direction bars to confirm a scale-up
SCALE_STAKE_FRAC = 0.5                # scale-in stake = fraction of current position

# ── Partial Exit (Phase 3) ─────────────────────────────────────────────
PARTIAL_EXIT_ENABLED = _get_bool("PARTIAL_EXIT_ENABLED", True)
PARTIAL_EXIT_FRACTION = 0.33          # fraction closed on first reversal signal
REVERSAL_FULL_CLOSE_STREAK = 2        # reversal signals before full close

# ── R-Multiple Partial Exits (Profit Taking) ───────────────────────────
R_PARTIAL_EXITS_ENABLED = _get_bool("R_PARTIAL_EXITS_ENABLED", True)   # enable R-multiple profit taking
R_PARTIAL_LEVELS = 3                     # number of R-levels (1R, 2R, 3R)
R_PARTIAL_FRACTIONS = [0.33, 0.33, 0.34]  # fractions at each level (must sum to ~1.0)
R_PARTIAL_THRESHOLD_BUFFER = 0.05        # trigger partial exits at 0.95R, 1.95R, 2.95R (5% buffer)

# ── Scale-In Improvements ──────────────────────────────────────────────
SCALE_MIN_PROFIT_R = 0.5                 # minimum profit (in R) before allowing scale-in
SCALE_STAKE_FRAC = 0.33                  # scale-in stake = fraction of current position (reduced from 0.5)

# ── Trailing SL Improvements ───────────────────────────────────────────
TRAILING_START_IMMEDIATE = _get_bool("TRAILING_START_IMMEDIATE", True)  # trail from entry, not after 1R
TRAILING_MIN_DISTANCE_ATR_MULT = 1.0       # minimum trailing distance as fraction of ATR (prevents too-tight trails) — INCREASED from 0.5
TRAILING_CHANDELIER_ENABLED = _get_bool("TRAILING_CHANDELIER_ENABLED", True)  # chandelier exit variant — ENABLED
TRAILING_CHANDELIER_MULT = 3.0             # chandelier exit ATR multiplier (typically 2-3)
TRAILING_ATR_MULT = 2.0                    # trail distance = mult * ATR14 — INCREASED from 1.5

# ── Time-Weighted SL Tightening ────────────────────────────────────────
TIME_WEIGHTED_SL_ENABLED = _get_bool("TIME_WEIGHTED_SL_ENABLED", True)
TIME_WEIGHTED_SL_TRIGGER = 0.75          # start tightening at 75% of max_hold (was 50%)
TIME_WEIGHTED_SL_PROGRESSIVE = _get_bool("TIME_WEIGHTED_SL_PROGRESSIVE", True)  # progressive tightening

# ── Dynamic Position Sizing (Volatility-Based) ─────────────────────────
DYNAMIC_SIZING_ENABLED = _get_bool("DYNAMIC_SIZING_ENABLED", True)      # scale stake by 1/abs_vol when vol elevated
DYNAMIC_SIZING_ABS_VOL_THRESHOLD = 1.5    # abs_vol ratio above which to reduce stake (e.g., 1.5 = 150% median)
DYNAMIC_SIZING_MIN_FRAC = 0.25            # minimum stake fraction multiplier (cap reduction)
DYNAMIC_SIZING_MAX_FRAC = 2.0             # maximum stake fraction multiplier (cap increase)

# ── Liquidity Gate Refinements ──────────────────────────────────────────
LIQ_ENTRY_SPREAD_BPS = 15.0               # entry spread threshold (separate from exit)
LIQ_ENTRY_MIN_DEPTH_USD = 75000.0         # entry min depth (separate from exit)
LIQ_ENTRY_MAX_VANISH_PCT = 50.0           # entry max vanish (separate from exit)
LIQ_ENTRY_MIN_INTENSITY_USD = 30000.0     # entry min trade intensity (separate from exit)
LIQ_IMBALANCE_ENABLED = _get_bool("LIQ_IMBALANCE_ENABLED", True)        # use bid/ask depth ratio
LIQ_IMBALANCE_THRESHOLD = 2.0             # max bid/ask depth ratio before blocking
LIQ_FUNDING_ENABLED = _get_bool("LIQ_FUNDING_ENABLED", True)            # consider funding rate
LIQ_FUNDING_THRESHOLD_BPS = 5.0           # funding rate threshold in bps (abs value)

# ── LIQ_EXIT Relaxation for Trending Markets ──────────────────────────────
LIQ_EXIT_ADAPTIVE_QUANTILE = 0.10             # use 10th percentile (was 25th) — more conservative floor
LIQ_EXIT_ADAPTIVE_MIN_USD = 3000.0            # absolute floor USD/s — LOWERED from 10000
LIQ_EXIT_ADAPTIVE_MAX_USD = 50000.0           # absolute ceiling USD/s — LOWERED from 100000
LIQ_EXIT_GRACE_PERIOD_MIN = 20                # minutes to suppress liq_exit after entry — INCREASED from 10
LIQ_EXIT_TREND_RELAX_ENABLED = _get_bool("LIQ_EXIT_TREND_RELAX_ENABLED", True)  # relax in strong trends
LIQ_EXIT_TREND_RELAX_MULT = 0.5               # multiply threshold by this in strong trend

# ── Trend-Following Filter ────────────────────────────────────────────────
TREND_FILTER_ENABLED = _get_bool("TREND_FILTER_ENABLED", True)   # enable EMA trend filter
TREND_FILTER_EMA_PERIOD = 50                  # EMA period for trend detection
TREND_FILTER_REQUIRE_ALIGNMENT = True         # require signal direction = trend direction

# ── Cost-to-Edge Optimization ───────────────────────────────────────────
COST_TRACKING_ENABLED = _get_bool("COST_TRACKING_ENABLED", True)        # track realized vs expected costs
COST_STAKE_POWER_ADJUST = _get_bool("COST_STAKE_POWER_ADJUST", True)    # adjust confidence power by cost_ratio
COST_STAKE_POWER_MIN = 0.3                # minimum confidence power when costs high
COST_STAKE_POWER_MAX = 0.7                # maximum confidence power when costs low
COST_SLIPPAGE_TOD_ENABLED = _get_bool("COST_SLIPPAGE_TOD_ENABLED", True) # time-of-day slippage model
COST_SLIPPAGE_TOD_WINDOWS = {"00-06": 1.5, "06-12": 1.0, "12-18": 0.8, "18-24": 1.2}  # mult per window

# ── Trade Cooldown (Phase 3) ───────────────────────────────────────────
TRADE_COOLDOWN_ENABLED = _get_bool("TRADE_COOLDOWN_ENABLED", True)
TRADE_COOLDOWN_MINUTES = 30           # minimum gap between new entries

# ── Circuit Breaker (Risk Management) ────────────────────────────────────
CIRCUIT_BREAKER_ENABLED = _get_bool("CIRCUIT_BREAKER_ENABLED", True)      # enable circuit breaker
CIRCUIT_BREAKER_MAX_DAILY_LOSS_PCT = 5.0     # max daily loss % of equity before halt (e.g., 5%)
CIRCUIT_BREAKER_MAX_CONSECUTIVE_LOSSES = 3   # max consecutive losing trades before halt
CIRCUIT_BREAKER_COOLDOWN_HOURS = 4           # hours to wait before resuming after trigger
CIRCUIT_BREAKER_MAX_DRAWDOWN_PCT = 10.0      # max total drawdown % before hard halt

# ── Startup Position Adoption ──────────────────────────────────────────
# Adopt an exchange position left open by a previous process (crash / killed
# session) so exits manage it instead of trading blindly beside it. When
# False, startup is refused if an open position exists.
ADOPT_EXISTING_POSITION = _get_bool("ADOPT_EXISTING_POSITION", True)

# ── Absolute Volume Gating (Oct 5 post-mortem) ─────────────────────────
# vol_ratio (vol_12/vol_48) is a RELATIVE metric: it stays near 1.0 when
# BOTH short-term and long-term return-volatility collapse together — which
# is exactly what happens during a liquidity vacuum.  quote_volume from the
# kline bar is the ABSOLUTE measure of actual trading activity.
#
# abs_vol_ratio = current_bar_quote_volume / rolling_median_quote_volume
#   < 0.10  → "abs_vol_dropped"  (volume < 10% of rolling median)
#   >= 0.10 → "normal"
#
# When abs_vol_ratio is critical, regime detection forces "halt" and the
# strategy blocks entries / exits positions.
ABS_VOL_RATIO_CRITICAL = 0.10             # ratio below this = absolute volume drop
ABS_VOL_RATIO_ROLLING_WINDOW = 48          # rolling window for median (candles)

# ── Signal Thresholds ──────────────────────────────────────────────────
# Tactical signal threshold (absolute; used when ADAPTIVE_THRESHOLD_ENABLED=False)
ABSOLUTE_THRESHOLD = 0.006          # Minimum prediction to trigger LONG (fallback)
ABSOLUTE_THRESHOLD_SHORT = 0.003    # Minimum prediction magnitude to trigger SHORT — LOWERED

# Adaptive threshold bounds
ADAPTIVE_MIN_THRESHOLD = 0.001      # floor so threshold never goes too low — LOWERED from 0.002
ADAPTIVE_MAX_THRESHOLD = 0.02       # ceiling so threshold never goes too high

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