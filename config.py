"""
Centralized configuration module.
Loads environment variables from .env file in repo root.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root
_REPO_ROOT = Path(__file__).parent
load_dotenv(_REPO_ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    """Get environment variable with optional default."""
    return os.getenv(key, default)


def _get_bool(key: str, default: bool = False) -> bool:
    """Get environment variable as boolean."""
    val = os.getenv(key, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _get_int(key: str, default: int = 0) -> int:
    """Get environment variable as integer."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float = 0.0) -> float:
    """Get environment variable as float."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# ── Binance API Credentials ───────────────────────────────────────────
# Testnet Futures
BINANCE_TESTNET_FUTURES_API_KEY: str = _get("BINANCE_TESTNET_FUTURES_API_KEY")
BINANCE_TESTNET_FUTURES_API_SECRET: str = _get("BINANCE_TESTNET_FUTURES_API_SECRET")

# Testnet Spot
BINANCE_TESTNET_SPOT_API_KEY: str = _get("BINANCE_TESTNET_SPOT_API_KEY")
BINANCE_TESTNET_SPOT_API_SECRET: str = _get("BINANCE_TESTNET_SPOT_API_SECRET")

# Live (production) - keep empty for safety, use testnet vars above
BINANCE_LIVE_FUTURES_API_KEY: str = _get("BINANCE_LIVE_FUTURES_API_KEY")
BINANCE_LIVE_FUTURES_API_SECRET: str = _get("BINANCE_LIVE_FUTURES_API_SECRET")
BINANCE_LIVE_SPOT_API_KEY: str = _get("BINANCE_LIVE_SPOT_API_KEY")
BINANCE_LIVE_SPOT_API_SECRET: str = _get("BINANCE_LIVE_SPOT_API_SECRET")


# ── Legacy/Compatibility aliases (for mltraining.py) ──────────────────
# These map to futures testnet by default
BINANCE_TESTNET_API_KEY: str = BINANCE_TESTNET_FUTURES_API_KEY
BINANCE_TESTNET_API_SECRET: str = BINANCE_TESTNET_FUTURES_API_SECRET


# ── Trading Configuration ─────────────────────────────────────────────
DEFAULT_SYMBOL: str = _get("SYMBOL", "BTCUSDT")
DEFAULT_MARKET_TYPE: str = _get("MARKET_TYPE", "futures")  # "futures" or "spot"
DEFAULT_TESTNET: bool = _get_bool("TESTNET", True)

# Strategy parameters
DEFAULT_SLEEPTIME: str = _get("SLEEPTIME", "5m")
DEFAULT_LEVERAGE: float = _get_float("LEVERAGE", 1.0)


# ── Model Configuration ───────────────────────────────────────────────
STRATEGIC_TIMEFRAME: str = _get("STRATEGIC_TIMEFRAME", "1h")
TACTICAL_TIMEFRAME: str = _get("TACTICAL_TIMEFRAME", "5m")
STRATEGIC_DAYS: int = _get_int("STRATEGIC_DAYS", 365)
TACTICAL_DAYS: int = _get_int("TACTICAL_DAYS", 45)

# Model training parameters
TACTICAL_ITERATIONS: int = _get_int("TACTICAL_ITERATIONS", 300)
STRATEGIC_ITERATIONS: int = _get_int("STRATEGIC_ITERATIONS", 300)
MODEL_TYPE: str = _get("MODEL_TYPE", "cat")


# ── Simulation Configuration ──────────────────────────────────────────
SIMULATION_FEE: float = _get_float("SIMULATION_FEE", 0.0004)
SIMULATION_SLIPPAGE: float = _get_float("SIMULATION_SLIPPAGE", 0.0003)
INITIAL_EQUITY: float = _get_float("INITIAL_EQUITY", 1.0)

# Walk-forward retraining
WALKFORWARD_RETRAIN_EVERY: int = _get_int("WALKFORWARD_RETRAIN_EVERY", 100)


# ── Paths ─────────────────────────────────────────────────────────────
MODEL_DIR: str = _get("MODEL_DIR", "models")
LABEL_DIR: str = _get("LABEL_DIR", "labeleddata")
LOG_DIR: str = _get("LOG_DIR", "logs")
DATA_DIR: str = _get("DATA_DIR", "data")
CACHE_DIR: str = _get("CACHE_DIR", "cache")


# ── Logging ───────────────────────────────────────────────────────────
CONSOLE_LOG_LEVEL: str = _get("CONSOLE_LOG_LEVEL", "INFO")
FILE_LOG_LEVEL: str = _get("FILE_LOG_LEVEL", "DEBUG")


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