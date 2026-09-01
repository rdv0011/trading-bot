"""
Unified logging for Dual-ML Bitcoin Trading Bot.
Based on original binancebasebroker.py's DailyDatedLogHandler.
"""

import os
import logging
import logging.handlers
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# ── Configuration ──────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

FILE_LOG_LEVEL = logging.DEBUG
CONSOLE_LOG_LEVEL = logging.INFO
LOG_RETENTION_DAYS = 10

# Trade CSV
TRADE_CSV_DIR = LOG_DIR
TRADE_CSV_PREFIX = "trades"


# ── Daily Dated Log Handler (from original) ─────────────────────────────
def _today_log_path() -> Path:
    """Path of the current UTC-day log file."""
    return LOG_DIR / f"trading_{datetime.now(timezone.utc):%Y-%m-%d}.log"


class DailyDatedLogHandler(logging.Handler):
    """Writes one daily file per UTC date with rotation and pruning."""

    def __init__(self, log_dir: Path = LOG_DIR, level: int = FILE_LOG_LEVEL):
        super().__init__(level=level)
        self.log_dir = log_dir
        self._stream = None

    @property
    def stream(self):
        return self._stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._stream is None:
                self._open_today()
            msg = self.format(record)
            if "\n" in msg:
                msg = msg.replace("\n", " | ")
            self._stream.write(msg + "\n")
            self.flush()
        except Exception:
            self.handleError(record)

    def _open_today(self) -> None:
        self._prune_old_files()
        self._stream = open(self._today(), "a", encoding="utf-8")

    def _today(self) -> Path:
        return _today_log_path()

    def _prune_old_files(self) -> None:
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=LOG_RETENTION_DAYS)
        if not self.log_dir.exists():
            return
        for f in self.log_dir.glob("trading_????-??-??.log"):
            try:
                fdate = datetime.strptime(f.stem.split("_", 1)[1], "%Y-%m-%d").date()
                if fdate < cutoff:
                    f.unlink()
            except (IndexError, ValueError):
                continue

    def flush(self):
        if self._stream is not None:
            self._stream.flush()

    def close(self):
        if self._stream is not None:
            self._stream.close()
        super().close()


# ── Setup ──────────────────────────────────────────────────────────────
def setup_logging(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_dir: Path = LOG_DIR,
) -> logging.Logger:
    """
    Configure root logger with daily-dated file handler and console.
    Returns the configured root logger.
    """
    log_dir.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(FILE_LOG_LEVEL)

    # Clear existing handlers (idempotent)
    root.handlers.clear()

    # ── File Handler (daily-dated, from original) ──────────────────────
    file_handler = DailyDatedLogHandler(log_dir)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s"
    ))
    file_handler.setLevel(FILE_LOG_LEVEL)
    root.addHandler(file_handler)

    # ── Console Handler (brief) ────────────────────────────────────────
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    console.setLevel(CONSOLE_LOG_LEVEL)
    root.addHandler(console)

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("binance").setLevel(logging.WARNING)
    logging.getLogger("catboost").setLevel(logging.WARNING)

    return root


# ── Trade CSV Logging ──────────────────────────────────────────────────
TRADE_CSV_FIELDS = [
    "timestamp",          # ISO format entry time
    "symbol",
    "side",               # "long" or "short"
    "entry_price",
    "exit_price",
    "qty",                # Position size (base currency)
    "stake_frac",         # Fraction of equity
    "leverage",
    "stop_loss",
    "take_profit",
    "max_hold_hours",
    "regime",
    "exit_reason",        # "sl", "tp", "max_hold", "reversal", "manual"
    "pnl",                # Realized PnL (quote currency)
    "pnl_pct",            # PnL as % of stake
    "equity_before",
    "equity_after",
    "fee_paid",
    "slippage_paid",
    "tactical_pred",      # Tactical ML prediction at entry
    "strategic_params",   # JSON string of strategic meta-params
]

def _get_trade_csv_path() -> Path:
    """Get today's trade CSV path."""
    today = datetime.now().strftime("%Y-%m-%d")
    return TRADE_CSV_DIR / f"{TRADE_CSV_PREFIX}_{today}.csv"


def log_trade(trade: Dict[str, Any]) -> None:
    """
    Append a completed trade to today's CSV log.
    Creates file with header if it doesn't exist.
    """
    csv_path = _get_trade_csv_path()
    file_exists = csv_path.exists()

    # Ensure all fields present
    row = {field: trade.get(field, "") for field in TRADE_CSV_FIELDS}

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # Also log to file logger
    logger = logging.getLogger(__name__)
    logger.debug(
        f"TRADE | {trade.get('timestamp')} | {trade.get('side').upper()} | "
        f"Entry: {trade.get('entry_price'):.2f} | Exit: {trade.get('exit_price'):.2f} | "
        f"PnL: {trade.get('pnl'):.6f} ({trade.get('pnl_pct'):.2%}) | "
        f"Regime: {trade.get('regime')} | Reason: {trade.get('exit_reason')}"
    )


def log_trade_entry(
    timestamp: datetime,
    symbol: str,
    side: str,
    entry_price: float,
    qty: float,
    stake_frac: float,
    leverage: float,
    stop_loss: float,
    take_profit: float,
    max_hold_hours: float,
    regime: str,
    tactical_pred: float,
    strategic_params: dict,
    equity_before: float,
) -> Dict[str, Any]:
    """
    Create a trade dict for an entry (exit fields left empty).
    Returns the trade dict to be updated on exit.
    """
    import json
    return {
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "side": side,
        "entry_price": round(entry_price, 2),
        "exit_price": "",
        "qty": round(qty, 6),
        "stake_frac": round(stake_frac, 4),
        "leverage": round(leverage, 2),
        "stop_loss": round(stop_loss, 4),
        "take_profit": round(take_profit, 4),
        "max_hold_hours": round(max_hold_hours, 2),
        "regime": regime,
        "exit_reason": "",
        "pnl": "",
        "pnl_pct": "",
        "equity_before": round(equity_before, 6),
        "equity_after": "",
        "fee_paid": "",
        "slippage_paid": "",
        "tactical_pred": round(tactical_pred, 6),
        "strategic_params": json.dumps(strategic_params),
    }


def log_trade_exit(
    trade: Dict[str, Any],
    exit_price: float,
    exit_reason: str,
    pnl: float,
    pnl_pct: float,
    equity_after: float,
    fee_paid: float,
    slippage_paid: float,
) -> Dict[str, Any]:
    """
    Update trade dict with exit info and log to CSV.
    Returns the completed trade dict.
    """
    trade.update({
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "pnl": round(pnl, 6),
        "pnl_pct": round(pnl_pct, 6),
        "equity_after": round(equity_after, 6),
        "fee_paid": round(fee_paid, 6),
        "slippage_paid": round(slippage_paid, 6),
    })
    log_trade(trade)
    return trade


# ── Convenience Logging Functions ──────────────────────────────────────
def log_info(msg: str) -> None:
    """Log INFO to console + file."""
    logging.getLogger(__name__).info(msg)


def log_debug(msg: str) -> None:
    """Log DEBUG to file only."""
    logging.getLogger(__name__).debug(msg)


def log_warning(msg: str) -> None:
    """Log WARNING to console + file."""
    logging.getLogger(__name__).warning(msg)


def log_error(msg: str) -> None:
    """Log ERROR to console + file."""
    logging.getLogger(__name__).error(msg)


# ── Equity Curve Logging (Optional) ────────────────────────────────────
EQUITY_CSV_FIELDS = ["timestamp", "equity", "position", "unrealized_pnl"]

def _get_equity_csv_path() -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR / f"equity_{today}.csv"


def log_equity(timestamp: datetime, equity: float, position: float = 0.0, unrealized_pnl: float = 0.0):
    """Log equity curve point to CSV."""
    csv_path = _get_equity_csv_path()
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EQUITY_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": timestamp.isoformat(),
            "equity": round(equity, 6),
            "position": round(position, 6),
            "unrealized_pnl": round(unrealized_pnl, 6),
        })


# ── Module Initialization ──────────────────────────────────────────────
# Auto-setup when imported
if not logging.getLogger().handlers:
    setup_logging()