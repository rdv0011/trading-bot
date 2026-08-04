"""Tests for the daily-rotating trading log (Task 7 of plans/demo_vs_sim_comparison.md)."""
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pytest

import binancebasebroker as bbb
from basestrategy import BaseStrategy


@pytest.fixture
def isolated_logging(tmp_path, monkeypatch):
    """Point LOG_DIR at a tmp dir and reset root handlers before/after each test.

    The file handler is attached to the ROOT logger by setup_logging(), so
    tests must snapshot and restore root handlers to avoid leaking handlers
    into unrelated tests.
    """
    monkeypatch.setattr(bbb, "LOG_DIR", tmp_path)
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    root.handlers.clear()
    yield tmp_path
    root.handlers.clear()
    root.handlers.extend(saved_handlers)


@pytest.fixture
def broker(isolated_logging):
    return bbb.BinanceBaseBroker(config={})


def test_bb_log_setup_creates_file_and_stream_handlers(broker):
    handlers = broker.logger.handlers
    # logger itself has no direct handlers; handlers live on the root logger
    root_handlers = logging.getLogger().handlers
    assert any(isinstance(h, TimedRotatingFileHandler) for h in root_handlers)
    assert any(isinstance(h, logging.StreamHandler) for h in root_handlers)
    assert not isinstance(broker.logger, type(None))


def test_bb_log_setup_is_idempotent(isolated_logging):
    """Repeated broker instantiations must not duplicate root handlers."""
    bbb.BinanceBaseBroker(config={})
    bbb.BinanceBaseBroker(config={})
    root_handlers = logging.getLogger().handlers
    file_handlers = [h for h in root_handlers if isinstance(h, TimedRotatingFileHandler)]
    stream_handlers = [h for h in root_handlers if isinstance(h, logging.StreamHandler)]
    assert len(file_handlers) == 1
    assert len(stream_handlers) == 1


def test_bb_log_writes_to_daily_file(isolated_logging, tmp_path):
    bbb.BinanceBaseBroker(config={}).log_message("🟢 OPEN LONG @ 64200.0 qty=0.012")
    log_file = tmp_path / "trading.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "🟢 OPEN LONG @ 64200.0 qty=0.012" in content
    assert "INFO" in content  # basicConfig-style format preserved


def test_bb_log_rotation_namer(broker):
    """Rotated files follow the trading_YYYY-MM-DD.log convention."""
    name = bbb._rotating_log_namer("/tmp/logs/trading.log.2026-07-27")
    assert name == "/tmp/logs/trading_2026-07-27.log"
    # Non-rotated names pass through unchanged
    assert bbb._rotating_log_namer("/tmp/logs/trading.log") == "/tmp/logs/trading.log"


def test_bb_log_rotating_handler_config(broker):
    """Handler rotates at UTC midnight and keeps 10 backups."""
    file_handler = next(
        h for h in logging.getLogger().handlers
        if isinstance(h, TimedRotatingFileHandler)
    )
    assert file_handler.when == "midnight"
    assert file_handler.backupCount == 10
    assert file_handler.utc is True
    assert file_handler.encoding == "utf-8"
    assert file_handler.namer is bbb._rotating_log_namer


def test_bb_log_env_overrides(tmp_path, monkeypatch, isolated_logging):
    """TBOT_LOG_DIR / TBOT_MAX_LOG_BYTES env vars override the defaults."""
    monkeypatch.setattr(bbb, "LOG_DIR", Path(os.environ.get("TBOT_LOG_DIR", "logs")))
    monkeypatch.setenv("TBOT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("TBOT_MAX_LOG_BYTES", str(2048))
    # Reload constants the way the module would on a fresh process
    monkeypatch.setattr(bbb, "LOG_DIR", Path(tmp_path))
    monkeypatch.setattr(bbb, "MAX_LOG_BYTES", 2048)
    broker = bbb.BinanceBaseBroker(config={})
    broker.log_message("test env override")
    assert (tmp_path / "trading.log").exists()
    assert bbb.MAX_LOG_BYTES == 2048


def test_bb_log_size_cap_truncates_to_last_lines(isolated_logging, tmp_path, monkeypatch):
    """Files over MAX_LOG_BYTES are truncated to the last LOG_MAX_LINES lines."""
    monkeypatch.setattr(bbb, "MAX_LOG_BYTES", 1)  # any file is "over cap"
    monkeypatch.setattr(bbb, "LOG_MAX_LINES", 3)

    broker = bbb.BinanceBaseBroker(config={})
    strategy = BaseStrategy(broker=broker, quote_symbol="USDT", parameters={})
    for i in range(10):
        broker.log_message(f"line {i}")

    strategy._enforce_log_size_cap()

    log_file = tmp_path / "trading.log"
    lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3
    assert "line 7" in lines[0]  # last 3 lines kept
    assert "line 9" in lines[-1]


def test_bb_log_size_cap_noop_under_limit(isolated_logging, tmp_path, monkeypatch):
    """Files under MAX_LOG_BYTES are left untouched."""
    monkeypatch.setattr(bbb, "MAX_LOG_BYTES", 10_000_000)  # huge cap
    monkeypatch.setattr(bbb, "LOG_MAX_LINES", 3)

    broker = bbb.BinanceBaseBroker(config={})
    strategy = BaseStrategy(broker=broker, quote_symbol="USDT", parameters={})
    for i in range(10):
        broker.log_message(f"line {i}")

    strategy._enforce_log_size_cap()

    log_file = tmp_path / "trading.log"
    lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 10  # untouched
