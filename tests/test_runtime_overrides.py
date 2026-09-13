"""Tests for runtime flag overrides (runtime_overrides.json):

  1. _cfg_flag() prefers a loaded override over injected/module config
  2. reload_runtime_overrides() loads a JSON file into the override dict
  3. Rewriting the file + mtime bump refreshes overrides at runtime
  4. Missing file / corrupt JSON / non-dict content -> fail-open, previous
     overrides preserved, False returned
  5. Integration: the liquidity exit gate honors a runtime override
"""

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import strategy as strat_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_runtime_state():
    """Reset the module-global override dict/mtime before and after each test."""
    prev_overrides = dict(strat_mod._RUNTIME_OVERRIDES)
    prev_mtime = strat_mod._RUNTIME_OVERRIDES_MTIME
    yield
    strat_mod._RUNTIME_OVERRIDES.clear()
    strat_mod._RUNTIME_OVERRIDES.update(prev_overrides)
    strat_mod._RUNTIME_OVERRIDES_MTIME = prev_mtime


def _bump_mtime(path: Path) -> None:
    """Force a strictly newer mtime (future-timestamped) so the mtime guard
    sees a change even when writes land within the same clock second."""
    t = time.time() + 5.0
    os.utime(path, (t, t))


def _write_overrides(tmp_path: Path, payload) -> Path:
    path = tmp_path / "runtime_overrides.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _cfg(**overrides):
    cfg = SimpleNamespace(GATE_LIQUIDITY_EXIT=True, LIQ_EXIT_SPREAD_BPS=20.0,
                          LIQ_EXIT_MIN_DEPTH_USD=25000.0)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_strat(cfg, monitor=None):
    broker = MagicMock()
    broker.symbol = "BTCUSDT"
    return strat_mod.DualMLStrategy(
        broker=broker, config=cfg, tactical_model=MagicMock(),
        strategic_model=MagicMock(), feature_cols=["close"],
        liquidity_monitor=monitor,
    )


class _FakeMonitor:
    def __init__(self, snap):
        self._snap = snap

    def snapshot(self):
        return self._snap


def _fresh_snapshot(**overrides):
    snap = {"fresh": True, "spread_bps": 1.0, "depth_total_usd": 400000.0}
    snap.update(overrides)
    return snap


# ── _cfg_flag override precedence ──────────────────────────────────────
def test_cfg_flag_prefers_override_over_injected_config():
    strat_mod._RUNTIME_OVERRIDES["LIQ_EXIT_SPREAD_BPS"] = 5.0
    cfg = _cfg(LIQ_EXIT_SPREAD_BPS=20.0)
    assert strat_mod._cfg_flag("LIQ_EXIT_SPREAD_BPS", 999.0, cfg) == 5.0


def test_cfg_flag_override_beats_module_config_default():
    strat_mod._RUNTIME_OVERRIDES["LIQ_EXIT_SPREAD_BPS"] = 5.0
    # No injected cfg -> falls to module config; override must win
    assert strat_mod._cfg_flag("LIQ_EXIT_SPREAD_BPS", 999.0) == 5.0


def test_cfg_flag_unset_override_falls_back_to_injected_config():
    cfg = _cfg(LIQ_EXIT_SPREAD_BPS=20.0)
    assert strat_mod._cfg_flag("LIQ_EXIT_SPREAD_BPS", 999.0, cfg) == 20.0


# ── reload_runtime_overrides loading ───────────────────────────────────
def test_reload_loads_file_and_applies(tmp_path):
    path = _write_overrides(tmp_path, {"LIQ_EXIT_SPREAD_BPS": 5.0})
    assert strat_mod.reload_runtime_overrides(path) is True
    assert strat_mod._cfg_flag("LIQ_EXIT_SPREAD_BPS", 999.0) == 5.0


def test_reload_unchanged_file_returns_false(tmp_path):
    path = _write_overrides(tmp_path, {"LIQ_EXIT_SPREAD_BPS": 5.0})
    assert strat_mod.reload_runtime_overrides(path) is True
    # Same file, same mtime -> not re-parsed, dict untouched
    assert strat_mod.reload_runtime_overrides(path) is False
    assert strat_mod._cfg_flag("LIQ_EXIT_SPREAD_BPS", 999.0) == 5.0


def test_reload_refreshes_after_file_change(tmp_path):
    path = _write_overrides(tmp_path, {"LIQ_EXIT_SPREAD_BPS": 5.0})
    assert strat_mod.reload_runtime_overrides(path) is True
    path.write_text(json.dumps({"LIQ_EXIT_SPREAD_BPS": 30.0}), encoding="utf-8")
    _bump_mtime(path)
    assert strat_mod.reload_runtime_overrides(path) is True
    assert strat_mod._cfg_flag("LIQ_EXIT_SPREAD_BPS", 999.0) == 30.0


def test_reload_accumulates_new_keys(tmp_path):
    path = _write_overrides(tmp_path, {"LIQ_EXIT_SPREAD_BPS": 5.0})
    strat_mod.reload_runtime_overrides(path)
    path.write_text(json.dumps({"GATE_LIQUIDITY_EXIT": False}), encoding="utf-8")
    _bump_mtime(path)
    strat_mod.reload_runtime_overrides(path)
    assert strat_mod._cfg_flag("LIQ_EXIT_SPREAD_BPS", 999.0) == 5.0
    assert strat_mod._cfg_flag("GATE_LIQUIDITY_EXIT", True) is False


# ── Fail-open semantics ────────────────────────────────────────────────
def test_reload_missing_file_returns_false(tmp_path):
    assert strat_mod.reload_runtime_overrides(tmp_path / "nope.json") is False


def test_reload_corrupt_json_preserves_previous_overrides(tmp_path):
    path = _write_overrides(tmp_path, {"LIQ_EXIT_SPREAD_BPS": 5.0})
    assert strat_mod.reload_runtime_overrides(path) is True
    path.write_text("{not valid json", encoding="utf-8")
    _bump_mtime(path)
    assert strat_mod.reload_runtime_overrides(path) is False
    assert strat_mod._cfg_flag("LIQ_EXIT_SPREAD_BPS", 999.0) == 5.0


def test_reload_non_dict_json_preserves_previous(tmp_path):
    path = _write_overrides(tmp_path, {"LIQ_EXIT_SPREAD_BPS": 5.0})
    assert strat_mod.reload_runtime_overrides(path) is True
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    _bump_mtime(path)
    assert strat_mod.reload_runtime_overrides(path) is False
    assert strat_mod._cfg_flag("LIQ_EXIT_SPREAD_BPS", 999.0) == 5.0


# ── Integration: liquidity exit gate honors runtime overrides ──────────
def test_exit_gate_honors_runtime_override(tmp_path):
    path = _write_overrides(tmp_path, {"LIQ_EXIT_SPREAD_BPS": 5.0})
    strat_mod.RUNTIME_OVERRIDES_PATH = path
    strat_mod.reload_runtime_overrides(path)
    try:
        cfg = _cfg(LIQ_EXIT_SPREAD_BPS=20.0)
        strat = _make_strat(cfg, _FakeMonitor(_fresh_snapshot(spread_bps=10.0)))
        # Config threshold is 20.0 (no exit at 10bps); override is 5.0 -> exit
        assert strat._liquidity_exit_gate() == "liq_exit"
    finally:
        strat_mod.RUNTIME_OVERRIDES_PATH = strat_mod.Path(
            strat_mod.__file__
        ).parent / "runtime_overrides.json"