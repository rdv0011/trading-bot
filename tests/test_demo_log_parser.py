"""Tests for the demo log parser (Task 2 of plans/demo_vs_sim_comparison.md)."""
import csv
from pathlib import Path
from datetime import datetime, timezone

import demo_log_parser as dlp

LOG_DAY_A = """\
2026-07-19 14:00:03,123 - INFO - ✅ DualMLStrategy initialized
2026-07-19 14:00:05,101 - INFO - Tactical | signal=BUY pred=0.6001 min=0.500000 max=0.700000
2026-07-19 14:00:05,102 - INFO - Strategic | allow=True regime=trend vol=0.12 leverage=1.0x exposure=0.10
2026-07-19 14:00:05,200 - INFO - 🟢 OPEN LONG @ 64200.0 qty=0.012
2026-07-19 14:00:05,201 - INFO - 🛡 TP=64800.0 SL=63500.0
2026-07-19 14:15:03,100 - INFO - Tactical | signal=HOLD pred=0.500000 min=0.600000 max=0.400000
2026-07-19 14:15:03,105 - INFO - Strategic | allow=True regime=trend vol=0.12 leverage=1.0x exposure=0.10
2026-07-19 18:00:00,500 - INFO - 🔵 FULL CLOSE (start) reason=take_profit — side=LONG amount=0.0120 entry=64200.0
2026-07-19 18:00:00,600 - INFO - 🔵 FULL CLOSE reason=take_profit — position closed
2026-07-19 18:00:01,100 - INFO - ℹ️ Gate counter summary | day=2026-07-19 vol_flt=5 htf_trd=12 adapt_thr=34 riskguard=0 chop=8 veto=2
2026-07-19 18:00:01,200 - INFO - 📊 Regime distribution (2026-07-19): chop=45% | high_vol=12% | trend=43%
"""

LOG_DAY_B = """\
2026-07-20 08:00:03,100 - INFO - Tactical | signal=SHORT pred=-0.712300 min=0.600000 max=0.400000
2026-07-20 08:00:03,101 - INFO - Strategic | allow=True regime=high_vol vol=0.30 leverage=1.0x exposure=0.10
2026-07-20 08:00:03,200 - INFO - 🟢 OPEN SHORT @ 65000.0 qty=0.010
2026-07-20 08:00:03,201 - INFO - 🛡 TP=64000.0 SL=66100.0
2026-07-20 09:30:00,100 - INFO - 🔵 FULL CLOSE (start) reason=stop_loss — side=SHORT amount=0.0100 entry=65000.0
2026-07-20 09:30:00,300 - INFO - 🔵 FULL CLOSE reason=stop_loss — position closed
2026-07-20 09:31:00,100 - INFO - 🔴 EMERGENCY CLOSE (live) qty=0.012 - position closed
2026-07-20 12:00:03,100 - INFO - Tactical | signal=BUY pred=0.900000 min=0.600000 max=0.400000
2026-07-20 12:00:03,200 - INFO - 🟢 OPEN LONG @ 64000.5 qty=0.020
2026-07-20 12:30:00,100 - INFO - 📈 SCALE UP LONG +0.010 (total=0.0300, scale#1)
2026-07-20 13:00:00,100 - INFO - 🔽 PARTIAL CLOSE -0.005 (remaining=0.0100)
2026-07-20 14:45:00,100 - INFO - 🔵 FULL CLOSE reason=stop_loss — position closed
"""


def _write_day(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_trade_take_profit_inference(tmp_path):
    p = _write_day(tmp_path, "trading_2026-07-19.log", LOG_DAY_A)
    res = dlp.parse_demo_logs([p])
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t["side"] == "LONG"
    assert t["entry_price"] == 64200.0
    assert t["exit_reason"] == "take_profit"
    assert t["tp"] == 64800.0
    assert t["sl"] == 63500.0
    assert t["regime"] == "trend"


def test_exit_price_inference_functions():
    tp = {"exit_reason": "take_profit", "tp": 64800.0, "sl": 63500.0}
    assert dlp._infer_exit_price(tp) == 64800.0
    sl = {"exit_reason": "stop_loss", "tp": 64800.0, "sl": 63500.0}
    assert dlp._infer_exit_price(sl) == 63500.0


def test_pnl_sign_convention():
    win = {"side": "LONG", "entry_price": 64200.0, "tp": 64800.0, "sl": 63500.0,
           "exit_reason": "take_profit"}
    assert dlp._pnl_raw(win) > 0
    loss = {"side": "SHORT", "entry_price": 65500.0, "tp": 64800.0, "sl": 66100.0,
            "exit_reason": "stop_loss"}
    assert dlp._pnl_raw(loss) < 0


def test_gate_counters_and_regime(tmp_path):
    p = _write_day(tmp_path, "trading_2026-07-19.log", LOG_DAY_A)
    res = dlp.parse_demo_logs([p])
    day = res.daily_gates["2026-07-19"]
    assert (day["vol_flt"], day["htf_trd"], day["adapt_thr"]) == (5, 12, 34)
    assert (day["riskguard"], day["chop"], day["veto"]) == (0, 8, 2)
    reg = res.regime_distribution["2026-07-19"]
    assert reg["chop"] == 45.0
    assert reg["trend"] == 43.0
    assert reg["high_vol"] == 12.0


def test_scale_partial_and_eof_open_trade(tmp_path):
    p = _write_day(tmp_path, "trading_2026-07-20.log", LOG_DAY_B)
    res = dlp.parse_demo_logs([p])
    reasons = {t["exit_reason"] for t in res.trades}
    assert "stop_loss" in reasons
    closed = [t for t in res.trades if t["side"] == "LONG"][0]
    assert closed["exit_reason"] == "stop_loss"
    assert closed["qty"] == 0.020 + 0.010 - 0.005
    assert closed["scale_count"] == 1


def test_write_csvs(tmp_path):
    _write_day(tmp_path, "trading_2026-07-19.log", LOG_DAY_A)
    _write_day(tmp_path, "trading_2026-07-20.log", LOG_DAY_B)
    outdir = tmp_path / "out"
    dlp.main(["--log", str(tmp_path), "--out", str(outdir)])
    trades = list(csv.DictReader(open(outdir / "demo_trades.csv")))
    daily = list(csv.DictReader(open(outdir / "demo_daily_summary.csv")))
    assert len(trades) == 3
    first = trades[0]
    assert first["side"] == "LONG"
    assert first["entry_ts"] == "2026-07-19 14:00:05"
    assert first["exit_ts"] == "2026-07-19 18:00:00"
    assert first["exit_reason"] == "take_profit"
    assert float(first["entry_price"]) == 64200.0
    assert float(first["exit_price"]) == 64800.0
    assert float(first["pnl_raw"]) > 0
    rows = {r["date"]: r for r in daily}
    assert rows["2026-07-19"]["entries"] == "1"
    assert rows["2026-07-19"]["vol_flt"] == "5"
    assert rows["2026-07-19"]["regime_trend_pct"] == "43.0"
    assert rows["2026-07-20"]["entries"] == "2"


def test_invalid_log_dir_raises():
    raised = False
    try:
        dlp._collect_log_files("/nonexistent")
    except FileNotFoundError:
        raised = True
    assert raised


def test_date_filter(tmp_path):
    _write_day(tmp_path, "trading_2026-07-19.log", LOG_DAY_A)
    _write_day(tmp_path, "trading_2026-07-20.log", LOG_DAY_B)
    res = dlp.parse_demo_logs(list(tmp_path.glob("trading_*.log")),
                              start_date="2026-07-20", end_date="2026-07-20")
    assert all(t["entry_ts"].date().isoformat() == "2026-07-20" for t in res.trades)