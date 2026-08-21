"""demo_log_parser.py — extract demo (live) trading records from the daily bot logs.

Parses ``logs/trading_YYYY-MM-DD.log`` files (written by ``DailyDatedLogHandler``
in ``binancebasebroker.py``, format ``%(asctime)s - %(levelname)s - %(message)s``,
UTC) into two comparison CSVs:

- ``trades/demo_trades.csv``         — one row per closed position
- ``trades/demo_daily_summary.csv``  — one row per UTC day (entries/exits/win-rate/PnL
  plus the live per-day gate counters and regime distribution)

The CSV schemas are mirrored by the windowed simulation harness
(``dualmlsimulation.py``) and consumed by ``compare_demo_vs_sim.py``.

Log line formats handled (portion after the ``- LEVEL - `` prefix; the leading
emoji may be absent depending on terminal encoding):

::

    🟢 OPEN LONG @ 64200.0 qty=0.012
    📈 SCALE UP LONG +0.006 (total=0.0180, scale#1)
    🔽 PARTIAL CLOSE -0.004 (remaining=0.0080)
    🔵 FULL CLOSE (start) reason=MAX_HOLD_TIME — side=LONG amount=0.0120 entry=64200.0
    🔵 FULL CLOSE reason=MAX_HOLD_TIME — position closed
    🔴 EMERGENCY CLOSE (live) qty=0.012 - position closed
      🛡 TP=64700.0 SL=63600.0
    ℹ️ Gate counter summary | day=2026-07-19 vol_flt=5 htf_trd=12 adapt_thr=34 riskguard=0 chop=8 veto=2 [| pred(...)]
    📊 Regime distribution (2026-07-19): chop=45% | high_vol=12% | trend=43%

Exit price inference: the log has no fill price for FULL CLOSE lines, so when the
exit reason is ``take_profit``/``stop_loss`` and the entry's TP/SL prices were
logged (``🛡 TP=... SL=...``), we use them; otherwise ``exit_price``/``pnl_raw``
are left empty and the trade is counted "unclosed/unpriced" in the summary.

Run (no pandas required — pure stdlib):

    python demo_log_parser.py --log logs --start-date 2026-07-19 --end-date 2026-07-26
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})")
LEVEL_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - \w+ - ")

# --- position lifecycle lines (the emoji is optional; match the ASCII core) ---
RE_OPEN = re.compile(r"OPEN (LONG|SHORT) @ ([\d.]+) qty=([\d.]+)")
RE_SCALE_UP = re.compile(r"SCALE UP (LONG|SHORT) \+([\d.]+) \(total=([\d.]+)")
RE_PARTIAL_CLOSE = re.compile(r"PARTIAL CLOSE -([\d.]+) \(remaining=([\d.]+)\)")
RE_FULL_CLOSE_START = re.compile(
    r"FULL CLOSE \(start\) reason=(\S+) .*?side=(LONG|SHORT) amount=([\d.]+)"
)
RE_FULL_CLOSE_DONE = re.compile(
    r"FULL CLOSE reason=(\S+) .*?position closed(?:\s+fill=([\d.]+))?"
)
RE_EMERGENCY = re.compile(r"EMERGENCY CLOSE")
RE_EMERGENCY_FILL = re.compile(r"EMERGENCY CLOSE \(live\) qty=[\d.]+ - position closed(?:\s+fill=([\d.]+))?")
RE_TP_SL = re.compile(r"TP=([\d.]+) SL=([\d.]+)")

# --- daily diagnostics lines ---
RE_GATE_SUMMARY = re.compile(r"Gate counter summary \| day=(\S+) (.*)")
RE_REGIME_DIST = re.compile(r"Regime distribution \((\S+)\): (.*)")

# --- heartbeat / signal lines ---
RE_TACTICAL = re.compile(r"Tactical \| signal=(\w+) .*?pred=([\d.-]+)")
RE_STRATEGIC = re.compile(r"Strategic \| allow=(\S+) regime=(\S+)")

GATE_KEYS = ("vol_flt", "htf_trd", "adapt_thr", "riskguard", "chop", "veto")
REGIME_KEYS = ("trend", "chop", "high_vol")

TRADES_HEADER = [
    "entry_ts",
    "exit_ts",
    "side",
    "entry_price",
    "exit_price",
    "qty",
    "exit_reason",
    "pnl_raw",
    "regime",
]
DAILY_SUMMARY_HEADER = [
    "date",
    "entries",
    "exits",
    "win_rate",
    "pnl_total",
    "vol_flt",
    "htf_trd",
    "adapt_thr",
    "riskguard",
    "chop",
    "veto",
    "regime_trend_pct",
    "regime_chop_pct",
    "regime_highvol_pct",
]


def _f(v) -> str:
    """Format helper: float -> str without trailing noise; None -> ''."""
    if v is None:
        return ""
    return f"{v:.8f}".rstrip("0").rstrip(".")


def _fmt_ts(ts) -> str:
    """Format a datetime as '%Y-%m-%d %H:%M:%S' (UTC); None -> ''."""
    if ts is None:
        return ""
    return ts.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class DemoLogResult:
    """Parsed demo log data, mirroring the plan's Task-2 deliverable."""

    trades: List[dict] = field(default_factory=list)
    daily_gates: Dict[str, Dict[str, int]] = field(default_factory=dict)
    regime_distribution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    heartbeat: List[dict] = field(default_factory=list)
    unclosed_trades: int = 0


def _parse_line_ts(line: str) -> Optional[datetime]:
    """Return the UTC datetime from a log line, e.g. '2026-07-19 14:00:03,999 - ...'."""
    m = TS_RE.match(line)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )


def _strip_level_prefix(line: str) -> str:
    """Remove the leading '2026-07-19 14:00:03,999 - INFO - ' prefix (if present)."""
    return LEVEL_PREFIX_RE.sub("", line, count=1)


def _parse_regime_pcts(body: str) -> Dict[str, float]:
    """Parse 'chop=45% | high_vol=12% | trend=43%' into floats."""
    out = {k: 0.0 for k in REGIME_KEYS}
    for key in REGIME_KEYS:
        m = re.search(rf"{key}=([\d.]+)%", body)
        if m:
            out[key] = float(m.group(1))
    return out


def parse_demo_logs(
    log_paths: List[Path],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> DemoLogResult:
    """Parse daily log file(s) into DemoLogResult.

    Args:
        log_paths: file path(s) to ``logs/trading_YYYY-MM-DD.log``-style files.
        start_date / end_date: 'YYYY-MM-DD' UTC; inclusive filter on trade entry_ts.
    Returns:
        DemoLogResult containing reconstructed trades and per-day summaries.
    """
    result = DemoLogResult()

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if start_date else None
    from datetime import timedelta as _timedelta

    end_dt = (
        datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + _timedelta(days=1)
        if end_date
        else None
    )

    # Serialize logs by file name (trading_YYYY-MM-DD.log sorts chronologically).
    active: Optional[dict] = None  # position being reconstructed
    latest_regime: str = ""  # from the most recent 'Strategic |' heartbeat
    for path in sorted(log_paths):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if not line:
                    continue

                ts = _parse_line_ts(line)
                if ts is None:
                    continue  # header / partial line
                if end_dt and ts >= end_dt:
                    continue
                if start_dt and ts < start_dt:
                    continue

                body = _strip_level_prefix(line)

                # ---- heartbeat / signal-level record ----
                hm = RE_TACTICAL.search(body)
                if hm:
                    result.heartbeat.append(
                        {"ts": ts, "line": body, "signal": hm.group(1), "pred": float(hm.group(2))}
                    )
                sm = RE_STRATEGIC.search(body)
                if sm:
                    result.heartbeat.append(
                        {"ts": ts, "line": body, "allow": sm.group(1), "regime": sm.group(2)}
                    )
                    if sm.group(1) == "True" and sm.group(2) != "unknown":
                        latest_regime = sm.group(2)

                # ---- daily counters / regime distribution ----
                gm = RE_GATE_SUMMARY.search(body)
                if gm:
                    day = gm.group(1)
                    if not start_date or day >= start_date:
                        if not end_date or day <= end_date:
                            result.daily_gates[day] = _parse_gate_pairs(body)
                rm = RE_REGIME_DIST.search(body)
                if rm:
                    day = rm.group(1)
                    if not start_date or day >= start_date:
                        if not end_date or day <= end_date:
                            result.regime_distribution[day] = _parse_regime_pcts(rm.group(2))

                # ---- position lifecycle ----
                if active is None:
                    om = RE_OPEN.search(body)
                    if om:
                        active = {
                            "side": om.group(1),
                            "entry_price": float(om.group(2)),
                            "qty": float(om.group(3)),
                            "scale_count": 0,
                            "tp": None,
                            "sl": None,
                            "entry_ts": ts,
                            "regime": latest_regime,
                        }
                        continue

                if active is not None:
                    fm = RE_FULL_CLOSE_DONE.search(body)
                    if fm:
                        active["exit_reason"] = fm.group(1)
                        active["exit_ts"] = ts
                        if fm.group(2):
                            active["exit_price"] = float(fm.group(2))
                        result.trades.append(active)
                        active = None
                        continue

                    em = RE_EMERGENCY_FILL.search(body)
                    if em:
                        # Broker-side / abrupt close — no 'FULL CLOSE reason=' line.
                        active["exit_reason"] = "emergency_close"
                        active["exit_ts"] = ts
                        if em.group(1):
                            active["exit_price"] = float(em.group(1))
                        result.trades.append(active)
                        active = None
                        continue

                    sm2 = RE_SCALE_UP.search(body)
                    if sm2 and sm2.group(1) == active["side"]:
                        active["qty"] += float(sm2.group(2))
                        active["scale_count"] += 1

                    pm = RE_PARTIAL_CLOSE.search(body)
                    if pm:
                        active["qty"] -= float(pm.group(1))

                    tsm = RE_TP_SL.search(body)
                    if tsm:
                        active["tp"] = float(tsm.group(1))
                        active["sl"] = float(tsm.group(2))

    # EOF: anything still open never printed a close line.
    if active is not None:
        active["exit_reason"] = "open_at_eof"
        active["exit_ts"] = active["entry_ts"]
        result.unclosed_trades += 1
        result.trades.append(active)

    return result


def _parse_gate_pairs(body: str) -> Dict[str, int]:
    """Parse 'vol_flt=5 htf_trd=12 adapt_thr=34 ...' into ints (missing -> 0)."""
    out = {k: 0 for k in GATE_KEYS}
    for key in GATE_KEYS:
        m = re.search(rf"{key}=(\d+)", body)
        if m:
            out[key] = int(m.group(1))
    return out


def _infer_exit_price(trade: dict) -> Optional[float]:
    """Infer the close price from logged TP/SL when the log doesn't print fills."""
    if trade["exit_reason"] == "take_profit" and trade.get("tp"):
        return trade["tp"]
    if trade["exit_reason"] == "stop_loss" and trade.get("sl"):
        return trade["sl"]
    return None


def _pnl_raw(trade: dict) -> Optional[float]:
    exit_price = trade.get("exit_price") or _infer_exit_price(trade)
    if exit_price is None or not trade["entry_price"]:
        return None
    if trade["side"] == "LONG":
        return exit_price / trade["entry_price"] - 1.0
    return trade["entry_price"] / exit_price - 1.0


def _write_trades_csv(result: DemoLogResult, out: Path):
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRADES_HEADER)
        writer.writeheader()
        for t in result.trades:
            exit_price = t.get("exit_price")
            if exit_price is None:
                exit_price = _infer_exit_price(t)
            writer.writerow(
                {
                    "entry_ts": _fmt_ts(t.get("entry_ts")),
                    "exit_ts": _fmt_ts(t.get("exit_ts")),
                    "side": t.get("side", ""),
                    "entry_price": _f(t.get("entry_price")),
                    "exit_price": _f(exit_price),
                    "qty": _f(t.get("qty")),
                    "exit_reason": t.get("exit_reason", ""),
                    "pnl_raw": _f(_pnl_raw(t)),
                    "regime": t.get("regime", ""),
                }
            )


def _write_daily_summary_csv(res: DemoLogResult, out: Path):
    # per-day aggregates
    day_stats: Dict[str, dict] = {}
    for t in res.trades:
        day = t["entry_ts"].strftime("%Y-%m-%d")
        st = day_stats.setdefault(
            day,
            {"entries": 0, "exits": 0, "wins": 0, "pnl": 0.0, "regimes": {}},
        )
        st["entries"] += 1
        if t.get("exit_ts"):
            st["exits"] += 1
        pnl = _pnl_raw(t)
        if pnl is not None:
            st["pnl"] += pnl
            if pnl > 0:
                st["wins"] += 1

    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DAILY_SUMMARY_HEADER)
        writer.writeheader()
        for day in sorted(day_stats):
            st = day_stats[day]
            gates = res.daily_gates.get(day, {k: 0 for k in GATE_KEYS})
            reg = res.regime_distribution.get(day, {k: 0.0 for k in REGIME_KEYS})
            writer.writerow(
                {
                    "date": day,
                    "entries": st["entries"],
                    "exits": st["exits"],
                    "win_rate": (st["wins"] / st["exits"]) if st["exits"] else 0.0,
                    "pnl_total": round(st["pnl"], 6),
                    "vol_flt": gates.get("vol_flt", 0),
                    "htf_trd": gates.get("htf_trd", 0),
                    "adapt_thr": gates.get("adapt_thr", 0),
                    "riskguard": gates.get("riskguard", 0),
                    "chop": gates.get("chop", 0),
                    "veto": gates.get("veto", 0),
                    "regime_trend_pct": reg.get("trend", 0.0),
                    "regime_chop_pct": reg.get("chop", 0.0),
                    "regime_highvol_pct": reg.get("high_vol", 0.0),
                }
            )


def _collect_log_files(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("trading_*.log"))
    raise FileNotFoundError(f"No log file/dir at: {path}")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Parse daily trading logs into demo_trades.csv + demo_daily_summary.csv"
    )
    parser.add_argument("--log", required=True, help="log file or directory (glob trading_*.log)")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--out", default="trades", help="output directory (default: trades)")
    args = parser.parse_args(argv)

    files = _collect_log_files(args.log)
    res = parse_demo_logs(files, start_date=args.start_date, end_date=args.end_date)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_csv = out_dir / "demo_trades.csv"
    daily_csv = out_dir / "demo_daily_summary.csv"
    _write_trades_csv(res, trades_csv)
    _write_daily_summary_csv(res, daily_csv)

    print(f"Parsed {len(files)} log file(s)")
    print(f"  trades:       {len(res.trades)}  (unclosed: {res.unclosed_trades})")
    print(f"  days:         {len(res.daily_gates)}")
    print(f"  wrote:        {trades_csv}")
    print(f"  wrote:        {daily_csv}")


if __name__ == "__main__":
    main()