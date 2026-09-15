"""
compare_live_vs_sim.py — Diagnose live-vs-simulation discrepancies.

Goal: identify *where* live trading deviates from simulation by replaying the
same trained models over the exact historical window the live bot traded, then
aligning per-trade.

Why not `main.py compare`:
  - `compare.parse_demo_logs` regexes the DEBUG .log for a "TRADE | ENTER ..."
    format the current live logger does *not* emit (confirmed: 0 matches), so
    the demo side comes back empty.
  - `simulate` replays only the pre-split `_val.csv`, not an explicit
    live-aligned window.

This script instead reads live round-trips from the *structured* trade CSVs
(logs/trades_YYYY-MM-DD.csv), replays the SAME models/ over the live window,
and aligns per-trade by timestamp to surface entry/exit/PnL/meta differences.

Two replay modes (both useful, expose different discrepancy classes):
  --replay walk   : walk-forward retrain (mirrors simulate_mode exactly).
                    Comparing this to live isolates "retrain-vs-frozen-model"
                    discrepancies.
  --replay frozen : frozen loaded models (mirrors _live_iteration exactly).
                    Comparing this to live isolates pure data/execution diffs.

Usage:
  python compare_live_vs_sim.py \
      --live-csv logs/trades_2026-09-03.csv \
      --model-dir models/ \
      --replay walk \
      --start 2026-08-26T00:00:00 --end 2026-09-02T23:59:00 \
      --output logs/live_vs_sim_report
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is importable regardless of CWD (script lives here).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    SYMBOL,
    TIMEFRAME,
    STRATEGIC_TF,
    WALKFORWARD_RETRAIN_EVERY_CANDLES,
    STRATEGIC_TARGET_COLS,
    ABSOLUTE_THRESHOLD,
)

from data import (  # noqa: E402
    download_historical,
    make_features_df,
    make_labels_df,
    add_strategic_features_df,
    get_feature_cols,
)
from model import (  # noqa: E402
    CatBoostModel,
    rolling_tactical_predict,
    predict_strategic_meta_params,
)
from simulate import run_simulation  # noqa: E402


# ── Live trades loader ───────────────────────────────────────────────────
_LIVE_CSV_COLS = {
    "timestamp",
    "symbol",
    "side",
    "entry_price",
    "exit_price",
    "qty",
    "pnl",
    "leverage",
    "stop_loss",
    "take_profit",
    "max_hold_hours",
    "regime",
    "exit_reason",
    "tactical_pred",
    "strategic_params",
}


def load_live_trades(live_csv: str) -> pd.DataFrame:
    """Load live/demo round-trip trades from the structured trade CSV."""
    path = Path(live_csv)
    if not path.exists():
        raise FileNotFoundError(f"Live trade CSV not found: {path}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # A "live" trade has a realised exit (exit_reason set); drop open/sim rows.
    df = df[df["exit_reason"].notna() & df["exit_reason"].astype(str).str.lower().ne("open")]
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in _LIVE_CSV_COLS:
        if col not in df.columns:  # keep missing optional cols as NaN
            df[col] = np.nan
    return df


def infer_window(live: pd.DataFrame, buffer_days: float) -> tuple:
    """Infer the live window from trade timestamps, padded by buffer."""
    if live.empty:
        raise ValueError("No live trades to infer a window from")
    start = live["timestamp"].min() - pd.Timedelta(days=buffer_days)
    end = live["timestamp"].max() + pd.Timedelta(days=buffer_days)
    return start, end


# ── Replay: reproduce simulate_mode over an explicit window ─────────────
def build_replay_sim(
    model_dir: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    replay: str,
    tactical_window: int,
    testnet: bool = False,
    retrain_every: int = 100,
) -> tuple:
    """
    Replay the SAME models over [start, end] and run the sim engine.
    Returns (trades_df, metrics, equity_curve).

    testnet: replay candles from Binance Futures TESTNET instead of the live
    API. The live bot trades on testnet, so replaying testnet candles exposes
    genuine data/execution discrepancies instead of source-market divergence
    (testnet prices/volume differ materially from mainnet).
    """
    # 1. Fetch same historical candles from the source the live bot used.
    #    download_historical fetches 'days' counting back from now; we request a
    #    window big enough to cover [start,end] plus warmup, then clip.
    pad_days = int(np.ceil(tactical_window / 96)) + 5  # warmup padding at 15m
    span_days = int(np.ceil((end - start).total_seconds() / 86400)) + pad_days + 2
    raw_all = download_historical(
        symbol=SYMBOL, days=max(span_days, pad_days + 2), timeframe=TIMEFRAME,
        testnet=testnet,
    )
    clip_lo = start - pd.Timedelta(days=pad_days)
    raw = raw_all.loc[clip_lo: end]

    # 2. Load models.
    tactical_model = CatBoostModel(model_type="tactical", model_dir=model_dir)
    tactical_model.load()
    strategic_model = CatBoostModel(model_type="strategic", model_dir=model_dir)
    strategic_model.load()

    # 3. Build feature frames for each timeframe. add_strategic_features_df
    #    needs base features already present (ret1/atr14), so the 1h raw data
    #    must be feature-engineered first on its own timeframe.
    df_base_feat = make_features_df(raw, timeframe=TIMEFRAME)
    df_strat = add_strategic_features_df(
        make_features_df(raw, timeframe=STRATEGIC_TF),
        timeframe=STRATEGIC_TF,
    )

    # Tactical feature cols from the model metadata (authoritative, matches live).
    tact_features = tactical_model.metadata.get("feature_cols") or get_feature_cols(df_base_feat)
    strat_features = strategic_model.metadata.get("feature_cols") or [
        c for c in get_feature_cols(df_strat) if c not in STRATEGIC_TARGET_COLS
    ]

    # 4. Tactical predictions (walk-forward retrain vs frozen).
    if replay == "walk":
        # rolling_tactical_predict retrains on a rolling window using the target
        # column (future_ret), so the labeled frame is required for this path.
        df_base = make_labels_df(df_base_feat, timeframe=TIMEFRAME)
        tactical_preds = rolling_tactical_predict(
            df_base, tactical_model, tact_features,
            retrain_every=retrain_every,
            window=tactical_window,
        )
    else:  # frozen — mirrors _live_iteration (no retrain, no labels needed)
        df_base = df_base_feat
        preds = tactical_model.predict(df_base, tact_features)
        tactical_preds = pd.Series(np.asarray(preds).ravel(), index=df_base.index)

    # 5. Strategic meta-params (batch, mirrors simulate_mode).
    strategic_meta_params = predict_strategic_meta_params(
        df_strat, strategic_model, strat_features
    )

    # 6. Align strategic (1h) params onto tactical (15m) rows via ffill.
    strat_index = df_strat.index
    strat_params_df = pd.DataFrame(strategic_meta_params, index=strat_index)
    strat_params_reindexed = strat_params_df.reindex(df_base.index, method="ffill")
    param_defaults = {
        "stake_long_frac": 0.1, "stake_short_frac": 0.05,
        "stop_loss_frac": 0.02, "take_profit_frac": 0.04,
        "max_hold_hours": 4.0, "recommended_leverage": 1.0,
        "max_exposure_frac": 1.0, "regime": "trend",
    }
    aligned_meta_params = [
        row.to_dict()
        for _, row in strat_params_reindexed.fillna(param_defaults).iterrows()
    ]

    # 7. Run simulation engine.
    trades_df, metrics, equity_curve = run_simulation(
        df_base, tactical_preds, aligned_meta_params,
    )
    return trades_df, metrics, equity_curve


# ── Alignment & discrepancy report ─────────────────────────────────────
def align_and_report(
    live: pd.DataFrame,
    sim: pd.DataFrame,
    tolerance_seconds: int,
    output: str,
):
    """Align live vs sim trades by timestamp; write discrepancy CSV + HTML."""
    live_s = live.sort_values("timestamp").reset_index(drop=True)
    sim_s = sim.sort_values("timestamp").reset_index(drop=True)

    sim_entries = sim_s["timestamp"].to_numpy(dtype="datetime64[s]")
    rows = []
    for _, lrow in live_s.iterrows():
        lt = np.datetime64(lrow["timestamp"].to_pydatetime(), "s")
        diffs = np.abs((sim_entries - lt).astype(np.int64))
        best_i = int(np.argmin(diffs)) if len(sim_entries) else -1
        best_diff = float(diffs[best_i]) if len(sim_entries) else np.inf

        row = {
            "live_ts": str(lrow["timestamp"]),
            "live_side": lrow.get("side"),
            "live_entry": _num(lrow, "entry_price"),
            "live_exit": _num(lrow, "exit_price"),
            "live_pnl": _num(lrow, "pnl"),
            "live_reason": lrow.get("exit_reason"),
            "live_leverage": _num(lrow, "leverage"),
        }
        if best_i >= 0 and best_diff <= tolerance_seconds:
            srow = sim_s.iloc[best_i]
            row.update({
                "match": True,
                "time_diff_s": best_diff,
                "sim_ts": str(srow["timestamp"]),
                "sim_side": srow.get("side"),
                "sim_entry": _num(srow, "entry_price"),
                "sim_exit": _num(srow, "exit_price"),
                "sim_pnl": _num(srow, "pnl"),
                "sim_reason": srow.get("exit_reason"),
                "sim_leverage": _num(srow, "leverage"),
                "entry_diff": (_num(srow, "entry_price") or 0.0) - (_num(lrow, "entry_price") or 0.0),
                "exit_diff": (_num(srow, "exit_price") or 0.0) - (_num(lrow, "exit_price") or 0.0),
                "pnl_diff": (_num(srow, "pnl") or 0.0) - (_num(lrow, "pnl") or 0.0),
                "side_match": str(srow.get("side")).lower() == str(lrow.get("side")).lower(),
            })
        else:
            row.update({
                "match": False,
                "time_diff_s": best_diff if len(sim_entries) else None,
                "sim_ts": None, "sim_side": None, "sim_entry": None,
                "sim_exit": None, "sim_pnl": None, "sim_reason": None,
                "sim_leverage": None, "entry_diff": None, "exit_diff": None,
                "pnl_diff": None, "side_match": None,
            })
        rows.append(row)

    rep = pd.DataFrame(rows)

    # Summary stats
    matched = rep[rep["match"]]
    n_match = len(matched)
    side_mismatch = int(matched["side_match"].eq(False).sum()) if n_match else 0
    entry_mad = float(matched["entry_diff"].abs().mean()) if n_match else float("nan")
    exit_mad = float(matched["exit_diff"].abs().mean()) if n_match else float("nan")
    pnl_diff = float(matched["pnl_diff"].sum()) if n_match else 0.0
    unmatched_live = int(rep["match"].eq(False).sum())

    fmt = os.path.splitext(output)[0]
    csv_path = f"{fmt}.csv"
    html_path = f"{fmt}.html"
    rep.to_csv(csv_path, index=False)

    _write_html(
        html_path,
        {
            "n_live": len(live_s),
            "n_sim": len(sim_s),
            "n_match": n_match,
            "unmatched_live": unmatched_live,
            "side_mismatch": side_mismatch,
            "entry_mad": entry_mad,
            "exit_mad": exit_mad,
            "pnl_diff": pnl_diff,
        },
        rep,
    )
    return rep, csv_path, html_path


def _num(row, col):
    v = row.get(col)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _write_html(path, stats, rep):
    rows_html = ""
    for _, r in rep.iterrows():
        cls = "miss" if not r["match"] else ("mis" if (r["side_match"] is False) else "ok")
        rows_html += (
            "<tr><td>" + str(r["live_ts"]) + "</td>"
            + f"<td>{r['live_side']}</td>"
            + f"<td>{'' if r['live_entry'] is None else round(r['live_entry'],2)}</td>"
            + f"<td>{'' if r['live_exit'] is None else round(r['live_exit'],2)}</td>"
            + f"<td>{'' if r['live_pnl'] is None else round(r['live_pnl'],6)}</td>"
            + f"<td>{r['live_reason']}</td>"
            + "<td class='" + cls + "'>" + ("MATCH" if r["match"] else "NO-SIM") + "</td>"
            + f"<td>{'' if r['sim_side'] is None else r['sim_side']}</td>"
            + f"<td>{'' if r['sim_entry'] is None else round(r['sim_entry'],2)}</td>"
            + f"<td>{'' if r['sim_exit'] is None else round(r['sim_exit'],2)}</td>"
            + f"<td>{'' if r['sim_pnl'] is None else round(r['sim_pnl'],6)}</td>"
            + f"<td>{'' if r['entry_diff'] is None else round(r['entry_diff'],2)}</td>"
            + f"<td>{'' if r['pnl_diff'] is None else round(r['pnl_diff'],6)}</td>"
            + "</tr>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Live vs Simulation Discrepancies</title><style>
body{{font-family:Arial,sans-serif;margin:20px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{border:1px solid #ddd;padding:5px;text-align:right}}
th{{background:#f2f2f2}}
td:nth-child(1),td:nth-child(13){{text-align:left}}
.ok{{color:green}} .mis{{color:orange;font-weight:bold}} .miss{{color:red;font-weight:bold}}
.metric-box{{display:inline-block;margin:6px;padding:10px;background:#f9f9f9;border-radius:5px}}
.ml{{font-size:11px;color:#666}} .mv{{font-size:20px;font-weight:bold}}
</style></head><body>
<h1>Live vs Simulation Discrepancy Report</h1>
<div class="metric-box"><div class="ml">Live trades</div><div class="mv">{stats['n_live']}</div></div>
<div class="metric-box"><div class="ml">Sim trades</div><div class="mv">{stats['n_sim']}</div></div>
<div class="metric-box"><div class="ml">Aligned pairs</div><div class="mv">{stats['n_match']}</div></div>
<div class="metric-box"><div class="ml">Unmatched live</div><div class="mv">{stats['unmatched_live']}</div></div>
<div class="metric-box"><div class="ml">Side mismatches</div><div class="mv">{stats['side_mismatch']}</div></div>
<div class="metric-box"><div class="ml">Entry MAD</div><div class="mv">{stats['entry_mad']:.4f}</div></div>
<div class="metric-box"><div class="ml">Exit MAD</div><div class="mv">{stats['exit_mad']:.4f}</div></div>
<div class="metric-box"><div class="ml">Sum PnL diff</div><div class="mv">{stats['pnl_diff']:.6f}</div></div>
<h2>Per-trade breakdown</h2>
<p>Legend: <span class="ok">OK</span>=aligned+side match ·
<span class="mis">SIDE</span>=aligned but side differs ·
<span class="miss">NO-SIM</span>=no sim trade within tolerance</p>
<table><tr><th>Live ts</th><th>Live side</th><th>Live entry</th><th>Live exit</th>
<th>Live pnl</th><th>Reason</th><th>Match</th><th>Sim side</th><th>Sim entry</th>
<th>Sim exit</th><th>Sim pnl</th><th>Entry diff</th><th>PnL diff</th></tr>
{rows_html}</table></body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Compare live trading vs simulation replay")
    ap.add_argument("--live-csv", required=True, help="Structured live trade CSV (logs/trades_*.csv)")
    ap.add_argument("--model-dir", default="models", help="Model directory")
    ap.add_argument("--replay", choices=["walk", "frozen"], default="walk",
                    help="walk=mirror simulate (retrain); frozen=mirror live (frozen models)")
    ap.add_argument("--start", default=None, help="Override window start (ISO)")
    ap.add_argument("--end", default=None, help="Override window end (ISO)")
    ap.add_argument("--buffer-days", type=float, default=0.0, help="Padding past trade window")
    ap.add_argument("--toleranc-s", type=int, default=900, help="Match tolerance seconds")
    ap.add_argument("--tactical-window", type=int, default=500, help="Walk-forward window")
    ap.add_argument("--retrain-every", type=int, default=None,
                    help=f"Walk-forward retrain cadence in candles (default {WALKFORWARD_RETRAIN_EVERY_CANDLES}, mirrors live's ~10 min retrain)")
    ap.add_argument("--testnet", action="store_true",
                    help="Replay TESTNET candles (matches live bot data source)")
    ap.add_argument("--output", default="logs/live_vs_sim_report", help="Report prefix")
    args = ap.parse_args()

    live = load_live_trades(args.live_csv)
    print(f"\nLoaded {len(live)} live trades from {args.live_csv}")

    # Window
    if args.start or args.end:
        start = pd.Timestamp(args.start) if args.start else live["timestamp"].min()
        end = pd.Timestamp(args.end) if args.end else live["timestamp"].max()
    else:
        start, end = infer_window(live, args.buffer_days)
    print(f"Replay window: {start} -> {end}")

    # Replay sim
    sim, metrics, _ = build_replay_sim(
        model_dir=args.model_dir,
        start=start,
        end=end,
        replay=args.replay,
        tactical_window=args.tactical_window,
        testnet=args.testnet,
        retrain_every=args.retrain_every if args.retrain_every else WALKFORWARD_RETRAIN_EVERY_CANDLES,
    )
    print(f"\nSim replay produced {len(sim)} trades")

    # Dump the full raw sim trade list too, so sim-only trades (that never
    # align) are visible alongside the matched rows in the discrepancy CSV.
    fmt = os.path.splitext(args.output)[0]
    sim_path = f"{fmt}_simtrades.csv"
    sim.to_csv(sim_path, index=False)
    print(f"Sim trades CSV : {sim_path}")

    # Align & report
    rep, csv_path, html_path = align_and_report(
        live, sim, args.toleranc_s, args.output
    )
    print(f"\nDiscrepancy CSV : {csv_path}")
    print(f"HTML report     : {html_path}")
    print(f"Summary: {len(rep)} live trades, "
          f"{int(rep['match'].sum())} aligned, "
          f"{int(rep['match'].eq(False).sum())} unmatched")


if __name__ == "__main__":
    main()