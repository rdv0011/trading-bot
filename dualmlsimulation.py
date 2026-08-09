import argparse
import csv
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from binance.client import Client
from joblib import Parallel, delayed
from tqdm import tqdm

from mltrainingcore import (
    TARGET_COLUMN,
    SIGNAL_COLUMN,
    build_feature_dataset,
    get_features,
    simulate_trades_core,
    calculate_metrics,
)
from mlio import (
    MODEL_DIR,
    download_historical_prices,
    load_featured_df,
    save_featured_df,
    load_labels,
    save_labels,
)
from tactical.tacticalml import TacticalML
from strategic.strategicml import StrategicML
from timeframe_config import TIMEFRAMES

warnings.filterwarnings("ignore")

USE_SAVED_FEATURED = False
USE_SAVED_PREDICTIONS = False

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_DAYS = 45
DEFAULT_TIMEFRAME = "15m"
TRAINING_FRACTION = 0.8

DEFAULT_PARAMS = {
    "stake_long_frac": 0.1,
    "stake_short_frac": 0.05,
    "stop_loss_frac": 0.02,
    "take_profit_frac": 0.04,
    "max_hold_hours": 4.0,
    "recommended_leverage": 1.0,
}


def _predict_chunk(df_full, features, tf_cfg, model_params, indices, pbar=None):
    from tactical.tacticalml import TacticalML as _TacticalML
    tactical = _TacticalML(model_params=model_params, tf_cfg=tf_cfg)
    window = tf_cfg.max_history_candles
    results = []
    for i in indices:
        df_train = df_full.iloc[i - window : i - 1]
        df_pred_row = df_full.iloc[[i]]
        sig = tactical.fit_and_predict(df_train, df_pred_row, features)
        results.append((i, sig.prediction))
        if pbar is not None:
            pbar.update(1)
    return results


def _rolling_tactical_predictions(df_full: pd.DataFrame, tf_cfg) -> pd.DataFrame:
    # iterations=100 (vs 300 live) is sufficient for offline backtest and gives ~3x speedup.
    # thread_count=1 prevents core thrashing when N workers each spawn N CatBoost threads.
    model_params = {"iterations": 100, "verbose": False, "thread_count": 1}

    features = get_features(df_full)
    n = len(df_full)
    window = tf_cfg.max_history_candles

    if window >= n:
        window = max(50, n // 3)
        print(f"[TacticalML] Auto-adjusted window={window} for dataset length={n}")

    all_indices = list(range(window, n))
    n_jobs = min(os.cpu_count() or 4, 8)
    chunks = [arr.tolist() for arr in np.array_split(all_indices, n_jobs)]

    print(f"Running {len(all_indices)} walk-forward predictions on {n_jobs} workers...")

    with tqdm(total=len(all_indices), desc="Walk-forward predictions", unit="row") as pbar:
        results_nested = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_predict_chunk)(df_full, features, tf_cfg, model_params, chunk, pbar)
            for chunk in chunks
        )

    flat = sorted(
        [item for chunk_result in results_nested for item in chunk_result],
        key=lambda x: x[0],
    )
    preds = [p for _, p in flat]

    df_out = df_full.iloc[window:].copy()
    df_out[SIGNAL_COLUMN] = preds
    return df_out.round(5)


def _build_strategic_param_list(
    df_test: pd.DataFrame,
    df_raw_5m: pd.DataFrame,
    strategic: StrategicML,
    strategic_tf_cfg,
) -> list:
    if not strategic.is_ready:
        print("WARNING: StrategicML has no model loaded — using DEFAULT_PARAMS for all test candles.")
        return [DEFAULT_PARAMS.copy() for _ in range(len(df_test))]

    df_1h = (
        df_raw_5m.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )

    max_hist = strategic_tf_cfg.max_history_candles
    min_rows_for_features = 250
    hist_window = max(max_hist, min_rows_for_features)
    param_list = []

    for ts in tqdm(df_test.index, desc="Strategic decisions", unit="candle"):
        df_hist_1h = df_1h[df_1h.index <= ts].tail(hist_window)

        if len(df_hist_1h) < min_rows_for_features:
            param_list.append(DEFAULT_PARAMS.copy())
            continue

        try:
            decision = strategic.predict(df_hist_1h)
        except Exception as exc:
            print(f"WARNING: StrategicML.predict failed at {ts}: {exc} — using defaults")
            param_list.append(DEFAULT_PARAMS.copy())
            continue

        params = {
            "stake_long_frac": decision.stake_long_frac if decision.allow_trading else 0.0,
            "stake_short_frac": decision.stake_short_frac if decision.allow_trading else 0.0,
            "stop_loss_frac": decision.stop_loss_frac,
            "take_profit_frac": decision.take_profit_frac,
            "max_hold_hours": decision.max_hold_hours,
            "recommended_leverage": decision.recommended_leverage,
        }
        param_list.append(params)

    return param_list


def run_predictions_only(
    symbol: str,
    days: int,
    timeframe: str,
) -> tuple:
    tf_cfg = TIMEFRAMES[timeframe]

    featured_file = f"dual_{symbol}_{timeframe}_{days}d_featured.csv"
    df_full = load_featured_df(featured_file) if USE_SAVED_FEATURED else None

    client = Client()
    df_raw = download_historical_prices(symbol, tf_cfg.binance_interval, days, client)

    if df_full is None:
        df_full = build_feature_dataset(df_raw, tf_cfg)
        save_featured_df(df_full, featured_file)
        print(f"Features saved: {featured_file}")

    print(f"Full featured dataset: {len(df_full)} rows")

    pred_file = f"dual_{symbol}_{timeframe}_{days}d_predictions.csv"
    df_predictions = load_featured_df(pred_file) if USE_SAVED_PREDICTIONS else None

    if df_predictions is None:
        df_predictions = _rolling_tactical_predictions(df_full, tf_cfg)
        save_featured_df(df_predictions, pred_file)
        print(f"Predictions saved: {pred_file}")

    print(f"Predictions dataset: {len(df_predictions)} rows")
    return df_predictions, df_raw


def run_simulation(symbol: str, days: int, timeframe: str, model_dir=MODEL_DIR):
    tf_cfg = TIMEFRAMES[timeframe]
    strategic_tf_cfg = TIMEFRAMES["1h"]

    df_predictions, df_raw = run_predictions_only(symbol, days, timeframe)

    n_total = len(df_predictions)
    n_train = int(np.floor(n_total * TRAINING_FRACTION))
    df_test = df_predictions.iloc[n_train:].copy()
    test_start = df_test.index[0]
    test_end = df_test.index[-1]
    test_calendar_days = (test_end - test_start).total_seconds() / 86400.0
    print(
        f"Split: total={n_total}, train={n_train}, test={len(df_test)} "
        f"({test_start.date()} to {test_end.date()}, "
        f"{test_calendar_days:.2f} calendar days)"
    )

    strategic = StrategicML(model_dir=model_dir, tf_cfg=strategic_tf_cfg)
    param_list = _build_strategic_param_list(df_test, df_raw, strategic, strategic_tf_cfg)

    df_hist = df_predictions.iloc[: tf_cfg.adaptive_history_candles].copy()

    df_result, metrics = simulate_trades_core(
        df=df_test,
        df_hist=df_hist,
        signal_col=SIGNAL_COLUMN,
        tf_cfg=tf_cfg,
        param_list=param_list,
        close_col="close",
    )

    sim_file = f"dual_{symbol}_{timeframe}_{days}d_final_test_sim.csv"
    save_labels(df_result, sim_file)
    print(f"Simulation saved: {sim_file}")

    trades = df_result.attrs.get("trades", [])
    _, full_metrics = calculate_metrics(trades, metrics.get("final_wallet", 1.0))

    total_return = full_metrics.get('final_wallet', 1.0) - 1.0
    annualized_return = (1.0 + total_return) ** (365.0 / test_calendar_days) - 1.0

    print("\n" + "=" * 60)
    print("DUAL-ML SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Test window:       {test_calendar_days:.2f} days")
    print(f"  Test period:       {test_start.date()} to {test_end.date()}")
    print(f"  Final wallet:      {full_metrics.get('final_wallet', 1.0):.4f}")
    print(f"  Total return:      {total_return * 100:.2f}%")
    print(f"  Annualized return: {annualized_return * 100:.2f}%")
    print(f"  Trades:            {full_metrics.get('trades_count', 0)}")
    print(f"  Win rate:          {full_metrics.get('win_rate', 0.0):.2%}")
    print(f"  Mean return/trade: {full_metrics.get('mean_return', 0.0):.4%}")
    print(f"  Objective score:   {full_metrics.get('objective_score', 0.0):.4f}")
    print("=" * 60)

    return df_result, full_metrics


# Schema mirrors demo_log_parser.TRADES_HEADER / DAILY_SUMMARY_HEADER so
# compare_demo_vs_sim.py can diff the two sides directly.
SIM_TRADES_HEADER = [
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
SIM_DAILY_HEADER = [
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
GATE_KEYS = ("vol_flt", "htf_trd", "adapt_thr", "riskguard", "chop", "veto")
REGIME_KEYS = ("trend", "chop", "high_vol")


def _sim_side(position: int) -> str:
    return "LONG" if position == 1 else "SHORT"


def _write_sim_trades_csv(trades: list, out: Path) -> None:
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SIM_TRADES_HEADER)
        writer.writeheader()
        for t in trades:
            writer.writerow(
                {
                    "entry_ts": t["entry_timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_ts": t["exit_timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "side": _sim_side(t["position"]),
                    "entry_price": f"{t['entry_price']:.8f}".rstrip("0").rstrip("."),
                    "exit_price": f"{t['exit_price']:.8f}".rstrip("0").rstrip("."),
                    "qty": f"{t.get('stake_frac', 0.0):.8f}".rstrip("0").rstrip("."),
                    "exit_reason": t.get("exit_reason", ""),
                    "pnl_raw": f"{t.get('return', 0.0):.8f}".rstrip("0").rstrip("."),
                    "regime": t.get("regime", ""),
                }
            )


def _write_sim_daily_summary(df_test: pd.DataFrame, trades: list, out: Path) -> None:
    entry_days = {}
    for t in trades:
        day = t["entry_timestamp"].strftime("%Y-%m-%d")
        st = entry_days.setdefault(day, {"entries": 0, "exits": 0, "wins": 0, "pnl": 0.0})
        st["entries"] += 1
        st["exits"] += 1
        ret = t.get("return", 0.0)
        st["pnl"] += ret
        if ret > 0:
            st["wins"] += 1

    regime_days = {}
    if "regime" in df_test.columns:
        for day, grp in df_test.groupby(df_test.index.date):
            counts = grp["regime"].value_counts(normalize=True).to_dict()
            regime_days[day.strftime("%Y-%m-%d")] = {
                k: round(counts.get(k, 0.0) * 100.0, 1) for k in REGIME_KEYS
            }

    days = sorted(set(entry_days) | set(regime_days))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SIM_DAILY_HEADER)
        writer.writeheader()
        for day in days:
            st = entry_days.get(day, {"entries": 0, "exits": 0, "wins": 0, "pnl": 0.0})
            reg = regime_days.get(day, {k: 0.0 for k in REGIME_KEYS})
            writer.writerow(
                {
                    "date": day,
                    "entries": st["entries"],
                    "exits": st["exits"],
                    "win_rate": (st["wins"] / st["exits"]) if st["exits"] else 0.0,
                    "pnl_total": round(st["pnl"], 6),
                    "vol_flt": 0,
                    "htf_trd": 0,
                    "adapt_thr": 0,
                    "riskguard": 0,
                    "chop": 0,
                    "veto": 0,
                    "regime_trend_pct": reg.get("trend", 0.0),
                    "regime_chop_pct": reg.get("chop", 0.0),
                    "regime_highvol_pct": reg.get("high_vol", 0.0),
                }
            )


def run_windowed_simulation(
    symbol: str,
    days: int,
    timeframe: str,
    start_date: str,
    end_date: str,
    model_dir=MODEL_DIR,
    live_faithful: bool = False,
) -> tuple:
    """Simulate only [start_date, end_date] UTC (inclusive of 15m candles).

    Warmup history (adaptive thresholds + strategic 1h model) comes from the
    fetch period BEFORE the window, mirroring live's accumulated prediction
    history.  live_faithful=True applies the Task-1 gates to mirror live:
    chop hard block, volume 0.8xSMA20, HTF-EMA50 trend filter, RiskGuard 5%.
    """
    tf_cfg = TIMEFRAMES[timeframe]
    strategic_tf_cfg = TIMEFRAMES["1h"]

    df_predictions, df_raw = run_predictions_only(symbol, days, timeframe)

    window_start = pd.Timestamp(start_date)
    window_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    df_hist = df_predictions[df_predictions.index < window_start].tail(
        tf_cfg.adaptive_history_candles
    )
    df_test = df_predictions[
        (df_predictions.index >= window_start) & (df_predictions.index < window_end)
    ].copy()

    if len(df_test) == 0:
        raise ValueError(
            f"No prediction rows in [{start_date}, {end_date}] — "
            f"is the fetch window (days={days}) large enough?"
        )

    print(f"Windowed sim: {start_date} .. {end_date} ({len(df_test)} candles, "
          f"hist={len(df_hist)} candles before window)")

    strategic = StrategicML(model_dir=model_dir, tf_cfg=strategic_tf_cfg)
    param_list = _build_strategic_param_list(df_test, df_raw, strategic, strategic_tf_cfg)

    sim_kwargs = {
        "df": df_test,
        "df_hist": df_hist,
        "signal_col": SIGNAL_COLUMN,
        "tf_cfg": tf_cfg,
        "param_list": param_list,
        "close_col": "close",
    }
    if live_faithful:
        htf_span = 50
        df_1h = df_raw.resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        df_1h[f"htf_ema{htf_span}"] = df_1h["close"].ewm(span=htf_span, adjust=False).mean()
        df_1h["htf_close"] = df_1h["close"]
        merge_cols = df_1h[["htf_close", f"htf_ema{htf_span}"]].ffill()
        df_test = df_test.join(merge_cols, how="left")

        sim_kwargs.update(
            {
                "regime_stake_mult": {"trend": 1.0, "high_vol": 0.5, "chop": 0.0},
                "volume_filter_threshold": 0.8,
                "htf_ema_span": htf_span,
                "max_daily_loss_frac": 0.05,
                "max_drawdown_frac": 0.15,
            }
        )

    df_result, metrics = simulate_trades_core(**sim_kwargs)

    trades = df_result.attrs.get("trades", [])
    mode = "live-faithful" if live_faithful else "raw"
    print(f"Windowed sim [{mode}]: {len(trades)} trades")

    out_dir = Path("trades")
    out_dir.mkdir(exist_ok=True)
    tag = f"{start_date}_{end_date}"
    trades_csv = out_dir / f"sim_trades_{tag}.csv"
    daily_csv = out_dir / f"sim_daily_summary_{tag}.csv"
    _write_sim_trades_csv(trades, trades_csv)
    _write_sim_daily_summary(df_test, trades, daily_csv)
    print(f"  wrote: {trades_csv}")
    print(f"  wrote: {daily_csv}")

    return df_result, metrics, df_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-ML walk-forward backtest.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument(
        "--timeframe", default=DEFAULT_TIMEFRAME, choices=list(TIMEFRAMES.keys())
    )
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--start-date", default=None, help="window start, YYYY-MM-DD UTC")
    parser.add_argument("--end-date", default=None, help="window end, YYYY-MM-DD UTC")
    parser.add_argument("--live-faithful", action="store_true",
                        help="mirror live gates: chop block, volume, HTF EMA, RiskGuard")
    args = parser.parse_args()

    if args.start_date and args.end_date:
        run_windowed_simulation(
            symbol=args.symbol,
            days=args.days,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            model_dir=args.model_dir,
            live_faithful=args.live_faithful,
        )
    else:
        run_simulation(
            symbol=args.symbol,
            days=args.days,
            timeframe=args.timeframe,
            model_dir=args.model_dir,
        )
