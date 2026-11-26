#!/usr/bin/env python3
"""
Dual-ML Bitcoin Trading Bot - Main Entry Point.

Modes:
  - train: Train tactical + strategic models on historical data
  - simulate: Run backtest on validation data
  - live: Run live trading on Binance testnet (reads API keys from .env)
  - compare: Compare simulation vs demo trading logs
"""

import argparse
import sys
import time
import pandas as pd
from datetime import datetime

from config import (
    get_broker_config, is_testnet_env,
    SYMBOL, TIMEFRAME,
    HISTORY_DAYS, TRAIN_FRACTION,
    FEATURE_LAGS, EMA_SPANS, ATR_PERIOD,
    TACTICAL_MODEL_PARAMS,
    MIN_TREES,
    INITIAL_EQUITY, FEE, SLIPPAGE,
    WALKFORWARD_RETRAIN_EVERY_CANDLES, ABSOLUTE_THRESHOLD,
    LIQUIDITY_RECORDER_ENABLED,
)

from logger import setup_logging, log_info, log_error
from orderbook import estimate_initial_trade_intensity
from data import run_full_pipeline, load_featured_df
from model import (
    CatBoostModel, rolling_tactical_predict
)
from simulate import run_simulation
from compare import run_comparison

MODEL_DIR = "models"


def train_mode(args):
    """Train tactical model on historical data."""
    log_info("=" * 60)
    log_info("TRAINING MODE")
    log_info("=" * 60)

    # Run data pipeline for tactical timeframe
    log_info(f"\n--- Training Tactical Model ({TIMEFRAME}) ---")
    df_train_tactical, df_val_tactical = run_full_pipeline(
        symbol=args.symbol,
        whole_days=args.days,
        timeframe=TIMEFRAME,
        train_frac=TRAIN_FRACTION,
    )

    # Get feature columns
    from data import get_feature_cols
    feature_cols = get_feature_cols(df_train_tactical)
    log_info(f"Feature columns: {len(feature_cols)} features")

    # Train tactical model
    log_info("\n--- Training Tactical Model ---")
    tactical_model = CatBoostModel(
        model_type="tactical",
        model_params=TACTICAL_MODEL_PARAMS
    )
    tactical_model.train(
        df_train_tactical,
        feature_cols,
        target_col="future_ret",
        save=True,
    )

    log_info("\n" + "=" * 60)
    log_info("TRAINING COMPLETE")
    log_info(f"Tactical model: models/model_tactical.cbm")
    log_info("=" * 60)


def simulate_mode(args):
    """Run simulation on validation data."""
    log_info("=" * 60)
    log_info("SIMULATION MODE")
    log_info("=" * 60)

    # Load tactical model
    log_info("Loading models...")
    tactical_model = CatBoostModel(model_type="tactical", model_dir=args.model_dir)
    tactical_model.load()

    # Pre-flight validation: reject underfit models
    tree_count = getattr(tactical_model.model, "tree_count_", None)
    if tree_count is not None and tree_count < 10:
        log_error(f"Model underfit: only {tree_count} trees. Minimum {MIN_TREES} required. Retrain required.")
        sys.exit(1)
    elif tree_count is not None:
        log_info(f"Tactical model tree count: {tree_count}")

    # Load validation data
    log_info("Loading validation data...")
    df_val = load_featured_df(f"df_{SYMBOL}_{TIMEFRAME}_val.csv")

    if df_val is None or df_val.empty:
        log_error("No validation data found. Run 'train' mode first.")
        sys.exit(1)

    # Get feature columns
    from data import get_feature_cols
    feature_cols = get_feature_cols(df_val)
    log_info(f"Validation data: {len(df_val)} candles")
    log_info(f"Features: {len(feature_cols)} columns")

    # Run tactical walk-forward predictions
    log_info("\n--- Running Tactical Walk-Forward Predictions ---")

    # Walk-forward window must fit within available validation data so the
    # simulation covers most of the period, not just the tail.
    wf_window = min(500, max(50, len(df_val) // 4))
    log_info(f"Walk-forward window: {wf_window} candles (data: {len(df_val)})")
    tactical_preds = rolling_tactical_predict(
        df_val,
        tactical_model,
        feature_cols,
        retrain_every=WALKFORWARD_RETRAIN_EVERY_CANDLES,
        window=wf_window,
    )

    # Generate meta-params heuristically (no strategic model)
    log_info("\n--- Generating Meta-Params (Heuristic) ---")
    from model import META_PARAM_BOUNDS, _clamp_float
    aligned_meta_params = []
    for _, row in df_val.iterrows():
        vol_12 = row.get("vol_12", 0.0)
        vol_48 = row.get("vol_48", 0.0)
        vol_ratio = vol_12 / max(vol_48, 1e-8)
        abs_vol_ratio = row.get("abs_vol_ratio", 1.0)
        atr14 = row.get("atr14", 0.0)
        close = row.get("close", 1.0)
        atr_pct = atr14 / max(close, 1e-8)

        if vol_ratio < 1.0:
            regime = "trend"
        elif vol_ratio < 1.6:
            regime = "high_vol"
        else:
            regime = "chop"

        base = {
            "trend":     dict(leverage=5.0, stake_long=0.20, stake_short=0.10, sl=0.015, hold=8.0),
            "high_vol":  dict(leverage=2.0, stake_long=0.10, stake_short=0.05, sl=0.030, hold=2.0),
            "chop":      dict(leverage=1.0, stake_long=0.10, stake_short=0.05, sl=0.020, hold=4.0),
        }[regime]

        vol_scale = min(1.0, 1.0 / max(abs_vol_ratio, 0.5))
        sl_frac = max(base["sl"], min(1.5 * atr_pct, 0.05))
        tp_frac = sl_frac * 2.0

        if vol_ratio > 1.6:
            max_exposure = 0.4
        elif vol_ratio > 1.0:
            max_exposure = 0.7
        else:
            max_exposure = 1.0

        meta = {
            "recommended_leverage": base["leverage"],
            "max_exposure_frac": max_exposure,
            "stake_long_frac": base["stake_long"] * vol_scale,
            "stake_short_frac": base["stake_short"] * vol_scale,
            "stop_loss_frac": sl_frac,
            "take_profit_frac": tp_frac,
            "max_hold_hours": base["hold"],
            "regime": regime,
        }
        aligned_meta_params.append(meta)

    # Run simulation
    log_info("\n--- Running Simulation ---")
    trades_df, metrics, equity_curve = run_simulation(
        df_val,
        tactical_preds,
        aligned_meta_params,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_path = f"logs/trades_sim_{timestamp}.csv"
    equity_path = f"logs/equity_sim_{timestamp}.csv"

    from utils import save_trades_csv
    save_trades_csv(trades_df, trades_path)
    equity_curve.to_csv(equity_path, index=False)

    log_info(f"\nTrades saved: {trades_path}")
    log_info(f"Equity curve saved: {equity_path}")


def live_mode(args):
    """Run live trading (testnet=debug by default, mainnet when TRADING_ENV=mainnet)."""
    trading_env = "TESTNET" if is_testnet_env() else "MAINNET"
    log_info("=" * 60)
    log_info(f"LIVE TRADING MODE ({trading_env})")
    log_info("=" * 60)

    # Load tactical model only (no strategic model)
    log_info("Loading tactical model...")
    tactical_model = CatBoostModel(model_type="tactical", model_dir=args.model_dir)
    tactical_model.load()

    # Pre-flight validation: reject underfit models
    tree_count = getattr(tactical_model.model, "tree_count_", None)
    if tree_count is not None and tree_count < 10:
        log_error(f"Model underfit: only {tree_count} trees. Minimum {MIN_TREES} required. Retrain required.")
        sys.exit(1)
    elif tree_count is not None:
        log_info(f"Tactical model tree count: {tree_count}")

    # Create broker (loads credentials from .env via config module).
    # TRADING_ENV controls testnet (debug) vs mainnet (production); a mainnet
    # incident can be debugged by flipping TRADING_ENV=testnet (same code path,
    # no real money at risk).
    from broker import BinanceBroker
    use_testnet = is_testnet_env()
    broker_config = get_broker_config("futures", testnet=use_testnet)
    broker = BinanceBroker(
        api_key=broker_config["api_key"],
        api_secret=broker_config["api_secret"],
        testnet=use_testnet,
        symbol=SYMBOL,
    )

    # Feature columns must match the trained tactical model. Derive them from
    # the model's own metadata (authoritative) rather than a raw OHLCV sample.
    feature_cols = list(tactical_model.metadata.get("feature_cols", []))
    if not feature_cols:
        log_error("Tactical model metadata has no feature_cols; aborting live startup")
        sys.exit(1)

    # Probe connectivity; the initial fetch may transiently fail (e.g. rate
    # limit / network), so retry with backoff before giving up on startup.
    # 7 days of 1h candles = 7 * 24 = 168 candles
    candles_7d = 7 * 24
    df_sample = None
    for _attempt in range(5):
        df_sample = broker.get_historical_prices(SYMBOL, candles_7d, TIMEFRAME)
        if df_sample is not None and len(df_sample) > 0:
            break
        log_error(
            f"Initial historical fetch returned no data (attempt {_attempt + 1}), retrying..."
        )
        time.sleep(5 * (2 ** _attempt))
    if df_sample is None or len(df_sample) == 0:
        log_error("Could not fetch initial market data; aborting live startup")
        sys.exit(1)

    # Phase 1-2: optional mainnet WS orderbook recorder feeding liquidity
    # gates. Fail-open: if it dies, strategy gates log and proceed (no halt).
    liquidity_monitor = None
    if LIQUIDITY_RECORDER_ENABLED:
        from orderbook import OrderBookRecorder
        liquidity_monitor = OrderBookRecorder(symbol=SYMBOL)
        liquidity_monitor.start()
        log_info("Liquidity recorder started (mainnet WS orderbook)")

    # Create strategy (no strategic model)
    from strategy import DualMLStrategy
    strategy = DualMLStrategy(
        broker=broker,
        tactical_model=tactical_model,
        feature_cols=feature_cols,
        model_dir=args.model_dir,
        liquidity_monitor=liquidity_monitor,
    )

    # Estimate initial trade intensity from recent aggTrades and seed
    # the adaptive liquidity threshold history. This gives the threshold
    # a realistic starting value instead of using the hardcoded fallback.
    if LIQUIDITY_RECORDER_ENABLED:
        initial_intensity = estimate_initial_trade_intensity(limit=1000)
        if initial_intensity is not None:
            strategy.seed_trade_intensity_history(initial_intensity, num_samples=50)
        else:
            log_warning("Could not estimate initial trade intensity; adaptive threshold will use fallback")

    # Seed ATR history from the initial klines fetch for adaptive trailing threshold
    if df_sample is not None and len(df_sample) > 50:
        strategy.seed_atr_history(df_sample, num_samples=50)
    else:
        log_warning("Insufficient klines data for ATR history seeding")

    # Seed vol_ratio history from the initial klines fetch for adaptive extreme vol threshold
    if df_sample is not None and len(df_sample) > 60:
        strategy.seed_vol_history(df_sample, num_samples=50)
    else:
        log_warning("Insufficient klines data for vol_ratio history seeding")

    # Seed spread/depth history from REST snapshots for adaptive liquidity gates
    strategy.seed_spread_depth_history(num_samples=20)

    # Adopt any position left open by a previous process so exits manage it
    # instead of trading blindly beside it (refuse to start if disabled).
    if not strategy.sync_exchange_position():
        log_error("Refusing to start: open exchange position exists and ADOPT_EXISTING_POSITION=False")
        sys.exit(1)

    log_info(f"Starting live loop (sleep: {args.sleep}s, max: {args.max_iterations})")
    log_info("Press Ctrl+C to stop")

    try:
        strategy.run_live_loop(
            sleep_seconds=args.sleep,
            max_iterations=args.max_iterations,
        )
    finally:
        if liquidity_monitor is not None:
            liquidity_monitor.stop()
            log_info("Liquidity recorder stopped")


def compare_mode(args):
    """Compare simulation vs demo trading logs."""
    log_info("=" * 60)
    log_info("COMPARISON MODE: Sim vs Demo")
    log_info("=" * 60)

    metrics = run_comparison(
        log_dir=args.log_dir,
        demo_pattern=args.demo_pattern,
        sim_pattern=args.sim_pattern,
        output_html=args.output,
    )

    if not metrics:
        log_error("No trades found for comparison")
        sys.exit(1)

    log_info("\nComparison complete!")
    log_info(f"Report: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Dual-ML Bitcoin Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train
  python main.py simulate
  python main.py live --sleep 60
  python main.py compare --output logs/report.html
        """
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")
    subparsers.required = True

    # ── Train Mode ──────────────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Train models on historical data")
    train_parser.add_argument("--symbol", default=SYMBOL, help="Trading symbol")
    train_parser.add_argument("--days", type=int, default=HISTORY_DAYS, help="Historical days")

    # ── Simulate Mode ───────────────────────────────────────────────────
    sim_parser = subparsers.add_parser("simulate", help="Run backtest simulation")
    sim_parser.add_argument("--model-dir", default="models", help="Model directory")
    sim_parser.add_argument("--output", default=None, help="Output trades CSV path")

    # ── Live Mode ───────────────────────────────────────────────────────
    live_parser = subparsers.add_parser("live", help="Run live trading (TRADING_ENV: testnet=debug | mainnet=production)")
    live_parser.add_argument("--sleep", type=int, default=60, help="Seconds between iterations")
    live_parser.add_argument("--max-iterations", type=int, default=None, help="Max iterations (None=infinite)")
    live_parser.add_argument("--model-dir", default="models", help="Model directory")

    # ── Compare Mode ────────────────────────────────────────────────────
    compare_parser = subparsers.add_parser("compare", help="Compare sim vs demo logs")
    compare_parser.add_argument("--log-dir", default="logs", help="Log directory")
    compare_parser.add_argument("--demo-pattern", default="trading_*.log", help="Demo log pattern")
    compare_parser.add_argument("--sim-pattern", default="trades_sim_*.csv", help="Sim trade CSV pattern")
    compare_parser.add_argument("--output", default="logs/comparison_report.html", help="Report output path")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Dispatch
    if args.mode == "train":
        train_mode(args)
    elif args.mode == "simulate":
        simulate_mode(args)
    elif args.mode == "live":
        live_mode(args)
    elif args.mode == "compare":
        compare_mode(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
