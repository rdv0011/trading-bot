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
from pathlib import Path
from datetime import datetime

# Load root config.py FIRST for broker credentials (reads .env from repo root)
_parent_path = str(Path(__file__).parent.parent)
if _parent_path not in sys.path:
    sys.path.insert(0, _parent_path)

from config import get_broker_config

# Clear cached config module so local import works
sys.modules.pop("config", None)

# Remove parent path so local modules use simplified/config.py
if _parent_path in sys.path:
    sys.path.remove(_parent_path)

# Local config for strategy parameters (from simplified/config.py)
from config import (
    SYMBOL, TIMEFRAME, STRATEGIC_TF,
    HISTORY_DAYS, TRAIN_FRACTION,
    FEATURE_LAGS, EMA_SPANS, ATR_PERIOD,
    TACTICAL_MODEL_PARAMS, STRATEGIC_MODEL_PARAMS,
    INITIAL_EQUITY, FEE, SLIPPAGE,
    WALKFORWARD_RETRAIN_EVERY, ABSOLUTE_THRESHOLD,
)

# Local logger module (renamed from logging.py to avoid stdlib conflict)
from logger import setup_logging, log_info, log_error
from data import run_full_pipeline, load_featured_df
from model import (
    CatBoostModel, rolling_tactical_predict,
    predict_strategic_meta_params
)
from simulate import run_simulation
from compare import run_comparison

MODEL_DIR = "models"


def train_mode(args):
    """Train tactical and strategic models on historical data."""
    log_info("=" * 60)
    log_info("TRAINING MODE")
    log_info("=" * 60)

    # Run data pipeline for both timeframes
    log_info(f"\n--- Training Tactical Model ({TIMEFRAME}) ---")
    df_train_tactical, df_val_tactical = run_full_pipeline(
        symbol=SYMBOL,
        whole_days=HISTORY_DAYS,
        timeframe=TIMEFRAME,
        train_frac=TRAIN_FRACTION,
    )

    log_info(f"\n--- Training Strategic Model ({STRATEGIC_TF}) ---")
    df_train_strategic, df_val_strategic = run_full_pipeline(
        symbol=SYMBOL,
        whole_days=HISTORY_DAYS,
        timeframe=STRATEGIC_TF,
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
        save_model=True,
    )

    # Train strategic model
    log_info("\n--- Training Strategic Model ---")
    strategic_model = CatBoostModel(
        model_type="strategic",
        model_params=STRATEGIC_MODEL_PARAMS
    )
    strategic_model.train(
        df_train_strategic,
        feature_cols,
        target_col="future_ret",
        save_model=True,
    )

    log_info("\n" + "=" * 60)
    log_info("TRAINING COMPLETE")
    log_info(f"Tactical model: models/tactical_model.cbm")
    log_info(f"Strategic model: models/strategic_model.cbm")
    log_info("=" * 60)


def simulate_mode(args):
    """Run simulation on validation data."""
    log_info("=" * 60)
    log_info("SIMULATION MODE")
    log_info("=" * 60)

    # Load models
    log_info("Loading models...")
    tactical_model = CatBoostModel(model_type="tactical")
    tactical_model.load(model_dir=args.model_dir)

    strategic_model = CatBoostModel(model_type="strategic")
    strategic_model.load(model_dir=args.model_dir)

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
    from timeframe_config import TIMEFRAMES
    tactical_tf_cfg = TIMEFRAMES[TIMEFRAME]

    tactical_preds = rolling_tactical_predict(
        df_val,
        tactical_model,
        feature_cols,
        tactical_tf_cfg,
        retrain_every=WALKFORWARD_RETRAIN_EVERY,
    )

    # Run strategic batch predictions for meta-params
    log_info("\n--- Running Strategic Predictions ---")
    strategic_meta_params = predict_strategic_meta_params(
        df_val,
        strategic_model,
        feature_cols,
    )

    # Run simulation
    log_info("\n--- Running Simulation ---")
    trades_df, metrics, equity_curve = run_simulation(
        df_val,
        tactical_preds,
        strategic_meta_params,
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
    """Run live trading on Binance testnet (reads API keys from .env)."""
    log_info("=" * 60)
    log_info("LIVE TRADING MODE (Testnet)")
    log_info("=" * 60)

    # Load models
    log_info("Loading models...")
    tactical_model = CatBoostModel(model_type="tactical")
    tactical_model.load(model_dir=args.model_dir)

    strategic_model = CatBoostModel(model_type="strategic")
    strategic_model.load(model_dir=args.model_dir)

    # Create broker (loads credentials from .env via config module)
    from broker import BinanceBroker
    broker_config = get_broker_config("futures", testnet=True)
    broker = BinanceBroker(
        api_key=broker_config["api_key"],
        api_secret=broker_config["api_secret"],
        testnet=True,
        symbol=SYMBOL,
    )

    # Get feature columns
    from data import get_feature_cols
    df_sample = broker.get_historical_prices(SYMBOL, TIMEFRAME, 7)
    feature_cols = get_feature_cols(df_sample)

    # Create strategy
    from strategy import DualMLStrategy
    strategy = DualMLStrategy(
        broker=broker,
        tactical_model=tactical_model,
        strategic_model=strategic_model,
        feature_cols=feature_cols,
    )

    log_info(f"Starting live loop (sleep: {args.sleep}s, max: {args.max_iterations})")
    log_info("Press Ctrl+C to stop")

    strategy.run_live_loop(
        sleep_seconds=args.sleep,
        max_iterations=args.max_iterations,
    )


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
    live_parser = subparsers.add_parser("live", help="Run live trading (testnet, reads from .env)")
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