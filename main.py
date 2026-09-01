import argparse

from binancebasebroker import MARKET_TYPE_FUTURES, MARKET_TYPE_SPOT
from mlstrategy import MLStrategy
from dualmlstrategy import DualMLStrategy
from binancebrokerfactory import create_binance_broker
from config import get_broker_config, DEFAULT_SYMBOL

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run trading bot strategy.")

    parser.add_argument(
        "--market-type",
        choices=["spot", "futures"],
        default="futures",
    )
    parser.add_argument(
        "--strategy",
        choices=["legacy", "dual"],
        default="dual",
        help="'legacy' = original single-ML, 'dual' = new two-tier ML system",
    )
    parser.add_argument(
        "--train-strategic",
        action="store_true",
        help="Run strategic model training then exit (no live trading).",
    )
    parser.add_argument("--strategic-days", type=int, default=365)
    parser.add_argument("--strategic-timeframe", default="1h")
    parser.add_argument(
        "--tactical-days",
        type=int,
        default=45,
        help="Days of 5m data used for walk-forward param optimisation (requires --train-strategic).",
    )
    parser.add_argument(
        "--optimize-params",
        action="store_true",
        help="Use simulation-driven param optimisation when training the strategic model.",
    )
    args = parser.parse_args()

    if args.train_strategic:
        from strategic.strategictraining import run_training
        from mlio import MODEL_DIR

        df_5m_predictions = None
        if args.optimize_params:
            from dualmlsimulation import run_predictions_only
            print(f"Running walk-forward tactical predictions ({args.tactical_days}d 5m)...")
            df_5m_predictions, _ = run_predictions_only(
                symbol="BTCUSDT",
                days=args.tactical_days,
                timeframe="5m",
            )

        run_training(
            symbol="BTCUSDT",
            days=args.strategic_days,
            timeframe=args.strategic_timeframe,
            model_dir=MODEL_DIR,
            df_5m_predictions=df_5m_predictions,
        )
        raise SystemExit(0)

    # Get broker config from centralized config (loads from .env)
    testnet = True  # Default to testnet for safety
    broker_config = get_broker_config(args.market_type, testnet=testnet)

    broker = create_binance_broker(broker_config)

    base_symbol = DEFAULT_SYMBOL.replace("USDT", "")
    quote_symbol = "USDT"

    if args.strategy == "dual":
        parameters = {
            "asset_symbol": base_symbol,
            "model_type": "cat",
            "market_type": args.market_type,
            "tactical_timeframe": "5m",
            "strategic_timeframe": "1h",
            "model_params": {"iterations": 300, "verbose": False},
            "sleeptime": "5m",
        }
        strategy = DualMLStrategy(
            broker=broker,
            quote_symbol=quote_symbol,
            parameters=parameters,
        )
    else:
        parameters = {
            "asset_symbol": base_symbol,
            "historical_prices_unit": "5m",
            "model_type": "cat",
            "auto_reload": True,
            "sleeptime": "5m",
            "market_type": args.market_type,
        }
        strategy = MLStrategy(
            broker=broker,
            quote_symbol=quote_symbol,
            parameters=parameters,
        )

    strategy.run()