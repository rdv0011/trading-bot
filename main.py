import os
import argparse
from dotenv import load_dotenv

from binancebasebroker import MARKET_TYPE_FUTURES, MARKET_TYPE_SPOT
from mlstrategy import MLStrategy
from dualmlstrategy import DualMLStrategy
from binancebrokerfactory import create_binance_broker

load_dotenv()

BINANCE_TESTNET_FUTURES_API_KEY = os.getenv("BINANCE_TESTNET_FUTURES_API_KEY")
BINANCE_TESTNET_FUTURES_API_SECRET = os.getenv("BINANCE_TESTNET_FUTURES_API_SECRET")
BINANCE_TESTNET_SPOT_API_KEY = os.getenv("BINANCE_TESTNET_SPOT_API_KEY")
BINANCE_TESTNET_SPOT_API_SECRET = os.getenv("BINANCE_TESTNET_SPOT_API_SECRET")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run trading bot strategy.")

    parser.add_argument(
        "--model-type",
        choices=["xgb", "cat"],
        default="cat",
    )
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

    args = parser.parse_args()

    if args.train_strategic:
        from strategic.strategictraining import run_training
        from mlio import MODEL_DIR
        run_training(
            symbol="BTCUSDT",
            days=args.strategic_days,
            model_type=args.model_type,
            timeframe=args.strategic_timeframe,
            model_dir=MODEL_DIR,
        )
        raise SystemExit(0)

    if args.market_type == "futures":
        if not BINANCE_TESTNET_FUTURES_API_KEY or not BINANCE_TESTNET_FUTURES_API_SECRET:
            raise ValueError("BINANCE_TESTNET_FUTURES_API_KEY and BINANCE_TESTNET_FUTURES_API_SECRET must be set")
        broker_config = {
            "api_key": BINANCE_TESTNET_FUTURES_API_KEY,
            "api_secret": BINANCE_TESTNET_FUTURES_API_SECRET,
            "market_type": MARKET_TYPE_FUTURES,
            "testnet": True,
        }
    else:
        if not BINANCE_TESTNET_SPOT_API_KEY or not BINANCE_TESTNET_SPOT_API_SECRET:
            raise ValueError("BINANCE_TESTNET_SPOT_API_KEY and BINANCE_TESTNET_SPOT_API_SECRET must be set")
        broker_config = {
            "api_key": BINANCE_TESTNET_SPOT_API_KEY,
            "api_secret": BINANCE_TESTNET_SPOT_API_SECRET,
            "market_type": MARKET_TYPE_SPOT,
            "testnet": True,
        }

    broker = create_binance_broker(broker_config)

    base_symbol = "BTC"
    quote_symbol = "USDT"

    if args.strategy == "dual":
        parameters = {
            "asset_symbol": base_symbol,
            "model_type": args.model_type,
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
            "model_type": args.model_type,
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
