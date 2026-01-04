import os
import argparse
from dotenv import load_dotenv

from mlstrategy import MLStrategy
from binancebrokerfactory import create_binance_broker

load_dotenv()

# Futures (Testnet)
BINANCE_TESTNET_FUTURES_API_KEY = os.getenv("BINANCE_TESTNET_FUTURES_API_KEY", "")
BINANCE_TESTNET_FUTURES_API_SECRET = os.getenv("BINANCE_TESTNET_FUTURES_API_SECRET", "")

# Spot (Testnet or Live – Binance Spot testnet uses same keys)
BINANCE_SPOT_API_KEY = os.getenv("BINANCE_SPOT_API_KEY", "")
BINANCE_SPOT_API_SECRET = os.getenv("BINANCE_SPOT_API_SECRET", "")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run trading bot strategy.")

    parser.add_argument(
        "--model-type",
        choices=["xgb", "cat"],
        default="cat",
        help="Select the model type to use."
    )

    parser.add_argument(
        "--market-type",
        choices=["spot", "futures"],
        default="futures",
        help="Select Binance market type."
    )

    args = parser.parse_args()

    # ---------------- Broker Config ----------------

    if args.market_type == "futures":
        broker_config = {
            "api_key": BINANCE_TESTNET_FUTURES_API_KEY,
            "api_secret": BINANCE_TESTNET_FUTURES_API_SECRET,
            "market_type": "futures",
            "testnet": True,
        }
    else:
        broker_config = {
            "api_key": BINANCE_SPOT_API_KEY,
            "api_secret": BINANCE_SPOT_API_SECRET,
            "market_type": "spot",
            "testnet": True,
        }

    broker = create_binance_broker(broker_config)

    # ---------------- Strategy Params ----------------

    base_symbol = "BTC"
    quote_symbol = "USDT"

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