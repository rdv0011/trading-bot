import os
from dotenv import load_dotenv
from mlstrategy import XGCatBoostStrategy
from binancetbroker import BinanceBroker
import argparse

load_dotenv()

BINANCE_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
BINANCE_TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")

# Initialize and run the strategy
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run trading bot strategy.")
    parser.add_argument(
        "--model-type",
        choices=["xgb", "cat"],
        default="cat",
        help="Select the model type to use."
    )
    args = parser.parse_args()

    broker_config = {
        'api_key': BINANCE_TESTNET_API_KEY,
        'api_secret': BINANCE_TESTNET_API_SECRET,
    }

    broker = BinanceBroker(broker_config)

    base_symbol = "BTC"
    quote_symbol = "USDT"
    parameters = {
        "asset_symbol" : base_symbol,
        "historical_prices_unit" : "5m",
        "model_type": args.model_type,  # or 'cat' for CatBoost
        "auto_reload": True,
        "sleeptime": "5m"  # 5 minutes
    }

    strategy = XGCatBoostStrategy(
        broker=broker,
        quote_symbol=quote_symbol,
        parameters=parameters,
    )
    strategy.run()