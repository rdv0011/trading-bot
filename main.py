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
        "stake_frac" : 0.5,
        "stop_loss_frac" : 0.01,
        "take_profit_frac" : 0.02,
        "max_hold_hours" : 24,
        "max_history_size" : 600,
        "historical_prices_length" : 500,
        "historical_prices_unit" : "5m",
        "predict_with_signal_num_candles" : 600, # 600 predictions needed for adaptive thresholding
        "predict_with_signal_label_window" : 200,
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