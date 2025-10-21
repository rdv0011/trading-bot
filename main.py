import os
import ccxt
from lumibot.brokers import Ccxt
from dotenv import load_dotenv
from mlstrategy import XGCatBoostStrategy
from lumibot.traders import Trader
from lumibot.entities import Asset

load_dotenv()

BINANCE_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
BINANCE_TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")

# Initialize and run the strategy
if __name__ == "__main__":

    BINANCE_SPOT_TESTNET_CREDS = {
        "exchange_id": "binance",
        "apiKey": BINANCE_TESTNET_API_KEY,
        "secret": BINANCE_TESTNET_API_SECRET,
        "sandbox": True,
        'options': {
            'defaultType': 'stop'
        },
        'urls': {
            'api': {
                "public": "https://testnet.binance.vision/api",
                "private": "https://testnet.binance.vision/api"
            }
        }
    }

    BINANCE_FUTURE_TESTNET_CREDS = {
        "exchange_id": "binance",
        "apiKey": BINANCE_TESTNET_API_KEY,
        "secret": BINANCE_TESTNET_API_SECRET,
        "sandbox": True,
        'options': {
            'defaultType': 'future'
        },
        'urls': {
            'api': {
                'public': 'https://testnet.binancefuture.com/fapi/v1',
                'private': 'https://testnet.binancefuture.com/fapi/v1'
            }
        }
    }

    broker = Ccxt(BINANCE_FUTURE_TESTNET_CREDS)

    base_symbol = "BTC"
    quote_symbol = "USDT"
    quote_asset = Asset(symbol=quote_symbol, asset_type="crypto")
    parameters = {
        "asset_symbol" : base_symbol,
        "stake_pct" : 0.05,
        "stop_loss_pct" : 0.04,
        "max_hold_hours" : 24,
        "max_history_size" : 600,
        "historical_prices_length" : 500,
        "historical_prices_unit" : "minute",
        "predict_with_signal_num_candles" : 600,
        "predict_with_signal_label_window" : 200,
        "model_type": "xgb",  # or 'cat' for CatBoost
        "auto_reload": True,
        "sleeptime": "5M"  # 5 minutes
    }

    strategy = XGCatBoostStrategy(
        broker=broker,
        quote_asset=quote_asset,
        parameters=parameters,
    )
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()