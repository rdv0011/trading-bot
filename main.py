import os
from lumibot.brokers import Ccxt
from dotenv import load_dotenv
from mlstrategy import MLTradingStrategy
from lumibot.traders import Trader
from lumibot.entities import Asset

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Initialize and run the strategy
if __name__ == "__main__":
            
    BINANCE_CREDS = {
        "exchange_id": "binance",
        "urls": {
            "api": "https://testnet.binance.vision/api"
        },
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_API_SECRET,
        "sandbox": True,
        'options': {
            'defaultType': 'spot',
            'fetchMarkets': ['spot']
        }
    }

    base_symbol = "BTC"
    quote_symbol = "USDT"
    quote_asset = Asset(symbol=quote_symbol, asset_type="forex")
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
    }

    broker = Ccxt(BINANCE_CREDS)
    strategy = MLTradingStrategy(
        broker=broker,
        quote_asset=quote_asset,
        parameters=parameters,
    )
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()