from binancespotbroker import BinanceSpotBroker
from binancefuturesbroker import BinanceFuturesBroker
from binancebasebroker import BinanceBaseBroker
from typing import Dict, Any

def create_binance_broker(config: Dict[str, Any]) -> BinanceBaseBroker:
    """
    Factory for creating a Binance broker instance (spot or futures).

    config example:
    {
        "api_key": "...",
        "api_secret": "...",
        "testnet": True,
        "market_type": "futures" | "spot"
    }
    """
    market = config.get("market_type", "futures").lower()

    if market == "spot":
        return BinanceSpotBroker(config)

    if market == "futures":
        return BinanceFuturesBroker(config)

    raise ValueError(f"Unsupported market_type: {market}")