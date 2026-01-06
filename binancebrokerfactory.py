from binancespotbroker import BinanceSpotBroker
from binancefuturesbroker import BinanceFuturesBroker
from binancebasebroker import MARKET_TYPE_SPOT, MARKET_TYPE_FUTURES, BinanceBaseBroker
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

    if market == MARKET_TYPE_SPOT:
        return BinanceSpotBroker(config)

    if market == MARKET_TYPE_FUTURES:
        return BinanceFuturesBroker(config)

    raise ValueError(f"Unsupported market_type: {market}")