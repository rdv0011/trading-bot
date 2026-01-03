from binancespotbroker import BinanceSpotBroker
from binancefuturesbroker import BinanceFuturesBroker
from basebinancebroker import BaseBinanceBroker
from typing import Dict, Any

def create_binance_broker(config: Dict[str, Any]) -> BaseBinanceBroker:
    market = config.get("market_type", "futures").lower()

    if market == "spot":
        return BinanceSpotBroker(config)
    elif market == "futures":
        return BinanceFuturesBroker(config)
    else:
        raise ValueError(f"Unsupported market_type: {market}")