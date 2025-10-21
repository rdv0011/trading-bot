import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

BINANCE_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
BINANCE_TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")

# Initialize and run the strategy
if __name__ == "__main__":

    # --- 1. Connect to Binance ---
    exchange = ccxt.binance({
        'api_key': BINANCE_TESTNET_API_KEY,
        'secret': BINANCE_TESTNET_API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',  # Use 'future' for Binance Futures, or 'spot' for spot trading
        },
    })

    exchange.set_sandbox_mode(True)

    symbol = 'BTC/USDT'
    position_size = 0
    order_side = 'buy'   # or 'sell'

    # --- 2. Fetch current positions ---
    balance = exchange.fetch_balance()
    positions = balance['info']['positions']

    order = exchange.create_market_order(symbol, 'buy', amount=0.049)
    print("Order placed:", order)
    
    # Find position for BTC/USDT
    for pos in positions:
        if pos['symbol'] == symbol.replace('/', ''):
            position_size = float(pos['positionAmt'])
            print(f"Current position size for {symbol}: {position_size}")
            break

    # --- 3. Place an order if no open position ---
    if position_size == 0:
        print("No open position found. Placing a market order...")
        order = exchange.create_market_order(symbol, order_side, amount=0.001)
        print("Order placed:", order)
    else:
        print("Position already open, no new order placed.")