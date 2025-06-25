# Machine Learning Crypto Trading Bot

A trading bot that uses machine learning to predict cryptocurrency price movements and execute trades based on those predictions.

## Overview

This project implements a machine learning-based trading strategy for cryptocurrencies using the Lumibot framework. It incorporates models like AutoTS and scikit-learn's RandomForestRegressor to forecast price changes and make trading decisions.

## Features

- **Machine Learning Models**: Utilizes AutoTS and RandomForestRegressor for price prediction.
- **Technical Indicators**: Calculates various indicators like RSI, EMA, Bollinger Bands, and more to enhance prediction accuracy.
- **Backtesting**: Supports backtesting with historical data to evaluate strategy performance.
- **Live Trading**: Configurable for live trading with Alpaca broker API.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd lumibot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file with your Alpaca API credentials:
     ```
     API_KEY=your_api_key
     API_SECRET=your_api_secret
     ```

## Usage

- **Backtesting**: Modify the `backtesting_start` and `backtesting_end` dates in `forecasting_bot.py` to run a backtest.
- **Live Trading**: Set `is_live = True` in `forecasting_bot.py` to start live trading with Alpaca.

```python
# For backtesting
python forecasting_bot.py

# Ensure is_live is set to True for live trading
```

## Data

Historical cryptocurrency data is downloaded from Binance and cached locally in the `data` directory for backtesting purposes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
