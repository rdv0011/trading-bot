import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ta
from autots import AutoTS
from lumibot.backtesting import PandasDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.entities import Asset, Data
from lumibot.entities.asset import Asset
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from sklearn.ensemble import RandomForestRegressor
import requests
import zipfile
from dateutil.relativedelta import relativedelta
import io

API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
BASE_URL = "https://paper-api.alpaca.markets/v2"

class MachineLearningCrypto(Strategy):
    """Parameters:

    symbol (str, optional): The symbol that we want to trade. Defaults to "SRNE".
    compute_frequency (int, optional): The time (in minutes) that we should retrain our model.
    lookback_period (int, optional): The amount of data (in minutes) that we get from our data source to use in the model.
    pct_portfolio_per_trade (float, optional): The size that each trade will be (in percent of the total portfolio).
    price_change_threshold_up (float, optional): The difference between predicted price and the current price that will trigger a buy order (in percentage change).
    price_change_threshold_down (float, optional): The difference between predicted price and the current price that will trigger a sell order (in percentage change).
    max_pct_portfolio (float, optional): The maximum that the strategy will buy or sell as a percentage of the portfolio (eg. if this is 0.8 - or 80% - and our portfolio is worth $100k, then we will stop buying when we own $80k worth of the symbol)
    take_profit_factor: Where you place your limit order based on the prediction
    stop_loss_factor: Where you place your stop order based on the prediction
    """

    parameters = {
        "compute_frequency": 15,
        "lookback_period": 200,  # Increasing this will improve accuracy but will take longer to train
        "pct_portfolio_per_trade": 0.95,
        "price_change_threshold_up": 0.005,
        "price_change_threshold_down": -0.005,
        "max_pct_portfolio_long": 2,
        "max_pct_portfolio_short": 0.3,
        "take_profit_factor": 1,
        "stop_loss_factor": 0.5,
        "ml_model_type": "sklearn",  # "autots" or "sklearn"
    }

    def initialize(self):
        # Set the initial variables or constants

        # Built in Variables
        if self.is_backtesting:
            # If we are backtesting we do not need to check very often
            self.sleeptime = self.parameters["compute_frequency"]
        else:
            # Check more often if we are trading in order to get more data
            self.sleeptime = "10S"

        # Variable initial states
        self.last_compute = None
        self.prediction = None
        self.last_price = None
        self.asset_value = None
        self.shares_owned = None
        self.cache_df = None

        self.set_market("24/7")

        self.model = AutoTS(
            forecast_length=self.parameters["compute_frequency"],
            frequency="infer",
            prediction_interval=0.9,
            ensemble=None,
            model_list="superfast",  # "superfast", "default", "fast_parallel"
            transformer_list="superfast",  # "superfast",
            drop_most_recent=1,
            max_generations=2,
            num_validations=2,
            validation_method="backwards",
            verbose=False,
        )

    def on_trading_iteration(self):
        # Get parameters for this iteration
        asset = self.parameters["asset"]
        compute_frequency = self.parameters["compute_frequency"]
        lookback_period = self.parameters["lookback_period"]
        pct_portfolio_per_trade = self.parameters["pct_portfolio_per_trade"]
        price_change_threshold_up = self.parameters["price_change_threshold_up"]
        price_change_threshold_down = self.parameters["price_change_threshold_down"]
        max_pct_portfolio_long = self.parameters["max_pct_portfolio_long"]
        max_pct_portfolio_short = self.parameters["max_pct_portfolio_short"]
        take_profit_factor = self.parameters["take_profit_factor"]
        stop_loss_factor = self.parameters["stop_loss_factor"]
        ml_model_type = self.parameters["ml_model_type"]

        dt = self.get_datetime()
        time_since_last_compute = None
        if self.last_compute is not None:
            time_since_last_compute = dt - self.last_compute

        if time_since_last_compute is None or time_since_last_compute >= timedelta(
            minutes=compute_frequency
        ):
            self.last_compute = dt

            # Get the data
            data = self.get_data(
                asset, self.quote_asset, compute_frequency * lookback_period
            )

            # The current price of the asset
            self.last_price = data.iloc[-1]["close"]

            # Get how much we currently own of the asset
            position = self.get_position(asset)
            if position is None:
                self.shares_owned = 0
            else:
                self.shares_owned = float(position.quantity)
            self.asset_value = self.shares_owned * self.last_price

            # Reset the prediction
            self.prediction = None
            directory = "cache"
            cache_filepath = (
                f"{directory}/{ml_model_type}_{compute_frequency}_{asset.symbol}.csv"
            )
            if self.is_backtesting:
                # Check if file exists, if not then create it
                if os.path.isfile(cache_filepath):
                    self.log_message("File exists")
                    if self.cache_df is None:
                        self.cache_df = pd.read_csv(cache_filepath)
                        self.cache_df["datetime"] = pd.to_datetime(
                            self.cache_df["datetime"]
                        )
                        self.cache_df = self.cache_df.set_index("datetime")

                    # check if the current datetime exists
                    # current_prediction = self.cache_df.loc[dt]
                    current_prediction = self.cache_df.loc[self.cache_df.index == dt]

                    if current_prediction is not None and len(current_prediction) == 1:
                        self.prediction = current_prediction["prediction"][0]
                else:
                    if not os.path.exists(directory):
                        os.mkdir(directory)
                    self.cache_df = pd.DataFrame(columns=["prediction"])
                    self.cache_df.index.name = "datetime"

            if self.prediction is None:
                # code to predict
                data["close_future"] = data["close"].shift(-compute_frequency)
                if "symbol" in data:
                    data = data.drop(["symbol"], axis=1)
                if "exchange" in data:
                    data = data.drop(["exchange"], axis=1)                 
                    
                    
                #print(data)
                data_train = data.dropna()

                if ml_model_type == "autots":
                    self.model = self.model.fit(data_train)
                    predictions = self.model.predict().forecast

                    # Our model's preduicted price
                    self.prediction = predictions["close"][0]
                elif ml_model_type == "sklearn":
                    # Predict
                        
                    rf = RandomForestRegressor().fit(
                        X=data_train.drop(["close_future"], axis=1),
                        y=data_train["close_future"],
                    )

                    # Our current situation
                    last_row = data.iloc[[-1]]

                    X_test = last_row.drop(["close_future"], axis=1)
                    predictions = rf.predict(X_test)

                    # Our model's preduicted price
                    self.prediction = predictions[0]

                df = pd.DataFrame([self.prediction], columns=["prediction"], index=[dt])
                df.index.name = "datetime"
                self.cache_df = pd.concat([self.cache_df, df])
                self.cache_df.sort_index(inplace=True)
                self.cache_df.to_csv(cache_filepath)

            # Calculate the percentage change that the model predicts
            expected_price_change = self.prediction - self.last_price
            self.expected_price_change_pct = expected_price_change / self.last_price
            
            self.log_message(f"predicting {self.prediction} and last price was {self.last_price}, so we are expecting a price change of {self.expected_price_change_pct * 100:0.2f}%", color="green")

            # Our machine learning model is predicting that the asset will increase in value
            if self.expected_price_change_pct > price_change_threshold_up:
                max_position_size = max_pct_portfolio_long * self.portfolio_value
                value_to_trade = self.portfolio_value * pct_portfolio_per_trade
                quantity = value_to_trade / self.last_price

                # Check that we are not buying too much of the asset
                if (self.asset_value + value_to_trade) < max_position_size:
                    # Calculate take profit and stop loss prices
                    expected_price_move = abs(
                        self.last_price * self.expected_price_change_pct
                    )
                    take_profit_price = self.last_price + (expected_price_move * take_profit_factor)
                    stop_loss_price = self.last_price - (
                        expected_price_move * stop_loss_factor
                    )
                    
                    self.log_message(f"Submitting bracket order for {quantity} shares. Take Profit: {take_profit_price}, Stop Loss: {stop_loss_price}")
                    
                    # Create a single bracket order
                    order = self.create_order(
                        asset,
                        quantity,
                        "buy",
                        take_profit_price=take_profit_price,
                        stop_loss_price=stop_loss_price,
                        quote=self.quote_asset
                    )
                    self.submit_order(order)

            # Our machine learning model is predicting that the asset will decrease in value
            elif self.expected_price_change_pct < price_change_threshold_down:
                max_position_size = max_pct_portfolio_short * self.portfolio_value
                value_to_trade = self.portfolio_value * pct_portfolio_per_trade
                quantity = value_to_trade / self.last_price

                # Check that we are not selling too much of the asset
                if (self.asset_value - value_to_trade) > -max_position_size:
                    # Calculate take profit and stop loss prices
                    expected_price_move = abs(
                        self.last_price * self.expected_price_change_pct
                    )
                    take_profit_price = self.last_price - (expected_price_move * take_profit_factor)
                    stop_loss_price = self.last_price + (
                        expected_price_move * stop_loss_factor
                    )
                    
                    self.log_message(f"Submitting bracket order for {quantity} shares. Take Profit: {take_profit_price}, Stop Loss: {stop_loss_price}")

                    # Create a single bracket order for shorting
                    order = self.create_order(
                        asset,
                        quantity,
                        "sell",
                        take_profit_price=take_profit_price,
                        stop_loss_price=stop_loss_price,
                        quote=self.quote_asset
                    )
                    self.submit_order(order)

    def on_abrupt_closing(self):
        self.sell_all()

    # Add our predictions to stats.csv so that we can see how to improve our strategy
    # Eg. Will tell us whether our predictions are accurate or not
    def trace_stats(self, context, snapshot_before):
        row = {
            "prediction": self.prediction,
            "last_price": self.last_price,
            "absolute_error": abs(self.prediction - self.last_price),
            "squared_error": (self.prediction - self.last_price) ** 2,
            "expected_price_change_pct": self.expected_price_change_pct,
        }

        return row

    def get_data(self, asset, quote_asset, window_size):
        """Gets pricing data from our data source, then calculates a bunch of technical indicators

        Args:
            asset (Asset): The asset that we want the data for
            window_size (int): The amount of data points that we want to get from our data source (in minutes)

        Returns:
            pandas.DataFrame: A DataFrame with the asset's prices and technical indicators
        """
        data_length = window_size + 40

        bars = self.get_historical_prices(
            asset, data_length, "minute", quote=quote_asset
        )
        data = bars.df

        times = data.index.to_series()
        current_datetime = self.get_datetime()
        data["timeofday"] = (times.dt.hour * 60) + times.dt.minute
        data["timeofdaysq"] = ((times.dt.hour * 60) + times.dt.minute) ** 2
        data["unixtime"] = data.index.astype(np.int64) // 10 ** 9
        data["unixtimesq"] = data.index.astype(np.int64) // 10 ** 8
        data["time_from_now"] = current_datetime.timestamp() - data["unixtime"]
        data["time_from_now_sq"] = data["time_from_now"] ** 2

        data["delta"] = np.append(
            None,
            (np.array(data["close"])[1:] - np.array(data["close"])[:-1])
            / np.array(data["close"])[:-1],
        )
        data["rsi"] = ta.momentum.rsi(data["close"])
        data["ema"] = ta.trend.ema_indicator(data["close"])
        data["cmf"] = ta.volume.chaikin_money_flow(
            data["high"], data["low"], data["close"], data["volume"]
        )
        data["vwap"] = ta.volume.volume_weighted_average_price(
            data["high"], data["low"], data["close"], data["volume"]
        )
        data["bollinger_high"] = ta.volatility.bollinger_hband(data["close"])
        data["bollinger_low"] = ta.volatility.bollinger_lband(data["close"])
        data["macd"] = ta.trend.macd(data["close"])
        # data["adx"] = ta.trend.adx(data["high"], data["low"], data["close"])
        ichimoku = ta.trend.IchimokuIndicator(data["high"], data["low"])
        data["ichimoku_a"] = ichimoku.ichimoku_a()
        data["ichimoku_b"] = ichimoku.ichimoku_b()
        data["ichimoku_base"] = ichimoku.ichimoku_base_line()
        data["ichimoku_conversion"] = ichimoku.ichimoku_conversion_line()
        data["stoch"] = ta.momentum.stoch(data["high"], data["low"], data["close"])

        # This was causing the problem. It was adding NaN values to the dataframe
        # data["kama"] = ta.momentum.kama(data["close"])

        data = data.dropna()

        data = data.iloc[-window_size:]

        return data

def download_and_cache_binance_klines(symbol="BTCUSDT", interval="1m", year=2021, month=2):
    month_str = f"{month:02d}"
    file_stem = f"{symbol}-{interval}-{year}-{month_str}"
    csv_path = os.path.join(DATA_DIR, f"{file_stem}.csv")

    # Use cache if available
    if os.path.exists(csv_path):
        print(f"Using cached file: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["date"])

        df.set_index("date", inplace=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Ensure column types are consistent
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        return df

    # Download and unzip from Binance
    print(f"Downloading {file_stem} from Binance...")
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{file_stem}.zip"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {url}")

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_file = z.namelist()[0]
        with z.open(csv_file) as f:
            df = pd.read_csv(
                f,
                header=None,
                names=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
                ]
            )

    # Format and save
    # Convert open_time safely
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df[(df["open_time"] >= 1262304000000000) & (df["open_time"] <= 4102444800000000)].copy()

    # Convert microseconds to datetime
    df["date"] = pd.to_datetime(df["open_time"], unit="us")
    df.set_index("date", inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # Cache locally
    df.reset_index().to_csv(csv_path, index=False)
    print(f"Saved to cache: {csv_path}")
    return df


def generate_month_range(start_date, end_date):
    current = datetime(start_date.year, start_date.month, 1)
    end = datetime(end_date.year, end_date.month, 1)
    while current <= end:
        yield current.year, current.month
        current += relativedelta(months=1)

if __name__ == "__main__":
    is_live = False

    if is_live:
        ####
        # Run the strategy
        ####

        # Add in Alpaca API details

        ALPACA_CREDS = {
            "API_KEY":API_KEY,
            "API_SECRET":API_SECRET,
            "PAPER": True
        }

        broker = Alpaca(ALPACA_CREDS)
        
        asset = Asset(symbol="BTC", asset_type="crypto")

        strategy = MachineLearningCrypto(
            broker=broker,
            parameters={
                "asset": asset,
            },
        )

        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()

    else:
        ####
        # Backtest
        ####

        # Backtesting range
        backtesting_start = datetime(2025, 1, 1)
        backtesting_end = datetime(2025, 1, 31)

        # Define symbols
        base_symbol = "BTC"
        quote_symbol = "USD"
        binance_symbol = base_symbol + "USDT"     # Binance convention
        benchmark_symbol = f"{base_symbol}-{quote_symbol}"  # Lumibot convention

        # Define assets
        asset = Asset(symbol=base_symbol, asset_type="crypto")
        quote_asset = Asset(symbol=quote_symbol, asset_type="forex")

        DATA_DIR = "data"
        os.makedirs(DATA_DIR, exist_ok=True)

        # Download and merge data
        dfs = []
        for year, month in generate_month_range(backtesting_start, backtesting_end):
            df_month = download_and_cache_binance_klines(symbol=binance_symbol, interval="1m", year=year, month=month)
            dfs.append(df_month)

        df = pd.concat(dfs)

        # Clip data to backtesting period
        df = df.loc[(df.index >= pd.Timestamp(backtesting_start, tz="UTC")) &
                    (df.index <= pd.Timestamp(backtesting_end, tz="UTC"))]

        # Wrap into Lumibot-compatible data format
        pandas_data = {asset: Data(asset, df, timestep="minute", quote=quote_asset)}

        # Run backtest
        MachineLearningCrypto.backtest(
            PandasDataBacktesting,
            backtesting_start,
            backtesting_end,
            pandas_data=pandas_data,
            benchmark_asset=benchmark_symbol,
            quote_asset=quote_asset,
            parameters={
                "asset": asset,
            },
        )
