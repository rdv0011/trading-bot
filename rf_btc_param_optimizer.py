import os
import joblib
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestRegressor
from lumibot.strategies.strategy import Strategy
from lumibot.entities import Asset, Data
from lumibot.backtesting import PandasDataBacktesting
from datetime import timedelta
from zoneinfo import ZoneInfo
from threading import Lock
from historical_data import get_historical_data
from datetime import datetime
import csv
import argparse


TIMEZONE = "UTC"
MODEL_FILE = "rf_model.pkl"
CACHE_DIR = "cache"
FEATURE_FILE = "rf_features.txt"
BINARY_SEARCH_ITERS = 1  # Default number of binary search iterations

# -----------------------------
# RandomForest predictor
# -----------------------------
class RandomForestPredictor:
    _cache_lock = Lock()

    def __init__(self, model_file=MODEL_FILE, cache_dir=CACHE_DIR, feature_file=FEATURE_FILE):
        self.model_file = os.path.join(cache_dir, model_file)
        self.feature_file = os.path.join(cache_dir, feature_file)
        self.cache_dir = cache_dir
        self.model = None
        self.features = None
        os.makedirs(cache_dir, exist_ok=True)

        # Load model if available
        if os.path.exists(self.model_file):
            try:
                self.model = joblib.load(self.model_file)
                print(f"[RF] Loaded model from {self.model_file}")
            except Exception as e:
                print(f"[RF] Failed to load model: {e}")

        # Load saved feature list if available
        if os.path.exists(self.feature_file):
            with open(self.feature_file, "r") as f:
                self.features = [line.strip() for line in f.readlines() if line.strip()]

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure dataframe columns match training feature names"""
        if self.features is None:
            return df

        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0  # Add missing columns with zeros

        # Drop any extra columns not in the trained model
        df = df[self.features]
        return df

    def train(self, df, horizon=1, persist=True, force_retrain=False):
        if self.model is not None and not force_retrain:
            return

        df = df.copy()
        df["target"] = df["close"].shift(-horizon)
        df = df.dropna()

        X = df.drop(["target"], axis=1)
        y = df["target"]

        self.features = list(X.columns)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)

        if persist:
            joblib.dump(self.model, self.model_file)
            with open(self.feature_file, "w") as f:
                f.write("\n".join(self.features))
            print(f"[RF] Model + feature list saved to {self.cache_dir}")

    def predict(self, df):
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        last_row = df.iloc[[-1]].drop(["target"], axis=1, errors="ignore")
        last_row = self._align_features(last_row)
        return float(self.model.predict(last_row)[0])

    def cache_prediction(self, date, prediction, tag="btc_rf"):
        filepath = os.path.join(self.cache_dir, f"{tag}_predictions.csv")

        if date.tzinfo is None:
            date = date.replace(microsecond=0, tzinfo=ZoneInfo(TIMEZONE))
        else:
            date = date.astimezone(ZoneInfo(TIMEZONE)).replace(microsecond=0)

        row = pd.DataFrame([[date, round(prediction, 2)]], columns=["date", "prediction"])
        with self._cache_lock:
            if os.path.exists(filepath):
                row.to_csv(filepath, mode="a", header=False, index=False)
            else:
                row.to_csv(filepath, index=False)

    def get_cached_prediction(self, date, tag="btc_rf"):
        filepath = os.path.join(self.cache_dir, f"{tag}_predictions.csv")
        if not os.path.exists(filepath):
            return None
        try:
            with self._cache_lock:
                df = pd.read_csv(filepath, parse_dates=["date"])
            row = df[df["date"] == pd.to_datetime(date)]
            if not row.empty:
                return float(row["prediction"].iloc[0])
        except Exception as e:
            print(f"Cache read error: {e}")
        return None


# -----------------------------
# Lumibot Strategy
# -----------------------------
class RandomForestBTCStrategy(Strategy):
    parameters = {
        "compute_frequency": 15,
        "lookback_period": 200,
        "fraction_portfolio_per_trade": 0.1,
        "price_change_threshold_up": 0.006,
        "price_change_threshold_down": -0.005,
        "max_portfolio_exposure_long": 1.0,
        "max_portfolio_exposure_short": 0.5,
        "take_profit_multiplier": 1.2,
        "stop_loss_multiplier": 0.8,
        # Optimization config
        "binary_search_iters": BINARY_SEARCH_ITERS,
    }

    def initialize(self):
        asset_symbol = self.parameters.get("asset_symbol", "BTC")
        self.asset = Asset(asset_symbol, "crypto")
        self.predictor = RandomForestPredictor()
        self.current_position = None
        self.last_compute = None
        self.set_market("24/7")

    # -----------------------------
    # Indicators
    # -----------------------------
    def get_data(self, window_size: int) -> pd.DataFrame:
        data_length = window_size + 40
        bars = self.get_historical_prices(self.asset, data_length, "minute", quote=self.quote_asset)
        df = bars.df.copy()
        df = self._compute_features(df, current_datetime=self.get_datetime())
        return df.iloc[-window_size:].copy()

    
    @staticmethod
    def _compute_features(df: pd.DataFrame, current_datetime: datetime | None = None) -> pd.DataFrame:
        df = df.copy()
        if current_datetime is None:
            current_datetime = datetime.now(ZoneInfo(TIMEZONE))

        times = df.index.to_series()
        df["timeofday"] = (times.dt.hour * 60) + times.dt.minute
        df["timeofdaysq"] = df["timeofday"] ** 2
        df["unixtime"] = df.index.astype(np.int64) // 10 ** 9
        df["time_from_now"] = current_datetime.timestamp() - df["unixtime"]

        df["delta"] = df["close"].pct_change()
        df["rsi"] = ta.momentum.rsi(df["close"])
        df["ema"] = ta.trend.ema_indicator(df["close"])
        df["cmf"] = ta.volume.chaikin_money_flow(df["high"], df["low"], df["close"], df["volume"])
        df["vwap"] = ta.volume.volume_weighted_average_price(df["high"], df["low"], df["close"], df["volume"])
        df["bollinger_high"] = ta.volatility.bollinger_hband(df["close"])
        df["bollinger_low"] = ta.volatility.bollinger_lband(df["close"])
        df["macd"] = ta.trend.macd(df["close"])
        ichimoku = ta.trend.IchimokuIndicator(df["high"], df["low"])
        df["ichimoku_a"] = ichimoku.ichimoku_a()
        df["ichimoku_b"] = ichimoku.ichimoku_b()
        df["ichimoku_base"] = ichimoku.ichimoku_base_line()
        df["ichimoku_conversion"] = ichimoku.ichimoku_conversion_line()
        df["stoch"] = ta.momentum.stoch(df["high"], df["low"], df["close"])
        return df.dropna().copy()


    # -----------------------------
    # Trading iteration
    # -----------------------------
    def on_trading_iteration(self):
        p = self.parameters.copy()
        dt = self.get_datetime()

        if self.last_compute is not None:
            if dt - self.last_compute < timedelta(minutes=p["compute_frequency"]):
                return
        self.last_compute = dt

        train_window_size = p["compute_frequency"] * p["lookback_period"]
        data = self.get_data(train_window_size)
        last_price = data["close"].iloc[-1]

        pred = self.predictor.get_cached_prediction(dt)
        if pred is None:
            self.predictor.train(data, horizon=p["compute_frequency"])
            pred = self.predictor.predict(data)
            self.predictor.cache_prediction(dt, pred)

        expected_change = (pred - last_price) / last_price
        self.log_message(f"Predicted {pred:.2f}, last {last_price:.2f}, expected {expected_change:.4f}")

        if expected_change > p["price_change_threshold_up"]:
            self.open_position_with_orders("long", last_price, p, expected_change)
        elif expected_change < p["price_change_threshold_down"]:
            self.open_position_with_orders("short", last_price, p, expected_change)

        # -----------------------------
    # Order creation helpers (Lumibot style)
    # -----------------------------
    def open_position_with_orders(self, direction: str, price: float, params: dict, expected_change: float):
        # Calculate sizing
        value_to_trade = self.portfolio_value * params["fraction_portfolio_per_trade"]
        quantity = value_to_trade / price

        expected_move = abs(price * expected_change)
        if expected_move == 0:
            # fallback small distance
            expected_move = price * 0.001
        
        position = self.get_position(self.asset)
        if position is None:
            self.shares_owned = 0
        else:
            self.shares_owned = float(position.quantity)
        
        asset_value = self.shares_owned * price

        quote_position = self.get_position(self.quote_asset)
        if quote_position is None:
            quote_shares_owned = 0
        else:
            quote_shares_owned = float(quote_position.quantity)
        self.log_message(f"Portfolio assets: {self.shares_owned}, quote assets: {quote_shares_owned}, total value: {asset_value + quote_shares_owned}", color="yellow")

        max_position_size_long = params["max_portfolio_exposure_long"] * self.portfolio_value
        if direction == "long" and (asset_value + value_to_trade) >= max_position_size_long:
            self.log_message(f"Skip placing the oder because low asset value: {asset_value + value_to_trade} > {max_position_size_long}", color="yellow")
            return
        
        max_position_size_short = params["max_portfolio_exposure_short"] * self.portfolio_value
        if direction == "short" and (asset_value - value_to_trade) <= -max_position_size_short:
            self.log_message(f"Skip placing the oder because low asset value: {value_to_trade - asset_value} < {max_position_size_short}", color="yellow")
            return
        

        if direction == "long":
            take_profit_price = price + (expected_move * params["take_profit_multiplier"])
            stop_loss_price = price - (expected_move * params["stop_loss_multiplier"])
            entry_side = "buy"
            exit_side = "sell"
        else:
            take_profit_price = price - (expected_move * params["take_profit_multiplier"])
            stop_loss_price = price + (expected_move * params["stop_loss_multiplier"])
            entry_side = "sell"
            exit_side = "buy"

        if self.is_backtesting:
            order = self.create_order(
                self.asset,
                quantity,
                entry_side,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                quote=self.quote_asset
            )
            self.submit_order(order)
        else:
            # # Market entry
            entry_order = self.create_order(self.asset, quantity, entry_side, quote=self.quote_asset)
            self.submit_order(entry_order)

            # OCO exit (attach after filled)
            exit_order = self.create_order(
                self.asset,
                quantity,
                exit_side,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                position_filled=True,
                quote=self.quote_asset
            )
            self.submit_order(exit_order)

        self.current_position = {
            "side": direction,
            "quantity": quantity,
            "entry_price": price,
            "tp": take_profit_price,
            "sl": stop_loss_price,
        }
        self.log_message(f"Opened {direction.upper()} {quantity:.6f} @ {price:.2f} TP={take_profit_price:.2f} SL={stop_loss_price:.2f}")

    @staticmethod
    def optimize_parameters(
        hist_df: pd.DataFrame,
        start_dt: datetime,
        end_dt: datetime,
        initial_params: dict = None,
        asset_symbol: str = "BTC",
        quote_asset_symbol: str = "USDT",
        benchmark_asset: str = "SPY",
        params_log: str = "optimized_params.csv",
        binary_search_iters: int = None,
        opt_workers: int = None,
        logger=print,
        reuse_model=True,   # <-- new flag
    ):
        """
        Static optimizer. Does NOT require creating a strategy instance.

        Args:
            hist_df: DataFrame of historical minute data (indexed by datetime).
            start_dt, end_dt: datetime objects defining the backtest window.
            initial_params: dict to override default parameters.
            asset_symbol: e.g. "BTC" (used to create Asset for backtest payload).
            quote_asset_symbol: e.g. "USDT".
            params_log: CSV path to append optimized row (compatible with build_training_dataset).
            binary_search_iters: if provided override default.
            opt_workers: override default number of parallel backtests.
            logger: callable for logging (defaults to print).

        Returns:
            best_params (dict) — optimized parameter values (same keys as strategy.parameters)
        """
        base = dict(RandomForestBTCStrategy.parameters)
        if initial_params:
            base.update(initial_params)

        if binary_search_iters is None:
            binary_search_iters = int(base.get("binary_search_iters", BINARY_SEARCH_ITERS))
        if opt_workers is None:
            opt_workers = int(base.get("opt_workers", 3))

        # create Asset + Data payload for Pandas backtesting
        asset = Asset(asset_symbol, "crypto")
        quote_asset = Asset(symbol=quote_asset_symbol, asset_type="forex")
        pandas_data = {asset: Data(asset, hist_df, timestep="minute", quote=quote_asset)}

        # ✅ Attach one global predictor for all runs
        predictor = RandomForestPredictor()
            # Ensure feature set matches live trading
        hist_df = RandomForestBTCStrategy._compute_features(hist_df)
        if reuse_model:
            predictor.train(hist_df, horizon=base["compute_frequency"], force_retrain=False)
        else:
            predictor.train(hist_df, horizon=base["compute_frequency"], force_retrain=True)

        def run_backtest(temp_params):
            try:
                result = RandomForestBTCStrategy.backtest(
                    PandasDataBacktesting,
                    start_dt if isinstance(start_dt, datetime) else start_dt.to_pydatetime(),
                    end_dt if isinstance(end_dt, datetime) else end_dt.to_pydatetime(),
                    pandas_data=pandas_data,
                    benchmark_asset=benchmark_asset,
                    quote_asset=quote_asset,
                    parameters=temp_params,
                    show_plot=False,
                    save_tearsheet=False,
                    show_tearsheet=False,
                    show_indicators=False,
                    quiet_logs=True,
                    save_stats_file=False,
                )
                return RandomForestBTCStrategy._parse_profit_from_backtest_result(result, logger=logger)
            except Exception as e:
                logger(f"Backtest failed for params {temp_params}: {e}")
                return -np.inf

        # -----------------------------
        # Parameter tuning configuration
        # -----------------------------
        PARAM_CONFIG = [
            {"name": "lookback_period", "coarse_grid": [100, 200, 300], "fine_range": (100, 400), "step": 25, "priority": 1},
            {"name": "price_change_threshold_up", "coarse_grid": [0.003, 0.005, 0.007], "fine_range": (0.002, 0.01), "step": 0.001, "priority": 1},
            {"name": "price_change_threshold_down", "coarse_grid": [-0.003, -0.005, -0.007], "fine_range": (-0.01, -0.002), "step": 0.001, "priority": 1},
            {"name": "fraction_portfolio_per_trade", "coarse_grid": [0.8, 0.9, 1.0], "fine_range": (0.5, 1.0), "step": 0.05, "priority": 2},
            {"name": "take_profit_multiplier", "coarse_grid": [0.8, 1.0, 1.2], "fine_range": (0.5, 1.5), "step": 0.1, "priority": 2},
            {"name": "stop_loss_multiplier", "coarse_grid": [0.3, 0.5, 0.7], "fine_range": (0.2, 0.8), "step": 0.1, "priority": 2},
        ]

        best_params = dict(base)

        # Process parameters in order of priority
        for priority in sorted(set(cfg["priority"] for cfg in PARAM_CONFIG)):
            for cfg in [c for c in PARAM_CONFIG if c["priority"] == priority]:
                pname = cfg["name"]

                # 1) Coarse grid search
                logger(f"\n=== Coarse search for {pname} ===")
                best_val, best_profit = None, -np.inf
                for val in cfg["coarse_grid"]:
                    ptest = dict(best_params)
                    ptest[pname] = int(val) if pname == "lookback_period" else float(val)
                    profit = run_backtest(ptest)
                    logger(f"  Candidate {pname}={val} -> profit={profit:.4f}")
                    if profit > best_profit:
                        best_val, best_profit = val, profit
                best_params[pname] = best_val
                logger(f"✔ Best coarse {pname}={best_val} profit={best_profit:.4f}")

                # 2) Binary search refinement inside fine_range
                low, high = cfg["fine_range"]
                step = cfg["step"]

                logger(f"\n--- Refining {pname} in range ({low}, {high}), step={step} ---")
                for it in range(binary_search_iters):
                    mid = (low + high) / 2
                    candidates = [low, mid, high]
                    profits = []
                    logger(f" Iteration {it+1}/{binary_search_iters} candidates: {candidates}")

                    for c in candidates:
                        val = int(round(c)) if pname == "lookback_period" else round(c, 6)
                        ptest = dict(best_params)
                        ptest[pname] = val
                        profit = run_backtest(ptest)
                        profits.append((val, profit))
                        logger(f"   Candidate {pname}={val} -> profit={profit:.4f}")

                    best_val, best_profit = max(profits, key=lambda x: x[1])
                    best_params[pname] = best_val
                    logger(f"✔ Best so far {pname}={best_val}, profit={best_profit:.4f}")

                    # Narrow range based on winner
                    if best_val == candidates[0]:
                        high = mid
                    elif best_val == candidates[2]:
                        low = mid
                    else:
                        low = max(low, mid - step)
                        high = min(high, mid + step)

                    logger(f" -> New search range ({low}, {high})")

        return best_params
    
    @staticmethod
    def _parse_profit_from_backtest_result(result, logger=print):
        """
        Try several common locations for 'total profit' in different Lumibot versions.
        Return -np.inf if parsing fails.
        """
        try:
            return float(result['total_return'])
        except Exception:
            pass

        logger("Warning: could not parse profit from backtest result; returning -inf")
        return -np.inf
    
def run_optimizer(hist_df, start_dt, end_dt, params_log, window_days, asset_symbol, quote_asset_symbol, benchmark_asset_symbol):
    """Run optimizer for given window and log results"""
    best_params = RandomForestBTCStrategy.optimize_parameters(
        hist_df=hist_df,
        start_dt=start_dt,
        end_dt=end_dt,
        asset_symbol=asset_symbol,
        quote_asset_symbol=quote_asset_symbol,
        benchmark_asset=benchmark_asset_symbol,
        params_log=params_log,
        binary_search_iters=BINARY_SEARCH_ITERS,
        opt_workers=3,
        logger=print,
    )

    # append window_days to the log row
    row = {
        "date": end_dt.strftime("%Y-%m-%d"),
        "window_days": window_days,
        "lookback_period": int(best_params["lookback_period"]),
        "price_change_threshold_up": float(best_params["price_change_threshold_up"]),
        "price_change_threshold_down": float(best_params["price_change_threshold_down"]),
        "take_profit_factor": float(best_params["take_profit_multiplier"]),
        "stop_loss_factor": float(best_params["stop_loss_multiplier"]),
    }

    header = [
        "date",
        "window_days",
        "lookback_period",
        "price_change_threshold_up",
        "price_change_threshold_down",
        "take_profit_factor",
        "stop_loss_factor",
    ]

    file_exists = os.path.exists(params_log)
    with open(params_log, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Finished optimization for {window_days}-day window ending {end_dt}")
    return best_params


def generate_training_dataset(start, end, loopback_days, step_days, historical_folder, params_log, asset_symbol, quote_asset_symbol, benchmark_asset_symbol):
    """Run optimizations for sliding windows and generate training data"""
    binance_symbol = asset_symbol + quote_asset_symbol  # Binance convention

    # load full historical data covering [start, end]
    full_data = get_historical_data(binance_symbol, "1m", start, end, cache_dir=historical_folder)

    # Ensure DataFrame index is datetime and UTC-aware
    full_data.index = pd.to_datetime(full_data.index)
    if full_data.index.tz is None:
        full_data.index = full_data.index.tz_localize("UTC")
    full_data = full_data.sort_index()

    current_start = start
    final_end = end

    while current_start + timedelta(days=loopback_days) <= final_end:
        window_end = current_start + timedelta(days=loopback_days)
        if window_end > final_end:
            break

        # Convert naive datetimes to UTC-aware before slicing
        current_start_aware = pd.to_datetime(current_start).tz_localize("UTC")
        window_end_aware = pd.to_datetime(window_end).tz_localize("UTC")

        # Slice DataFrame for this window
        df_window = full_data[(full_data.index >= current_start_aware) & (full_data.index <= window_end_aware)]
        if df_window.empty:
            current_start += timedelta(days=step_days)
            continue

        try:
            run_optimizer(
                df_window,
                current_start,
                window_end,
                params_log,
                loopback_days,
                asset_symbol=asset_symbol,
                quote_asset_symbol=quote_asset_symbol,
                benchmark_asset_symbol=benchmark_asset_symbol
            )
        except Exception as e:
            print(f"Failed optimization for window ending {window_end}: {e}")

        current_start += timedelta(days=step_days)


def remove_cached_model(model_file, cache_dir, feature_file):
    model_path = os.path.join(cache_dir, model_file)
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
            print(f"[RF] Removed existing model at {model_path} to force retraining.")
        except Exception as e:
            print(f"[RF] Failed to remove existing model: {e}")

    feature_path = os.path.join(cache_dir, feature_file)
    if os.path.exists(feature_path):
        try:
            os.remove(feature_path)
            print(f"[RF] Removed existing feature list at {feature_path} to force retraining.")
        except Exception as e:
            print(f"[RF] Failed to remove existing feature list: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset of optimized parameters over rolling windows")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--loopback-days", type=int, default=10, help="Window length in days")
    parser.add_argument("--step-days", type=int, default=1, help="Sliding step size in days")
    parser.add_argument("--historical-folder", required=True, default="./historical_data", help="Folder with historical data")
    parser.add_argument("--params-log", required=True, default="./optimized_params.csv", help="Path to CSV log of optimized parameters")

    args = parser.parse_args()

    # Convert string arguments to datetime
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    remove_cached_model(MODEL_FILE, CACHE_DIR, FEATURE_FILE)

    generate_training_dataset(
        start_dt,
        end_dt,
        args.loopback_days,
        args.step_days,
        args.historical_folder,
        args.params_log,
        asset_symbol="BTC",
        quote_asset_symbol="USDT",
        benchmark_asset_symbol="SPY"
    )