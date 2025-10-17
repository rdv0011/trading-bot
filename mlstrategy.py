from httpx import HTTPError, Timeout
from lumibot.strategies import Strategy
import time
import pandas as pd
import numpy as np
from mlio import MODEL_DIR
from tradingmodelpredictor import TradingModelPredictor

class XGCatBoostStrategy(Strategy):
    '''
    Machine Learning Trading Strategy using XGBoost/CatBoost
    '''
    
    def initialize(self):
        # Load trained model
        self.predictor = TradingModelPredictor(
            model_dir=self.parameters.get("model_dir", MODEL_DIR),
            model_path=self.parameters.get("model_path", None),
            features_path=self.parameters.get("features_path", None),
            model_type=self.parameters.get("model_type", "xgb"),
            auto_reload = self.parameters.get("auto_reload", True)
        )
        
        # Trading parameters
        self.symbol = self.parameters.get("asset_symbol", "BTC")
        self.stake_pct = self.parameters.get("stake_pct", 0.05)
        self.stop_loss_pct = self.parameters.get("stop_loss_pct", 0.04)
        self.max_hold_hours = self.parameters.get("max_hold_hours", 24)
        self.historical_prices_length = self.parameters.get("historical_prices_length", 500)
        self.historical_prices_unit = self.parameters.get("historical_prices_unit", "minute")
        self.predict_with_signal_num_candles = self.parameters.get("predict_with_signal_num_candles", 600)
        self.predict_with_signal_label_window = self.parameters.get("predict_with_signal_label_window", 200)
        
        # Prediction history for adaptive thresholding
        self.pred_history = []
        self.max_history_size = self.parameters.get("max_history_size", 600)
        
        # Position tracking
        self.entry_price = None
        self.entry_time = None

        # Model reload tracking
        self.last_model_check = time.time()
        self.model_check_interval = self.parameters.get("model_check_interval", 300)  # Check every 5 minutes

        
        # Sleep time between iterations
        self.sleeptime = self.parameters.get("sleeptime", "5M")  # 5 minutes to match training data

        # Log initial model info
        model_info = self.predictor.get_model_info()
        self.log_message(f"🤖 Strategy initialized with {model_info['model_type'].upper()} model")
        self.log_message(f"   Model: {model_info['model_path']}")
        self.log_message(f"   Features: {model_info['num_features']}")
        self.log_message(f"   Auto-reload: {model_info['auto_reload']}")
 
        # Pre-populate prediction history with historical data
        self._initialize_prediction_history()

    def on_trading_iteration(self):
        # Get historical data (need at least 240 candles for features)
        bars = self._get_historical_prices_with_retry(
            self.historical_prices_length, 
            self.historical_prices_unit
        )
        df = bars.df

        # Remove timezone info if present to avoid comparison issues
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Get prediction and signal
        try:
            result = self.predictor.predict_with_signal(
                df, 
                self.pred_history,
                num_candles=self.predict_with_signal_num_candles,
                label_window=self.predict_with_signal_label_window,
                check_reload=True
            )
            
            prediction = result['prediction']
            signal = result['signal']
            max_threshold = result['max_threshold']
            min_threshold = result['min_threshold']
            model_reloaded = result['model_reloaded']

            # Log if model was reloaded
            if model_reloaded:
                self.log_message("🔄 Model was reloaded with new version!")
                model_info = self.predictor.get_model_info()
                self.log_message(f"   New model: {model_info['model_path']}")
                # Clear prediction history when model changes
                self.pred_history = []
            
            # Update prediction history
            self.pred_history.append(prediction)
            if len(self.pred_history) > self.max_history_size:
                self.pred_history.pop(0)

            self.log_message(f"Prediction: {prediction:.6f}, Signal: {signal}, Max Threshold: {max_threshold:.6f}, Min Threshold: {min_threshold:.6f}")

        except Exception as e:
            self.log_message(f"Error in prediction: {e}")
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}")
            return
        
        # Get current position
        position = self.get_position(self.symbol)
        current_price = self.get_last_price(self.symbol)
        
        # Entry logic
        if position is None:
            if signal == 'long':
                # Calculate position size
                cash = self.get_cash()
                qty = (cash * self.stake_pct) / current_price
                
                # Place market order
                order = self.create_order(self.symbol, qty, "buy")
                self.submit_order(order)
                
                self.entry_price = current_price
                self.entry_time = self.get_datetime()

                # Set stop loss
                stop_loss_price = current_price * (1 - self.stop_loss_pct)
                stop_order = self.create_order(
                    self.symbol, 
                    qty, 
                    "sell",
                    stop_price=stop_loss_price,
                    order_type="stop"
                )
                self.submit_order(stop_order)
                
                self.log_message(f"🟢 LONG ENTRY at {current_price}")
                
            elif signal == 'short':
                # Calculate position size for short
                cash = self.get_cash()
                qty = (cash * self.stake_pct) / current_price
                
                # Place market sell order (short)
                order = self.create_order(self.symbol, qty, "sell")
                self.submit_order(order)
                
                self.entry_price = current_price
                self.entry_time = self.get_datetime()
                
                # Set stop loss for short (buy back at higher price)
                stop_loss_price = current_price * (1 + self.stop_loss_pct)
                stop_order = self.create_order(
                    self.symbol, 
                    qty, 
                    "buy",
                    stop_price=stop_loss_price,
                    order_type="stop"
                )
                self.submit_order(stop_order)
                
                self.log_message(f"🔴 SHORT ENTRY at {current_price}, Stop Loss at {stop_loss_price:.2f}")
        
        # Exit logic
        elif position is not None and self.entry_price is not None:
            # Determine if current position is long or short
            is_long = position.quantity > 0
            
            # Calculate performance
            if is_long:
                perf = (current_price / self.entry_price - 1)
            else:
                perf = (self.entry_price / current_price - 1)
            
            hold_time = (self.get_datetime() - self.entry_time).total_seconds() / 3600  # hours
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            # Check for manual stop loss hit (in case stop order didn't trigger)
            if is_long and perf <= -self.stop_loss_pct:
                should_exit = True
                exit_reason = "STOP LOSS (LONG)"
            elif not is_long and perf <= -self.stop_loss_pct:
                should_exit = True
                exit_reason = "STOP LOSS (SHORT)"
            elif hold_time >= self.max_hold_hours:
                should_exit = True
                exit_reason = "MAX HOLD TIME"
            elif is_long and signal == 'short':  # Opposite signal for long
                should_exit = True
                exit_reason = "SIGNAL REVERSAL (LONG->SHORT)"
            elif not is_long and signal == 'long':  # Opposite signal for short
                should_exit = True
                exit_reason = "SIGNAL REVERSAL (SHORT->LONG)"
            
            if should_exit:
                # Cancel any pending stop orders
                self.cancel_open_orders()
                
                # Close position
                if is_long:
                    self.sell_all()
                else:
                    # Close short by buying back
                    order = self.create_order(self.symbol, abs(position.quantity), "buy")
                    self.submit_order(order)
                
                self.log_message(f"🔵 EXIT ({exit_reason}) at {current_price}, PnL: {perf:.2%}")
                
                self.entry_price = None
                self.entry_time = None

    def on_abrupt_closing(self):
        """
        Close all positions on strategy shutdown or error.
        """
        try:
            self.log_message("⚠️  Abrupt closing triggered - closing all positions...")
            
            # Cancel all pending orders
            self.cancel_open_orders()
            
            # Get current position
            position = self.get_position(self.symbol)
            
            if position is not None:
                current_price = self.get_last_price(self.symbol)
                is_long = position.quantity > 0
                
                if is_long:
                    # Close long position
                    self.sell_all()
                    self.log_message(f"🔴 EMERGENCY EXIT (LONG) at {current_price}")
                else:
                    # Close short position
                    order = self.create_order(self.symbol, abs(position.quantity), "buy")
                    self.submit_order(order)
                    self.log_message(f"🔴 EMERGENCY EXIT (SHORT) at {current_price}")
                
                # Calculate final performance if entry price is known
                if self.entry_price is not None:
                    if is_long:
                        perf = (current_price / self.entry_price - 1)
                    else:
                        perf = (self.entry_price / current_price - 1)
                    self.log_message(f"   Final PnL: {perf:.2%}")
            else:
                self.log_message("✅ No positions to close")
                
        except Exception as e:
            self.log_message(f"❌ Error during abrupt closing: {e}")
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}")

    def _initialize_prediction_history(self):
        """
        Pre-populate prediction history with historical predictions
        to enable immediate adaptive thresholding.
        """
        try:
            # Fetch enough historical data for:
            # - 600 predictions needed for adaptive thresholding
            # - ~240 candles for feature calculation warmup
            init_length = self.predict_with_signal_num_candles + 250
            
            self.log_message(f"📊 Fetching {init_length} historical candles for prediction history...")
            bars = self._get_historical_prices_with_retry(init_length, self.historical_prices_unit)
            df = bars.df
            
            # Remove timezone info
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Generate predictions for each candle (rolling window)
            predictions = []
            min_candles_for_features = 250  # Ensure enough data for feature engineering
            
            for i in range(min_candles_for_features, len(df)):
                # Get data up to current point
                df_slice = df.iloc[:i+1]
                
                try:
                    # Get prediction without signal calculation
                    result = self.predictor.predict_with_signal(
                        df_slice, 
                        predictions,
                        num_candles=min(len(predictions), self.predict_with_signal_num_candles),
                        label_window=self.predict_with_signal_label_window,
                        check_reload=False  # Don't check for reload during initialization
                    )
                    predictions.append(result['prediction'])
                except Exception as e:
                    self.log_message(f"⚠️  Error generating historical prediction at index {i}: {e}")
                    continue
            
            # Keep only the most recent predictions
            self.pred_history = predictions[-self.max_history_size:]
            self.log_message(f"✅ Initialized prediction history with {len(self.pred_history)} predictions")
            
            # Verify adaptive thresholding will work
            if len(self.pred_history) >= self.predict_with_signal_num_candles:
                from xgcatboostcore import adaptive_thresholding
                max_th, min_th = adaptive_thresholding(
                    pd.Series(self.pred_history),
                    num_candles=self.predict_with_signal_num_candles,
                    label_window=self.predict_with_signal_label_window
                )
                if not np.isnan(max_th):
                    self.log_message(f"✅ Adaptive thresholding ready (max: {max_th:.6f}, min: {min_th:.6f})")
                else:
                    self.log_message("⚠️  Adaptive thresholding returned NaN - check parameters")
            else:
                self.log_message(f"⚠️  Only {len(self.pred_history)} predictions - need {self.predict_with_signal_num_candles}")
                
        except Exception as e:
            self.log_message(f"⚠️  Failed to initialize prediction history: {e}")
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}")
            # Continue with empty history - will accumulate during trading
    
    def _get_historical_prices_with_retry(self, length, timestep, retries=3, delay=5):
        """
        Fetch historical prices with retry logic for network errors.
        """
        for attempt in range(retries):
            try:
                return self.get_historical_prices(self.symbol, length, timestep)
            except (HTTPError, Timeout) as e:
                if attempt < retries - 1:
                    self.log_message(f"⚠️  Network error fetching prices (attempt {attempt + 1}/{retries}): {e}")
                    time.sleep(delay)
                else:
                    self.log_message(f"❌ Failed to fetch prices after {retries} attempts")
                    raise