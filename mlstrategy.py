from httpx import HTTPError, Timeout
from lumibot.strategies import Strategy
from datetime import time, timedelta
from tradingmodelpredictor import TradingModelPredictor

class MLTradingStrategy(Strategy):
    '''
    Machine Learning Trading Strategy using XGBoost/CatBoost
    '''
    
    def initialize(self):
        # Load trained model
        self.predictor = TradingModelPredictor(
            model_path=self.parameters.get("model_path", None),
            features_path=self.parameters.get("features_path", None),
            model_type=self.parameters.get("model_type", "xgb")
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
        
        # Sleep time between iterations
        self.sleeptime = "5M"  # 5 minutes to match training data
    
    def on_trading_iteration(self):
        # Get historical data (need at least 240 candles for features)
        bars = self._get_historical_prices_with_retry(
            self.historical_prices_length, 
            self.historical_prices_unit
        )
        df = bars.df
        
        # Get prediction and signal
        try:
            result = self.predictor.predict_with_signal(
                df, 
                self.pred_history,
                num_candles=self.predict_with_signal_num_candles,
                label_window=self.predict_with_signal_label_window
            )
            
            prediction = result['prediction']
            signal = result['signal']
            
            # Update prediction history
            self.pred_history.append(prediction)
            if len(self.pred_history) > self.max_history_size:
                self.pred_history.pop(0)
            
            self.log_message(f"Prediction: {prediction:.6f}, Signal: {signal}")
            
        except Exception as e:
            self.log_message(f"Error in prediction: {e}")
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
                self.log_message(f"🟢 LONG ENTRY at {current_price}")
                
            elif signal == 'short':
                # For crypto, you might need to use futures or skip shorts
                # This is a placeholder - adjust based on your broker's capabilities
                self.log_message(f"🔴 SHORT SIGNAL at {current_price} (skipped - not implemented)")
        
        # Exit logic
        elif position is not None and self.entry_price is not None:
            # Calculate performance
            perf = (current_price / self.entry_price - 1)
            hold_time = (self.get_datetime() - self.entry_time).total_seconds() / 3600  # hours
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            if perf <= -self.stop_loss_pct:
                should_exit = True
                exit_reason = "STOP LOSS"
            elif hold_time >= self.max_hold_hours:
                should_exit = True
                exit_reason = "MAX HOLD TIME"
            elif signal == 'short':  # Opposite signal
                should_exit = True
                exit_reason = "SIGNAL REVERSAL"
            
            if should_exit:
                # Close position
                self.sell_all()
                self.log_message(f"🔵 EXIT ({exit_reason}) at {current_price}, PnL: {perf:.2%}")
                
                self.entry_price = None
                self.entry_time = None
    
    def _get_historical_prices_with_retry(self, length, timestep, retries=3, delay=5):
        attempt = 0
        while attempt < retries:
            try:
                # Example call (depends on your setup)
                data = self.get_historical_prices(
                    asset=self.symbol, length=length, timestep=timestep, quote=self.quote_asset
                )
                return data
            except (ConnectionError, Timeout, HTTPError) as e:
                self.log_message(f"Connection failed: {e}. Retrying {attempt+1}/{retries}...", color="red")
                attempt += 1
                time.sleep(delay)
            except Exception as e:
                self.log_message(f"Unexpected error: {e}", color="red")
                break
        return None  # or raise custom exception