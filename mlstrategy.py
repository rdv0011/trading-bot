from ccxtstrategy import CCXTStrategy
import time
from mlio import MODEL_DIR
from tradingmodelpredictor import TradingModelPredictor
from xgcatboostcore import MAX_HISTORY_SIZE, PREDICT_WITH_SIGNAL_LABEL_WINDOW, PREDICT_WITH_SIGNAL_NUM_CANDLES
from xgcatboostcore import TARGET_COLUMN, NUMBER_OF_CANDLES_AHEAD, MIN_CANDLES_FOR_FEATURES
from xgcatboostcore import get_features, make_features, make_labels

MIN_TRADEABLE_QUANTITY = 0.001  # Minimum tradeable quantity for BTC on Binance
TRADEABLE_QUANTITY_PRECISION = 3  # Binance allows BTC quantities to 3 decimal places
BUY_SLIPPAGE = 1.005  # 0.5% slippage on buy orders
SELL_SLIPPAGE = 0.995  # 0.5% slippage on sell orders

class MissingHistoricalDataError(Exception):
    """Raised when get_historical_prices() returns None"""
    pass

class XGCatBoostStrategy(CCXTStrategy):
    '''
    Machine Learning Trading Strategy using XGBoost/CatBoost
    '''
    
    def initialize(self):
        # Trading parameters
        self.asset = self.parameters.get("asset_symbol", "BTC")
        self.historical_prices_length = self.parameters.get("historical_prices_length", 500)
        self.historical_prices_unit = self.parameters.get("historical_prices_unit", "5m")
        self.predict_with_signal_num_candles = self.parameters.get("predict_with_signal_num_candles", PREDICT_WITH_SIGNAL_NUM_CANDLES)
        self.predict_with_signal_label_window = self.parameters.get("predict_with_signal_label_window", PREDICT_WITH_SIGNAL_LABEL_WINDOW)
        self.max_history_size = self.parameters.get("max_history_size", MAX_HISTORY_SIZE)
        
        # Position tracking
        self.entry_price = None
        self.entry_time = None

        # Model reload tracking
        self.last_model_check = time.time()
        self.model_check_interval = self.parameters.get("model_check_interval", 300)  # Check every 5 minutes
        
        # Sleep time between iterations
        self.sleeptime = self.parameters.get("sleeptime", 300)  # 5 minutes to match training data

        # Prediction history for adaptive thresholding
        init_length = (
            MIN_CANDLES_FOR_FEATURES
            + self.predict_with_signal_num_candles
            + NUMBER_OF_CANDLES_AHEAD
        )
        self.log_message(f"📊 Fetching {init_length} historical candles for prediction history...")
        
        df_hist = self.get_historical_prices(self.asset, init_length, self.historical_prices_unit)
        if df_hist is None or len(df_hist) == 0:
            msg = (
                f"❌ Initialization aborted: get_historical_prices() returned None or empty "
                f"for asset={self.asset}, candles={init_length}, interval={self.historical_prices_unit}"
            )
            self.log_message(msg)
            raise MissingHistoricalDataError(msg)
        df_hist = make_features(df_hist)
        df_hist = make_labels(df_hist, H=NUMBER_OF_CANDLES_AHEAD)
        features = get_features(df_hist)

        self.predictor = TradingModelPredictor(
            model_dir=self.parameters.get("model_dir", MODEL_DIR),
            model_type=self.parameters.get("model_type", "cat"),
            model_params=self.parameters.get("model_params", {'iterations': 500, 'verbose': False}),
            df_hist = df_hist,
            features=features,
            min_candles_for_features=MIN_CANDLES_FOR_FEATURES,
            num_candles=self.predict_with_signal_num_candles,
            label_window=self.predict_with_signal_label_window,
            max_history_size=self.max_history_size,
            target_col=TARGET_COLUMN,
        )

        # Log initial model info
        model_info = self.predictor.get_model_info()
        self.log_message(f"🤖 Strategy initialized")
        for key, value in model_info.items():
            pretty_key = key.replace("_", " ").title()
            self.log_message(f"   {pretty_key}: {value}")

    def on_trading_iteration(self):
        # Get historical data (need at least 240 candles for features)
        df = self.get_historical_prices(self.asset, self.historical_prices_length, self.historical_prices_unit)

        if df is None or len(df) == 0:
            self.log_message(
                f"❌ Skipping iteration: get_historical_prices() returned None or empty "
                f"for asset={self.asset}, candles={self.historical_prices_length}, interval={self.historical_prices_unit}"
            )
            return

        # Update dynamic meta parameters from model
        try:
            meta_params = self.predictor.predict_meta_params(df)
            self.stake_long_pct = meta_params["stake_long_pct"]
            self.stake_short_pct = meta_params["stake_short_pct"]
            self.stop_loss_pct = meta_params["stop_loss_pct"]
            self.take_profit_pct = meta_params["take_profit_pct"]
            self.max_hold_hours = meta_params["max_hold_hours"]

            self.log_message(
                f"Updated meta parameters: stake_long_pct={self.stake_long_pct:.3f}, "
                f"stake_short_pct={self.stake_short_pct:.3f}, "
                f"stop_loss_pct={self.stop_loss_pct:.3f}, take_profit_pct={self.take_profit_pct:.3f}, "
                f"max_hold_hours={self.max_hold_hours:.1f}"
            )
        except Exception as e:
            self.log_message(f"⚠️ Failed to update meta parameters, using previous values: {e}")


        # Get prediction and signal
        if len(df) < 240:
            print(f"Need at least 240 candles, got {len(df)}")
            return
    
        df = make_features(df)
        df = make_labels(df, H=NUMBER_OF_CANDLES_AHEAD)
        features = get_features(df)

        result = self.predictor.predict_with_signal(
            df, 
            features=features,
        )
        
        prediction = result['prediction']
        signal = result['signal']
        max_threshold = result['max_threshold']
        min_threshold = result['min_threshold']

        self.log_message(f"Prediction: {prediction:.6f}, Signal: {signal}, Max Threshold: {max_threshold:.6f}, Min Threshold: {min_threshold:.6f}")
        
        # Get current position
        position = self.get_position(self.asset)
        current_price = self.get_last_price(self.asset)

        # Check if we have a meaningful position
        has_position = position is not None and abs(position) >= MIN_TRADEABLE_QUANTITY

        # Entry logic
        if not has_position:
            # Reset entry tracking if position was closed
            if self.entry_price is not None:
                # Close any remaining open orders (e.g., stop/take profit) if position closed externally
                self.cancel_open_orders(self.asset)
                self.entry_price = None
                self.entry_time = None
            
            if signal == 'long':
                self._enter_long_position(current_price)
            elif signal == 'short':
                self._enter_short_position(current_price)
        
        # Exit logic
        elif has_position and self.entry_price is not None:
            self._check_exit_conditions(position, current_price, signal)

    def _enter_long_position(self, current_price: float):
        """Enter long position with bracket orders"""
        # Calculate position size
        cash = self.get_cash()
        qty = (cash * self.stake_long_pct) / current_price
        qty = round(qty, TRADEABLE_QUANTITY_PRECISION)
        
        if qty < MIN_TRADEABLE_QUANTITY:
            self.log_message(f"⚠️  Order quantity {qty:.3f} BTC below minimum {MIN_TRADEABLE_QUANTITY}, skipping")
            return
        
        # Calculate bracket order prices
        stop_loss_price = round(current_price * (1 - self.stop_loss_pct), 2)
        take_profit_price = round(current_price * (1 + self.take_profit_pct), 2)
        
        # Place market buy order
        order = self.create_order(self.asset, qty, "buy")
        self.submit_order(order)

        # Place stop loss order
        stop_order = self.create_order(
            self.asset, qty, "sell",
            stop_price=stop_loss_price,
            limit_price=stop_loss_price * SELL_SLIPPAGE,
            order_type="stop_limit"
        )
        self.submit_order(stop_order)

        # Place take profit order
        tp_order = self.create_order(
            self.asset, qty, "sell",
            limit_price=take_profit_price,
            order_type="limit"
        )
        self.submit_order(tp_order)
        
        self.entry_price = current_price
        self.entry_time = self.get_datetime()
        
        self.log_message(f"🟢 LONG ENTRY at {current_price}, Stop Loss at {stop_loss_price:.2f}, Take Profit at {take_profit_price:.2f}, Qty: {qty:.3f}")
                
    def _enter_short_position(self, current_price: float):
        """Enter short position with bracket orders"""
        # Calculate position size for short
        cash = self.get_cash()
        qty = (cash * self.stake_short_pct) / current_price
        qty = round(qty, TRADEABLE_QUANTITY_PRECISION)
        
        if qty < MIN_TRADEABLE_QUANTITY:
            self.log_message(f"⚠️  Order quantity {qty:.3f} BTC below minimum {MIN_TRADEABLE_QUANTITY}, skipping")
            return
        
        # Calculate exit prices (inverted for short)
        stop_loss_price = round(current_price * (1 + self.stop_loss_pct), 2)
        take_profit_price = round(current_price * (1 - self.take_profit_pct), 2)
        
        # Place market sell order (short)
        order = self.create_order(self.asset, qty, "sell")
        self.submit_order(order)

        # Place stop loss order for short (buy back at higher price)
        stop_order = self.create_order(
            self.asset, qty, "buy",
            stop_price=stop_loss_price,
            limit_price=stop_loss_price * BUY_SLIPPAGE,
            order_type="stop_limit"
        )
        self.submit_order(stop_order)

        # Place take profit order for short (buy back at lower price)
        tp_order = self.create_order(
            self.asset, qty, "buy",
            limit_price=take_profit_price,
            order_type="limit"
        )
        self.submit_order(tp_order)
        
        self.entry_price = current_price
        self.entry_time = self.get_datetime()
        
        self.log_message(f"🔴 SHORT ENTRY at {current_price}, Stop Loss at {stop_loss_price:.2f}, Take Profit at {take_profit_price:.2f}, Qty: {qty:.3f}")
        
    def _check_exit_conditions(self, position, current_price: float, signal: str):
        """Check if position should be exited"""
        is_long = position > 0
        
        # Calculate performance
        if is_long:
            perf = (current_price / self.entry_price - 1)
        else:
            perf = (self.entry_price / current_price - 1)
        
        hold_time = (self.get_datetime() - self.entry_time).total_seconds() / 3600  # hours
        
        # Exit conditions
        should_exit = False
        exit_reason = ""
        
        # Check max hold time
        if hold_time >= self.max_hold_hours:
            should_exit = True
            exit_reason = "MAX HOLD TIME"
        # Check for signal reversal
        elif is_long and signal == 'short':
            should_exit = True
            exit_reason = "SIGNAL REVERSAL (LONG->SHORT)"
        elif not is_long and signal == 'long':
            should_exit = True
            exit_reason = "SIGNAL REVERSAL (SHORT->LONG)"

        if should_exit:
            self._exit_position(position, current_price, exit_reason, perf)
        
    def _exit_position(self, position, current_price: float, exit_reason: str, perf: float):
        """Exit current position"""
        # Cancel any pending orders
        self.cancel_open_orders()
        
        # Double-check position still exists and is above minimum
        position = self.get_position(self.asset)
        if position is not None and abs(position) >= MIN_TRADEABLE_QUANTITY:
            is_long = position > 0
            side = "sell" if is_long else "buy"
            order = self.create_order(
                self.asset,
                abs(position),
                side
            )
            self.submit_order(order)
            
            self.log_message(f"🔵 EXIT ({exit_reason}) at {current_price}, PnL: {perf:.2%}")
        else:
            self.log_message(f"✅ Position already closed or too small (possibly by stop/take profit), reason would be: {exit_reason}, PnL: {perf:.2%}")
        
        self.entry_price = None
        self.entry_time = None

    def on_abrupt_closing(self):
        """
        Close all positions on strategy shutdown or error.
        """
        try:
            self.log_message("⚠️  Abrupt closing triggered - closing all positions...")
            
            # Cancel all pending orders
            self.cancel_open_orders(self.asset)
            
            # Get current position
            position = self.get_position(self.asset)

            if position is not None and abs(position) >= MIN_TRADEABLE_QUANTITY:
                current_price = self.get_last_price(self.asset)
                is_long = position > 0

                side = "sell" if is_long else "buy"

                order = self.create_order(
                    self.asset,
                    abs(position),
                    side,
                    order_type="market"
                )
                self.submit_order(order)
                
                position_type = "LONG" if is_long else "SHORT"
                self.log_message(f"🔴 EMERGENCY EXIT ({position_type}) at {current_price}")

                # Calculate final performance if entry price is known
                if self.entry_price is not None:
                    if is_long:
                        perf = (current_price / self.entry_price - 1)
                    else:
                        perf = (self.entry_price / current_price - 1)
                    self.log_message(f"   Final PnL: {perf:.2%}")
            else:
                self.log_message("✅ Position was already closed or too small (possibly by stop/take profit)")
                
        except Exception as e:
            self.log_message(f"❌ Error during abrupt closing: {e}")
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}")