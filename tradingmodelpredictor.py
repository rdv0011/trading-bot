from tqdm import tqdm
from mlio import load_model, get_latest_model_paths
from xgcatboostcore import SEED_BASE, create_model, adaptive_thresholding
from xgcatboostcore import resolve_model_class, TARGET_COLUMN, predict_param_dicts_from_model
import pandas as pd
import numpy as np
import gc

# =============================================
# Prediction Helper Functions
# =============================================
# =============================================
# # Example
# =============================================
# Load the predictor
# predictor = TradingModelPredictor(
#     model_path='models/xgb_model_20251015_234550.pkl',
#     features_path='models/xgb_features_20251015_234550.pkl'
# )

# # Fetch fresh data
# exchange = ccxt.binance()
# ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=500)
# df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
# df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
# df = df.set_index('date').sort_index()

# # Predict with signal
# pred_history = []  # In real usage, maintain this across calls
# result = predictor.predict_with_signal(df, pred_history)

# print(f"\n🔮 PREDICTION RESULT:")
# print(f"   Predicted Return: {result['prediction']:.6f}")
# print(f"   Trading Signal: {result['signal'].upper()}")
# print(f"   Max Threshold: {result['max_threshold']:.6f}")
# print(f"   Min Threshold: {result['min_threshold']:.6f}")
# print(f"   Current BTC Price: ${df['close'].iloc[-1]:,.2f}")
class TradingModelPredictor:
    """
    Helper class for loading and using trained models in Lumibot strategies.
    Automatically detects and reloads new models when available.
    """
    
    def __init__(self, model_dir, model_type, model_params, df_hist, features,
                 min_candles_for_features, num_candles, label_window, max_history_size, target_col, auto_reload=True):
        """
        Initialize predictor with saved model and features.
        
        Args:
            model_type: 'xgb' or 'cat' - used when auto-detecting models
            auto_reload: Whether to automatically check for and reload new models
        """
        self.model_type = model_type
        self.model_cls = resolve_model_class(model_type)
        self.model_params = model_params
        self.auto_reload = auto_reload
        self.model_dir = model_dir
        self.max_history_size = max_history_size
        self.num_candles = num_candles
        self.label_window = label_window
        self.target_col = target_col
        
        model_path, meta_path = get_latest_model_paths(model_type, self.model_dir)
        print(f"🔍 Auto-detected latest {model_type.upper()} model")
        
        self.current_model_path = model_path
        self.current_meta_path = meta_path
        
        model, metadata = load_model(model_path, meta_path)
        self.model = model
        self.metadata = metadata
        self.pred_history = []

        self.make_prediction_history(df_hist, features, min_candles_for_features, max_history_size)

        print(f"✅ Predictor initialized with model: {self.current_model_path}")
    
    def check_for_new_model(self):
        """
        Check if a newer model is available and reload if found.
        
        Returns:
            bool: True if a new model was loaded, False otherwise
        """
        
        try:
            # Get latest model paths
            latest_model_path, latest_meta_path = get_latest_model_paths(
                self.model_type, 
                self.model_dir
            )
            
            # Check if paths have changed
            if (latest_model_path != self.current_model_path or 
                latest_meta_path != self.current_meta_path):
                
                print(f"\n🔄 New model detected! Reloading...")
                print(f"   Old model: {self.current_model_path}")
                print(f"   New model: {latest_model_path}")
                
                # Unload old model and clear memory
                old_model = self.model
                self.model = None
                del old_model
                gc.collect()  # Force garbage collection
                
                # Load new model and features
                self.model, self.metadata = load_model(latest_model_path, latest_meta_path)
                
                # Update current paths
                self.current_model_path = latest_model_path
                self.current_meta_path = latest_meta_path
                
                print(f"✅ Model reload complete! Now using {len(self.features)} features")
                return True
            
        except Exception as e:
            print(f"⚠️  Error checking for new model: {e}")
            return False
        
        return False
    
    def make_prediction_history(self, df: pd.DataFrame, features: list, min_candles_for_features: int, max_history_size: int):
        """
        Pre-populate prediction history with historical predictions
        to enable immediate adaptive thresholding.

        Args:
            df (pd.DataFrame): Historical DataFrame already containing features.
            features (list): List of feature column names corresponding to df.
        """
        try:
            n = len(df)
            window = max(50, min_candles_for_features)
            preds = []
            mdl = None
            retrain_every = 12  # Retrain every N candles for speed

            for i in tqdm(range(window, n), desc="Initializing prediction history", unit="candle"):
                if i % retrain_every == 0 or mdl is None:
                    # Train on rolling window
                    train_df = df.iloc[i - window:i]
                    X_train, y_train = train_df[features], train_df[TARGET_COLUMN]

                    random_seed = SEED_BASE + i - window
                    mdl = create_model(self.model_cls, random_seed, self.model_params)
                    mdl.fit(X_train, y_train)

                # Predict last row of current slice
                X_pred = df.iloc[[i]][features]
                p = float(mdl.predict(X_pred)[0])
                preds.append(p)

            # Keep only most recent predictions
            self.pred_history = preds[-max_history_size:]
            print(f"✅ Initialized prediction history with {len(self.pred_history)} predictions")

            # Verify adaptive thresholding will work
            max_th, min_th = adaptive_thresholding(
                pd.Series(self.pred_history),
                num_candles=min(self.num_candles, len(self.pred_history)),
                label_window=self.label_window
            )
            if not np.isnan(max_th):
                print(f"✅ Adaptive thresholding ready (max: {max_th:.6f}, min: {min_th:.6f})")
            else:
                print("⚠️ Adaptive thresholding returned NaN - check parameters")

        except Exception as e:
            print(f"⚠️ Failed to initialize prediction history: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    def predict_with_signal(
        self,
        df,
        features,
        random_seed=42,
    ):
        """
        Train a model on the provided historical features and target, then predict
        the last row and generate a trading signal using adaptive thresholding.

        Args:
            df (pd.DataFrame): DataFrame containing features and target column.
            target_col (str): Name of the target column.
            features (list): List of feature column names.
            random_seed (int): Random seed for reproducibility.
            num_candles (int): Number of candles for adaptive thresholding.
            label_window (int): Window size for adaptive thresholding.

        Returns:
            dict: {
                'prediction': float,
                'signal': str ('long', 'short', 'hold'),
                'max_threshold': float,
                'min_threshold': float
            }
        """

        # Prepare features and target
        X = df[features]
        y = df[self.target_col]

        # Create model instance
        mdl = create_model(self.model_cls, random_seed, self.model_params)

        # Train the model on historical data
        mdl.fit(X, y)

        # Predict on last row
        last_row = X.iloc[[-1]]
        prediction = float(mdl.predict(last_row)[0])

        # Calculate adaptive thresholds
        pred_series = pd.Series(list(self.pred_history) + [prediction])
        max_th, min_th = adaptive_thresholding(
            df=pred_series,
            num_candles=min(self.num_candles, len(self.pred_history)),
            label_window=self.label_window
            )

        # Generate trading signal
        if np.isnan(max_th):
            signal = "hold"
        elif prediction > max_th:
            signal = "long"
        elif prediction < min_th:
            signal = "short"
        else:
            signal = "hold"

        # Update prediction history
        self.pred_history.append(prediction)
        if len(self.pred_history) > self.max_history_size:
            self.pred_history.pop(0)

        return {
            "prediction": prediction,
            "signal": signal,
            "max_threshold": max_th,
            "min_threshold": min_th
        }
    
    def get_model_info(self):
        """
        Get information about the currently loaded model.
        
        Returns:
            dict: Model information including path, type, and feature count
        """
        return {
            'model_type': self.model_type,
            'model_path': self.current_model_path,
            'meta_path': self.current_meta_path,
            'auto_reload': self.auto_reload
        }
    
    def predict_meta_params(self, df):
        """
        Predict meta parameters using the *metadata-trained* model.
        """
        
        if self.auto_reload:
            self.check_for_new_model()

        # Use the EXACT feature_cols used in labeling
        feature_cols = self.metadata["feature_cols"]

        df_feat = df.copy()

        # Add missing feature columns (important for live trading)
        for col in feature_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0

        # Live inference must use fillna(0)
        X_live = df_feat[feature_cols].fillna(0)

        # Predict only last row
        X_last = X_live.iloc[[-1]]

        return predict_param_dicts_from_model(self.model, self.metadata, X_last)
