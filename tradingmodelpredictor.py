from mlio import MODEL_DIR, load_model, load_feature_columns, get_latest_model_paths
from xgcatboostcore import make_features, adaptive_thresholding
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
    
    def __init__(self, model_dir=MODEL_DIR, model_path=None, features_path=None, model_type='xgb', auto_reload=True):
        """
        Initialize predictor with saved model and features.
        
        Args:
            model_path: Path to saved model file (optional, will auto-detect if None)
            features_path: Path to saved features file (optional, will auto-detect if None)
            model_type: 'xgb' or 'cat' - used when auto-detecting models
            auto_reload: Whether to automatically check for and reload new models
        """
        self.model_type = model_type
        self.auto_reload = auto_reload
        self.model_dir = model_dir
        
        # Load initial model
        if model_path is None or features_path is None:
            model_path, features_path = get_latest_model_paths(model_type, self.model_dir)
            print(f"🔍 Auto-detected latest {model_type.upper()} model")
        
        self.current_model_path = model_path
        self.current_features_path = features_path
        
        self.model = load_model(model_path, model_type=model_type)
        self.features = load_feature_columns(features_path, model_type=model_type)
        print(f"✅ Predictor initialized with {len(self.features)} features")
    
    def check_for_new_model(self):
        """
        Check if a newer model is available and reload if found.
        
        Returns:
            bool: True if a new model was loaded, False otherwise
        """
        if not self.auto_reload:
            return False
        
        try:
            # Get latest model paths
            latest_model_path, latest_features_path = get_latest_model_paths(
                self.model_type, 
                self.model_dir
            )
            
            # Check if paths have changed
            if (latest_model_path != self.current_model_path or 
                latest_features_path != self.current_features_path):
                
                print(f"\n🔄 New model detected! Reloading...")
                print(f"   Old model: {self.current_model_path}")
                print(f"   New model: {latest_model_path}")
                
                # Unload old model and clear memory
                old_model = self.model
                self.model = None
                del old_model
                gc.collect()  # Force garbage collection
                
                # Load new model and features
                self.model = load_model(latest_model_path, model_type=self.model_type)
                self.features = load_feature_columns(latest_features_path, model_type=self.model_type)
                
                # Update current paths
                self.current_model_path = latest_model_path
                self.current_features_path = latest_features_path
                
                print(f"✅ Model reload complete! Now using {len(self.features)} features")
                return True
            
        except Exception as e:
            print(f"⚠️  Error checking for new model: {e}")
            return False
        
        return False
    
    def prepare_features(self, df):
        """
        Prepare features from raw OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with features ready for prediction
        """
        # Apply same feature engineering as training
        df = make_features(df)
        
        # Add missing features with default value of 0
        for feature in self.features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select only the features used during training, in the same order
        return df[self.features]
    
    def predict(self, df, check_reload=True):
        """
        Predict future return for the latest candle.
        
        Args:
            df: DataFrame with OHLCV data (must have at least 240 candles for features)
            check_reload: Whether to check for new models before predicting
        
        Returns:
            float: Predicted future return
        """
        # Check for new model if enabled
        if check_reload and self.auto_reload:
            self.check_for_new_model()
        
        if len(df) < 240:
            raise ValueError(f"Need at least 240 candles for feature calculation, got {len(df)}")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Predict on the last row
        prediction = self.model.predict(X.iloc[[-1]])[0]
        
        return prediction
    
    def predict_with_signal(self, df, pred_history, num_candles=600, label_window=200, check_reload=True):
        """
        Predict future return and generate trading signal using adaptive thresholding.
        
        Args:
            df: DataFrame with OHLCV data
            pred_history: List or Series of recent predictions for threshold calculation
            num_candles: Number of candles to use for threshold calculation
            label_window: Window size for threshold calculation
            check_reload: Whether to check for new models before predicting
        
        Returns:
            dict: {
                'prediction': float,
                'signal': str ('long', 'short', or 'hold'),
                'max_threshold': float,
                'min_threshold': float,
                'model_reloaded': bool
            }
        """
        # Check for new model if enabled
        model_reloaded = False
        if check_reload and self.auto_reload:
            model_reloaded = self.check_for_new_model()
        
        # Get prediction
        prediction = self.predict(df, check_reload=False)  # Already checked
        
        # Calculate adaptive thresholds
        pred_series = pd.Series(list(pred_history) + [prediction])
        max_th, min_th = adaptive_thresholding(pred_series, num_candles, label_window)
        
        # Generate signal
        signal = 'hold'
        if not np.isnan(max_th):
            if prediction > max_th:
                signal = 'long'
            elif prediction < min_th:
                signal = 'short'
        
        return {
            'prediction': prediction,
            'signal': signal,
            'max_threshold': max_th,
            'min_threshold': min_th,
            'model_reloaded': model_reloaded
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
            'features_path': self.current_features_path,
            'num_features': len(self.features),
            'auto_reload': self.auto_reload
        }