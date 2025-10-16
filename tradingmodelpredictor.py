from mlio import load_model, load_feature_columns
from xgcatboostcore import make_features, adaptive_thresholding
import pandas as pd
import numpy as np

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
    """
    
    def __init__(self, model_path, features_path, model_type='xgb'):
        """
        Initialize predictor with saved model and features.
        
        Args:
            model_path: Path to saved model file
            features_path: Path to saved features file
        """
        self.model = load_model(model_path, model_type=model_type)
        self.features = load_feature_columns(features_path, model_type=model_type)
        print(f"✅ Predictor initialized with {len(self.features)} features")
    
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
    
    def predict(self, df):
        """
        Predict future return for the latest candle.
        
        Args:
            df: DataFrame with OHLCV data (must have at least 240 candles for features)
        
        Returns:
            float: Predicted future return
        """
        if len(df) < 240:
            raise ValueError(f"Need at least 240 candles for feature calculation, got {len(df)}")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Predict on the last row
        prediction = self.model.predict(X.iloc[[-1]])[0]
        
        return prediction
    
    def predict_with_signal(self, df, pred_history, num_candles=600, label_window=200):
        """
        Predict future return and generate trading signal using adaptive thresholding.
        
        Args:
            df: DataFrame with OHLCV data
            pred_history: List or Series of recent predictions for threshold calculation
            num_candles: Number of candles to use for threshold calculation
            label_window: Window size for threshold calculation
        
        Returns:
            dict: {
                'prediction': float,
                'signal': str ('long', 'short', or 'hold'),
                'max_threshold': float,
                'min_threshold': float
            }
        """
        # Get prediction
        prediction = self.predict(df)
        
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
            'min_threshold': min_th
        }