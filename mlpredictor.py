from tqdm import tqdm
from mlio import load_model, get_latest_model_paths
from timeframe_config import TimeframeConfig
from mltrainingcore import SEED_BASE, create_model, adaptive_thresholding
from mltrainingcore import resolve_model_class, TARGET_COLUMN, predict_param_dicts_from_model
import pandas as pd
import numpy as np
import gc

class MlPredictor:
    """
    Helper class for loading and using trained models in Lumibot strategies.
    Automatically detects and reloads new models when available.
    """

    REGIME_STAKE_MULT = {
        "trend": 1.0,
        "high_vol": 0.5,
        "chop": 0.0
    }
    
    def __init__(self, model_dir, model_type, model_params, df_hist, features,
                 tf_cfg: TimeframeConfig, target_col, auto_reload=True):
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
        self.max_history_size = tf_cfg.max_history_candles
        self.num_candles = tf_cfg.adaptive_history_candles
        self.label_window = tf_cfg.label_window_candles
        self.target_col = target_col
        
        model_path, meta_path = get_latest_model_paths(model_type, self.model_dir)
        print(f"🔍 Auto-detected latest {model_type.upper()} model")
        
        self.current_model_path = model_path
        self.current_meta_path = meta_path
        
        model, metadata = load_model(model_path, meta_path)
        self.model = model
        self.metadata = metadata
        self.pred_history = []

        self.make_prediction_history(df_hist, features, tf_cfg=tf_cfg)

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
    
    def make_prediction_history(
        self, 
        df: pd.DataFrame, 
        features: list, 
        tf_cfg: TimeframeConfig
    ):
        """
        Pre-populate prediction history with historical predictions
        to enable immediate adaptive thresholding using tf_cfg.
        
        Args:
            df (pd.DataFrame): Historical DataFrame already containing features.
            features (list): List of feature column names corresponding to df.
            tf_cfg (TimeframeConfig): Timeframe configuration object.
        """
        try:
            n = len(df)
            window = max(50, tf_cfg.min_feature_candles)
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
            self.pred_history = preds[-tf_cfg.max_history_candles:]
            print(f"✅ Initialized prediction history with {len(self.pred_history)} predictions")

            # Adaptive thresholding using tf_cfg
            hist_len = tf_cfg.adaptive_history_candles
            if len(self.pred_history) < hist_len:
                print("⚠️ Not enough prediction history for adaptive thresholding")
                return

            adaptive_max, adaptive_min = adaptive_thresholding(
                pd.Series(self.pred_history[-hist_len:]),
                tf_cfg
            )
            if not np.isnan(adaptive_max):
                print(f"✅ Adaptive thresholding ready (max: {adaptive_max:.6f}, min: {adaptive_min:.6f})")
            else:
                print("⚠️ Adaptive thresholding returned NaN - check tf_cfg and history length")

        except Exception as e:
            print(f"⚠️ Failed to initialize prediction history: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    def predict_with_signal(
        self,
        df,
        features,
        tf_cfg,  # <- TimeframeConfig object
        random_seed=42,
    ):
        """
        Regime-aware prediction & adaptive signal generation
        using TimeframeConfig-based adaptive thresholding.
        """

        # --- Train model ---
        X = df[features]
        y = df[self.target_col]

        mdl = create_model(self.model_cls, random_seed, self.model_params)
        mdl.fit(X, y)

        # --- Predict ---
        last_row = X.iloc[[-1]]
        prediction = float(mdl.predict(last_row)[0])

        # --- Regime ---
        regime = df.iloc[-1].get("regime", "trend")
        stake_mult = self.REGIME_STAKE_MULT.get(regime, 0.0)

        # --- Adaptive thresholds using tf_cfg ---
        hist_len = tf_cfg.adaptive_history_candles
        pred_series = pd.Series(list(self.pred_history) + [prediction])
        
        if len(pred_series) < hist_len:
            # Not enough history yet
            max_th, min_th = np.nan, np.nan
        else:
            max_th, min_th = adaptive_thresholding(
                pd.Series(pred_series[-hist_len:]),
                tf_cfg
            )

        # --- Signal logic ---
        if stake_mult == 0.0 or np.isnan(max_th):
            signal = "hold"
        elif prediction > max_th:
            signal = "long"
        elif prediction < min_th:
            signal = "short"
        else:
            signal = "hold"

        # --- Update history ---
        self.pred_history.append(prediction)
        if len(self.pred_history) > self.max_history_size:
            self.pred_history.pop(0)

        return {
            "prediction": prediction,
            "signal": signal,
            "regime": regime,
            "stake_mult": stake_mult,
            "max_threshold": max_th,
            "min_threshold": min_th,
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
