import pickle
from pathlib import Path
import pandas as pd
import os
import sys

# Detect the directory of the main script (e.g., xgcatboost.py)
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
MODEL_DIR = Path(script_dir) / "models"
MODEL_DIR.mkdir(exist_ok=True)

# =============================================
# Model Persistence Functions
# =============================================
def _cleanup_old_files(pattern, model_dir, keep_count=2):
    """
    Remove old files, keeping only the most recent ones.
    
    Args:
        pattern: Glob pattern to match files
        model_dir: Directory containing files
        keep_count: Number of most recent files to keep
    """
    files = list(model_dir.glob(pattern))
    if len(files) <= keep_count:
        return
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Remove older files
    for old_file in files[keep_count:]:
        old_file.unlink()
        print(f"🗑️  Removed old file: {old_file.name}")

# =============================================
# Model Persistence Functions
# =============================================
def save_model(model, model_type, model_dir=MODEL_DIR, keep_count=2):
    """
    Save trained model to disk with timestamp.
    Keeps only the most recent models to save disk space.
    
    Args:
        model: Trained XGBoost or CatBoost model
        model_type: 'xgb' or 'cat'
        model_dir: Directory to save models
        keep_count: Number of recent models to keep (default: 2)
    
    Returns:
        Path to saved model file
    """
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_type}_model_{timestamp}.pkl"
    filepath = model_dir / filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved: {filepath}")
    
    # Cleanup old models
    _cleanup_old_files(f"{model_type}_model_*.pkl", model_dir, keep_count)
    
    return filepath

def load_model(filepath=None, model_type=None, model_dir=MODEL_DIR):
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to saved model file (optional)
        model_type: 'xgb' or 'cat' - required if filepath is None
        model_dir: Directory containing models
    
    Returns:
        Loaded model
    """
    if filepath is None:
        if model_type is None:
            raise ValueError("Either filepath or model_type must be provided")
        
        # Find the most recent model file
        pattern = f"{model_type}_model_*.pkl"
        files = list(model_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No model files found matching pattern: {pattern}")
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        filepath = files[0]
        print(f"📂 Loading latest model: {filepath.name}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Model loaded: {filepath}")
    return model

def save_feature_columns(features, model_type, model_dir=MODEL_DIR, keep_count=2):
    """
    Save feature column names for consistent prediction.
    Keeps only the most recent feature files to save disk space.
    
    Args:
        features: List of feature column names
        model_type: 'xgb' or 'cat'
        model_dir: Directory to save feature list
        keep_count: Number of recent feature files to keep (default: 2)
    
    Returns:
        Path to saved feature file
    """
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_type}_features_{timestamp}.pkl"
    filepath = model_dir / filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(features, f)
    
    print(f"✅ Features saved: {filepath}")
    
    # Cleanup old feature files
    _cleanup_old_files(f"{model_type}_features_*.pkl", model_dir, keep_count)
    
    return filepath

def load_feature_columns(filepath=None, model_type=None, model_dir=MODEL_DIR):
    """
    Load feature column names.
    
    Args:
        filepath: Path to saved feature file (optional)
        model_type: 'xgb' or 'cat' - required if filepath is None
        model_dir: Directory containing feature files
    
    Returns:
        List of feature column names
    """
    if filepath is None:
        if model_type is None:
            raise ValueError("Either filepath or model_type must be provided")
        
        # Find the most recent feature file
        pattern = f"{model_type}_features_*.pkl"
        files = list(model_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No feature files found matching pattern: {pattern}")
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        filepath = files[0]
        print(f"📂 Loading latest features: {filepath.name}")
    
    with open(filepath, 'rb') as f:
        features = pickle.load(f)
    
    return features

def get_latest_model_paths(model_type, model_dir=MODEL_DIR):
    """
    Get paths to the latest model and features files.
    
    Args:
        model_type: 'xgb' or 'cat'
        model_dir: Directory containing model files
    
    Returns:
        tuple: (model_path, features_path)
    """
    # Find latest model
    model_files = list(model_dir.glob(f"{model_type}_model_*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model files found for type: {model_type}")
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    model_path = model_files[0]
    
    # Find latest features
    feature_files = list(model_dir.glob(f"{model_type}_features_*.pkl"))
    if not feature_files:
        raise FileNotFoundError(f"No feature files found for type: {model_type}")
    feature_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    features_path = feature_files[0]
    
    return str(model_path), str(features_path)