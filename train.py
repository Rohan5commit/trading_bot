import os
import pandas as pd
import numpy as np
import yaml
import json
import logging
from datetime import datetime, timedelta, timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, 'config.yaml')

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path
        self.base_dir = os.path.dirname(os.path.abspath(config_path))
        self.feature_store_dir = os.path.join(self.base_dir, 'feature_store')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.registry_path = os.path.join(self.base_dir, 'model_registry.json')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.feature_store_dir, exist_ok=True)
        storage_cfg = (self.config.get("storage") or {}) if isinstance(self.config, dict) else {}
        self.store_feature_files = bool(storage_cfg.get("store_feature_files", True))
        self._feature_engineer = None
        
        # Load registry
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def _get_feature_engineer(self):
        if self._feature_engineer is None:
            from features import FeatureEngineer
            self._feature_engineer = FeatureEngineer(self.config_path)
        return self._feature_engineer

    def _min_retrain_timedelta(self):
        cfg = self.config.get("ml", {}) if isinstance(self.config, dict) else {}
        freq = str(cfg.get("retrain_frequency", "weekly") or "weekly").strip().lower()
        if freq in {"always", "on", "true"}:
            return timedelta(seconds=0)
        if freq in {"never", "off", "false"}:
            return None
        if freq == "daily":
            return timedelta(days=1)
        if freq == "weekly":
            return timedelta(days=7)
        if freq == "monthly":
            return timedelta(days=30)
        # Allow e.g. "3d"
        if freq.endswith("d"):
            try:
                return timedelta(days=int(freq[:-1]))
            except Exception:
                return timedelta(days=7)
        return timedelta(days=7)

    def _should_retrain(self, symbol: str) -> bool:
        window = self._min_retrain_timedelta()
        if window is None:
            return False
        if window.total_seconds() <= 0:
            return True

        entry = self.registry.get(symbol) if isinstance(self.registry, dict) else None
        trained_at = (entry or {}).get("trained_at") if isinstance(entry, dict) else None
        model_id = (entry or {}).get("latest_model") if isinstance(entry, dict) else None
        if not trained_at or not model_id:
            return True

        model_path = os.path.join(self.models_dir, f"{model_id}.npy")
        if not os.path.exists(model_path):
            return True

        try:
            dt = datetime.fromisoformat(str(trained_at))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (now - dt) >= window
        except Exception:
            return True

    def train_ols(self, symbol, features_df=None):
        """Simple OLS Linear Regression implementation using NumPy as a fallback"""
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return None

        # Skip retraining if the model is fresh enough.
        if not self._should_retrain(symbol):
            entry = self.registry.get(symbol, {})
            model_id = entry.get("latest_model")
            if model_id:
                logger.info("Skipping retrain for %s (retrain_frequency satisfied).", symbol)
                return model_id

        df = None
        feature_path = os.path.join(self.feature_store_dir, f"{symbol}_features.csv")
        if features_df is not None:
            df = features_df
        elif self.store_feature_files and os.path.exists(feature_path):
            df = pd.read_csv(feature_path)
        else:
            # Generate features from SQLite on-the-fly to minimize disk usage.
            try:
                fe = self._get_feature_engineer()
                df = fe.generate(symbol)
            except Exception as exc:
                logger.error("Failed to generate features for %s: %s", symbol, exc)
                return None

        if df is None or df.empty:
            logger.error(f"No features found for {symbol}")
            return None

        df = df.dropna()
        
        if df.empty:
            logger.error(f"DataFrame is empty after dropping NaNs for {symbol}")
            return None
            
        # Define features and target
        # Excluding non-feature columns
        exclude = ['symbol', 'date', 'target_return_1d']
        features = [c for c in df.columns if c not in exclude]
        
        # Walk-forward split (last 20% for testing)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[features].values
        y_train = train_df['target_return_1d'].values
        
        # Add bias term (intercept)
        X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
        
        # Solve OLS: (X^T X)^-1 X^T y
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X_train_bias, y_train, rcond=None)
            
            # Predict on test set
            X_test = test_df[features].values
            y_test = test_df['target_return_1d'].values
            X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]
            y_pred = X_test_bias @ coeffs
            
            # Simple metric: MSE
            mse = np.mean((y_test - y_pred)**2)
            logger.info(f"Model trained for {symbol}. MSE: {mse:.8f}")
            
            model_id = f"ols_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = os.path.join(self.models_dir, f"{model_id}.npy")
            np.save(model_path, coeffs)
            
            # Update registry
            self.registry[symbol] = {
                'latest_model': model_id,
                'type': 'ols_numpy',
                'features': features,
                'mse': float(mse),
                'trained_at': datetime.now().isoformat()
            }
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=4)
                
            return model_id
            
        except Exception as e:
            logger.error(f"Training failed for {symbol}: {e}")
            return None

    def prune_models_keep_latest_only(self):
        """
        Delete model files not referenced as latest in model_registry.json.
        Keeps disk usage and file count down.
        """
        keep = set()
        for sym, entry in (self.registry or {}).items():
            if isinstance(entry, dict) and entry.get("latest_model"):
                keep.add(f"{entry['latest_model']}.npy")

        try:
            removed = 0
            for fname in os.listdir(self.models_dir):
                if not fname.endswith(".npy"):
                    continue
                if fname not in keep:
                    try:
                        os.remove(os.path.join(self.models_dir, fname))
                        removed += 1
                    except Exception:
                        pass
            if removed:
                logger.info("Pruned %d old model files.", removed)
        except Exception as exc:
            logger.warning("Model prune failed: %s", exc)

if __name__ == "__main__":
    manager = ModelManager()
    # Test for AAPL
    model_id = manager.train_ols("AAPL")
    if model_id:
        print(f"Model ID created: {model_id}")
