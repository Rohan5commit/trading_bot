import os
import pandas as pd
import numpy as np
import json
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrategyEngine:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, 'config.yaml')
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.registry_path = './model_registry.json'
        self.models_dir = './models'
        self.feature_store_dir = './feature_store'
        
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}
        
        # Initialize Meta-Learner for adaptive adjustments
        from meta_learner import MetaLearner
        self.meta_learner = MetaLearner(config_path)

    def get_prediction(self, symbol):
        """Make a prediction for a single symbol using its latest features"""
        if symbol not in self.registry:
            logger.warning(f"No model found in registry for {symbol}")
            return None
        
        reg_entry = self.registry[symbol]
        model_id = reg_entry['latest_model']
        model_path = os.path.join(self.models_dir, f"{model_id}.npy")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found")
            return None
            
        coeffs = np.load(model_path)
        features_list = reg_entry['features']
        
        # Load latest features
        feat_path = os.path.join(self.feature_store_dir, f"{symbol}_features.csv")
        if not os.path.exists(feat_path):
            return None
            
        df = pd.read_csv(feat_path)
        if df.empty:
            return None
            
        # Get the very last row (most recent data)
        last_row = df.iloc[-1]
        
        X = last_row[features_list].values
        # Add bias
        X_bias = np.insert(X, 0, 1.0)
        
        # OLS prediction
        pred_return = X_bias @ coeffs
        return pred_return

    def generate_rankings(self):
        """Rank all stocks in the universe by predicted return, then apply meta-learner adjustments"""
        universe_file = self.config['universe']['source']
        universe_df = pd.read_csv(universe_file)
        
        rankings = []
        for ticker in universe_df['ticker']:
            pred = self.get_prediction(ticker)
            if pred is not None:
                rankings.append({'symbol': ticker, 'predicted_return': pred})
        
        rank_df = pd.DataFrame(rankings)
        if not rank_df.empty:
            # Apply meta-learner adjustments (penalties for symbols with poor history)
            rank_df = self.meta_learner.get_confidence_adjustments(rank_df)
            rank_df = rank_df.sort_values('adjusted_score', ascending=False)
            
        return rank_df

if __name__ == "__main__":
    engine = StrategyEngine()
    ranks = engine.generate_rankings()
    print("Top Rankings:")
    print(ranks.head(10))
