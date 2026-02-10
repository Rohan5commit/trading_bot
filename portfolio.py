import yaml
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, 'config.yaml')
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.max_positions = self.config['trading'].get('max_positions', 10)
        self.equal_weight = self.config['trading'].get('equal_weight', True)

    def generate_target_portfolio(self, rankings_df, max_positions=None):
        """Select top K long positions from rankings"""
        if rankings_df.empty:
            logger.warning("No rankings provided to generate portfolio")
            return pd.DataFrame()
            
        # Filtering for positive predicted returns? Or just top K?
        # Standard approach for ranking: just top K
        max_positions = max_positions or self.max_positions
        top_k = rankings_df.head(max_positions).copy()
        
        if self.equal_weight:
            top_k['weight'] = 1.0 / len(top_k)
        else:
            # Confidence-weighted sizing (prefer adjusted_score when available)
            score_col = 'adjusted_score' if 'adjusted_score' in top_k.columns else 'predicted_return'
            scores = top_k[score_col].clip(lower=0)
            total = scores.sum()
            if total == 0:
                top_k['weight'] = 1.0 / len(top_k)
            else:
                top_k['weight'] = scores / total
            
        logger.info(f"Target portfolio generated with {len(top_k)} positions")
        return top_k

if __name__ == "__main__":
    from strategy import StrategyEngine
    engine = StrategyEngine()
    ranks = engine.generate_rankings()
    
    pm = PortfolioManager()
    portfolio = pm.generate_target_portfolio(ranks)
    print("Target Portfolio:")
    print(portfolio)
