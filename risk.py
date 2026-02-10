import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config_path=None):
        import os
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, 'config.yaml')
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.risk_cfg = self.config.get('risk', {})
        self.max_pos_pct = self.risk_cfg.get('max_position_equity_pct', 0.10)
        self.max_exposure = self.risk_cfg.get('max_gross_exposure', 1.0)
        self.max_daily_loss = self.risk_cfg.get('max_daily_loss_pct', 0.02)

    def validate_portfolio(self, target_portfolio, current_equity):
        """Apply hard rails to the target portfolio"""
        if target_portfolio.empty:
            return target_portfolio
            
        validated = target_portfolio.copy()
        
        # 1. Cap individual position sizes
        validated['weight'] = validated['weight'].clip(upper=self.max_pos_pct)
        
        # 2. Cap total exposure
        total_weight = validated['weight'].sum()
        if total_weight > self.max_exposure:
            logger.warning(f"Total exposure {total_weight:.2f} exceeds cap {self.max_exposure}. Scaling down...")
            validated['weight'] = validated['weight'] * (self.max_exposure / total_weight)
            
        return validated

    def calculate_dynamic_stop(self, symbol, current_price, volatility):
        """Bot-determined stop loss based on volatility (e.g., 2*ATR)"""
        # For our swing bot, let's use a 5% fixed OR 2*volatility proxy
        # If volatility (std) is provided, we can use it.
        stop_dist = max(0.02, 2.0 * volatility) 
        return current_price * (1.0 - stop_dist)

    def check_kill_switch(self, daily_pnl_pct):
        """Hard rail: stop all trading if daily loss limit hit"""
        if daily_pnl_pct <= -self.max_daily_loss:
            logger.critical(f"KILL-SWITCH TRIGGERED: Daily loss {daily_pnl_pct:.2%} exceeds limit {self.max_daily_loss:.2%}")
            return True
        return False

if __name__ == "__main__":
    rm = RiskManager()
    # Dummy data
    import pandas as pd
    test_portfolio = pd.DataFrame({'symbol': ['AAPL', 'MSFT'], 'weight': [0.15, 0.15]})
    print("Pre-validation:")
    print(test_portfolio)
    valid_p = rm.validate_portfolio(test_portfolio, 100000)
    print("Post-validation (capped at 10%):")
    print(valid_p)
