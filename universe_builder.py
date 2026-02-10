import pandas as pd
import os
import yaml
import logging
import requests
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniverseBuilder:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, 'config.yaml')
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        base_dir = os.path.dirname(os.path.abspath(config_path))
        self.universe_file = self.config['universe']['source']
        if not os.path.isabs(self.universe_file):
            self.universe_file = os.path.join(base_dir, self.universe_file)
            
        self.min_market_cap = self.config['universe'].get('min_market_cap', 0)
        
    def load_universe(self):
        """Manual mode: Load the existing tech.csv"""
        if os.path.exists(self.universe_file):
            logger.info(f"Loading universe from {self.universe_file}")
            return pd.read_csv(self.universe_file)
        else:
            logger.warning(f"Universe file {self.universe_file} not found.")
            return pd.DataFrame(columns=['ticker', 'exchange', 'sector', 'market_cap'])

    def update_metadata(self):
        """Auto-update mode: Refresh metadata using yfinance and write back to tech.csv"""
        try:
            import yfinance as yf
        except Exception as e:
            logger.error(f"Failed to import yfinance: {e}. Auto-update mode might not work on this Python version.")
            return

        df = self.load_universe()
        if df.empty:
            logger.error("No tickers to update.")
            return
        
        updated_data = []
        for ticker in df['ticker']:
            try:
                logger.info(f"Fetching metadata for {ticker}...")
                info = yf.Ticker(ticker).info
                updated_data.append({
                    'ticker': ticker,
                    'exchange': info.get('exchange', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'market_cap': info.get('marketCap', 0)
                })
            except Exception as e:
                logger.error(f"Failed to fetch metadata for {ticker}: {e}")
                # Keep existing data if possible, or mark as N/A
                existing = df[df['ticker'] == ticker].to_dict('records')
                if existing:
                    updated_data.append(existing[0])
                else:
                    updated_data.append({'ticker': ticker, 'exchange': 'N/A', 'sector': 'N/A', 'market_cap': 0})
        
        updated_df = pd.DataFrame(updated_data)
        
        # Filter by market cap if configured
        if self.min_market_cap > 0:
            count_before = len(updated_df)
            updated_df = updated_df[updated_df['market_cap'] >= self.min_market_cap]
            logger.info(f"Filtered out {count_before - len(updated_df)} stocks below ${self.min_market_cap/1e9}B market cap")

        updated_df.to_csv(self.universe_file, index=False)
        logger.info(f"Successfully updated and saved universe to {self.universe_file}")
        return updated_df

    def initialize_from_seed(self, seed_file=None):
        """Initialize tech.csv from a list of tickers"""
        if seed_file is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            seed_file = os.path.join(base_dir, 'seed_tickers.txt')
            
        if os.path.exists(seed_file):
            with open(seed_file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            
            df = pd.DataFrame({'ticker': tickers})
            df['exchange'] = 'N/A'
            df['sector'] = 'N/A'
            df['market_cap'] = 0
            
            os.makedirs(os.path.dirname(self.universe_file), exist_ok=True)
            df.to_csv(self.universe_file, index=False)
            logger.info(f"Initialized universe with {len(tickers)} tickers from {seed_file}")
        else:
            logger.error(f"Seed file {seed_file} not found.")

    def initialize_sp500(self):
        """
        Download the current S&P 500 constituents list (503 tickers) and write it to universe file.

        Source: datasets/s-and-p-500-companies (GitHub raw CSV).
        Columns written: ticker, exchange, sector, market_cap
        """
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "trading_bot"})
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
        except Exception as exc:
            logger.error("Failed to download S&P 500 constituents: %s", exc)
            return None

        if df.empty or "Symbol" not in df.columns:
            logger.error("Unexpected S&P 500 CSV format.")
            return None

        out = df.rename(columns={"Symbol": "ticker", "GICS Sector": "sector"})
        out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
        out["exchange"] = "US"
        out["market_cap"] = 0
        out = out[["ticker", "exchange", "sector", "market_cap"]].sort_values("ticker").reset_index(drop=True)

        os.makedirs(os.path.dirname(self.universe_file), exist_ok=True)
        out.to_csv(self.universe_file, index=False)
        logger.info("Saved %d S&P 500 tickers to %s", len(out), self.universe_file)
        return out

if __name__ == "__main__":
    builder = UniverseBuilder()
    # If file doesn't exist, initialize it first
    if not os.path.exists(builder.universe_file):
        builder.initialize_from_seed()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1].strip().lower() in {"sp500", "s&p500"}:
        builder.initialize_sp500()
    print(builder.load_universe().head())
