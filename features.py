import os
import sqlite3
import pandas as pd
import numpy as np
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, 'config.yaml')
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.base_dir = os.path.dirname(os.path.abspath(config_path))
        self.db_path = self.config['data']['cache_path']
        if not os.path.isabs(self.db_path):
            self.db_path = os.path.join(self.base_dir, self.db_path)

        self.feature_store_dir = os.path.join(self.base_dir, 'feature_store')
        storage_cfg = (self.config.get("storage") or {}) if isinstance(self.config, dict) else {}
        self.store_feature_files = bool(storage_cfg.get("store_feature_files", True))
        os.makedirs(self.feature_store_dir, exist_ok=True)

    def load_data(self, symbol):
        conn = sqlite3.connect(self.db_path)
        prices = pd.read_sql(f"SELECT * FROM prices WHERE symbol='{symbol}' ORDER BY date", conn)
        news = pd.read_sql(f"SELECT * FROM news WHERE symbol='{symbol}' ORDER BY datetime", conn)
        conn.close()
        
        prices['date'] = pd.to_datetime(prices['date'])
        if not news.empty:
            news['datetime'] = pd.to_datetime(news['datetime'])
            
        return prices, news

    def build_technical_features(self, df):
        df = df.copy()
        # Returns
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        
        # Volatility
        df['volatility_20d'] = df['return_1d'].rolling(20).std()
        
        # Moving Averages
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        df['dist_ma_20'] = (df['close'] - df['ma_20']) / df['ma_20']
        df['dist_ma_50'] = (df['close'] - df['ma_50']) / df['ma_50']
        
        # RSI (Simple Implementation)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Volume features
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        return df

    def build_news_features(self, prices_df, news_df):
        # Aggregate news sentiment into rolling windows (e.g., 7 days)
        if news_df.empty:
            prices_df['news_count_7d'] = 0
            prices_df['news_sentiment_7d'] = 0.0
            return prices_df
        
        news_df['date'] = news_df['datetime'].dt.normalize()
        # Daily aggregate
        daily_news = news_df.groupby('date').agg(
            news_count=('url', 'count'),
            news_sentiment=('sentiment_score', 'mean')
        )
        
        # Rolling sum/mean
        daily_news['news_count_7d'] = daily_news['news_count'].rolling(7).sum()
        daily_news['news_sentiment_7d'] = daily_news['news_sentiment'].rolling(7).mean()
        
        df = pd.merge(prices_df, daily_news[['news_count_7d', 'news_sentiment_7d']], 
                      left_on='date', right_index=True, how='left')
        df['news_count_7d'] = df['news_count_7d'].fillna(0)
        df['news_sentiment_7d'] = df['news_sentiment_7d'].fillna(0)
        
        return df

    def generate(self, symbol):
        """Generate features dataframe for a symbol (does not write to disk)."""
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return None

        prices, news = self.load_data(symbol)
        if prices.empty:
            return None

        df = self.build_technical_features(prices)
        df = self.build_news_features(df, news)

        # Add labels (Next-day return)
        df['target_return_1d'] = df['return_1d'].shift(-1)

        # Remove initial NaNs from rolling windows
        df = df.dropna(subset=['rsi_14', 'volatility_20d'])
        return df

    def generate_and_save(self, symbol):
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return

        if not self.store_feature_files:
            return self.generate(symbol)

        # Skip feature rebuild if we already have features up-to-date with latest prices.
        save_path = os.path.join(self.feature_store_dir, f"{symbol}_features.csv")
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute("SELECT MAX(date) FROM prices WHERE symbol=?", (symbol,)).fetchone()
            conn.close()
            latest_price_date = row[0] if row and row[0] else None
        except Exception:
            latest_price_date = None

        if latest_price_date and os.path.exists(save_path):
            try:
                # Read the last non-empty line and extract the date column (2nd column).
                last_line = ""
                with open(save_path, "rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    end = handle.tell()
                    # Read up to last 8KB to find the last line.
                    handle.seek(max(0, end - 8192), os.SEEK_SET)
                    chunk = handle.read().decode("utf-8", errors="ignore")
                lines = [ln for ln in chunk.splitlines() if ln.strip()]
                if len(lines) >= 2:
                    last_line = lines[-1]
                if last_line:
                    # CSV schema: symbol,date,open,high,low,close,...
                    parts = last_line.split(",", 2)
                    last_date = parts[1] if len(parts) > 1 else ""
                    if last_date and str(last_date).startswith(str(latest_price_date)):
                        logger.info("Features up-to-date for %s (latest=%s). Skipping rebuild.", symbol, latest_price_date)
                        return
            except Exception:
                pass

        df = self.generate(symbol)
        if df is None:
            return None

        if self.store_feature_files:
            df.to_csv(save_path, index=False)
            logger.info(f"Saved {len(df)} features for {symbol} to {save_path}")
        else:
            logger.info("Features generated for %s (store_feature_files=false).", symbol)
        return df

if __name__ == "__main__":
    engineer = FeatureEngineer()
    # Test for AAPL
    df = engineer.generate_and_save("AAPL")
    if df is not None:
        print(df.tail())
