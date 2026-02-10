import os
import sqlite3
import pandas as pd
import requests
import yaml
import logging
import time
from datetime import datetime, timedelta
from llm_sentiment import NewsSentimentScorer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsIngestor:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, 'config.yaml')
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        base_dir = os.path.dirname(os.path.abspath(config_path))
        self.db_path = self.config['data']['cache_path']
        if not os.path.isabs(self.db_path):
            self.db_path = os.path.join(base_dir, self.db_path)

        self.sentiment_scorer = NewsSentimentScorer(self.config)
        # If GDELT starts returning 429, avoid hammering it for the rest of the run.
        self._gdelt_rate_limited_until = 0.0
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news (
                symbol TEXT,
                datetime TEXT,
                url TEXT,
                title TEXT,
                source TEXT,
                sentiment_score REAL,
                query TEXT,
                PRIMARY KEY (url, symbol)
            )
        ''')
        conn.commit()
        conn.close()

    def fetch_gdelt_news(self, symbol, company_name=None):
        """Fetch news from GDELT Doc API"""
        now = time.time()
        if now < float(self._gdelt_rate_limited_until or 0.0):
            # Quietly skip to keep the pipeline running fast.
            return []

        query = f'"{symbol}"'
        if company_name:
            query = f'("{symbol}" OR "{company_name}")'
        
        # Add finance context to filter out noise
        full_query = f'{query} (market OR stock OR trading OR finance OR earnings)'
        
        # GDELT API v2 Doc
        # mode=artlist: List of articles
        # format=json
        # maxrecords=50
        # timespan=7d (last 7 days)
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            'query': full_query,
            'mode': 'artlist',
            'format': 'json',
            'maxrecords': 50,
            'timespan': '7d'
        }
        
        try:
            logger.info(f"Fetching GDELT news for {symbol}...")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            extracted_news = []
            for art in articles:
                # GDELT Tone is a proxy for sentiment. 
                # It's not always in 'artlist' mode directly unless requested, 
                # but 'tone' is often part of the enrichment if available.
                # If not, we record 0.0 or use a local sentiment engine later.
                extracted_news.append({
                    'symbol': symbol,
                    'datetime': art.get('seendate'),
                    'url': art.get('url'),
                    'title': art.get('title'),
                    'source': art.get('sourcecountry', 'Unknown'),
                    'sentiment_score': 0.0, # Placeholder, Tone needs 'mode=artlist' with 'extra=tone'
                    'query': full_query
                })
            return extracted_news
            
        except requests.HTTPError as e:
            # Rate limit backoff: stop calling GDELT repeatedly when we're blocked.
            try:
                status = int(getattr(e.response, "status_code", 0) or 0)
            except Exception:
                status = 0
            if status == 429:
                self._gdelt_rate_limited_until = time.time() + 300.0  # 5 minutes
                logger.error("GDELT rate limited (429). Pausing GDELT calls for 5 minutes.")
            logger.error(f"Failed to fetch GDELT news for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch GDELT news for {symbol}: {e}")
            return []

    def fetch_and_store_news(self, symbol, company_name=None):
        news_items = self.fetch_gdelt_news(symbol, company_name)
        if not news_items:
            return []

        news_items = self._filter_existing_items(symbol, news_items)
        if not news_items:
            return []

        news_items = self.sentiment_scorer.score(symbol, news_items)
        self.store_news(news_items)
        return news_items

    def _filter_existing_items(self, symbol, news_items):
        urls = [item.get('url') for item in news_items if item.get('url')]
        if not urls:
            return news_items

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        placeholders = ",".join(["?"] * len(urls))
        query = f"""
            SELECT url FROM news
            WHERE symbol = ? AND url IN ({placeholders})
        """
        cursor.execute(query, [symbol] + urls)
        existing_urls = {row[0] for row in cursor.fetchall()}
        conn.close()

        if not existing_urls:
            return news_items
        return [item for item in news_items if item.get('url') not in existing_urls]

    def get_llm_status(self):
        return self.sentiment_scorer.get_status()

    def store_news(self, news_items):
        if not news_items:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = """
            INSERT INTO news (symbol, datetime, url, title, source, sentiment_score, query)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url, symbol) DO NOTHING
        """
        cursor.executemany(query, [
            (n['symbol'], n['datetime'], n['url'], n['title'], n['source'], n['sentiment_score'], n['query'])
            for n in news_items
        ])
        conn.commit()
        conn.close()
        logger.info(f"Stored {len(news_items)} news items.")

if __name__ == "__main__":
    ingestor = NewsIngestor()
    # Test for AAPL
    news = ingestor.fetch_and_store_news("AAPL", "Apple Inc")
    if news:
        logger.info("Stored %d LLM-scored news items for AAPL.", len(news))
