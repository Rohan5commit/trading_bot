import os
import sqlite3
import pandas as pd
import requests
import yaml
import logging
import time
from io import StringIO

# Load .env from repo root reliably (scheduler may run with a different cwd).
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from api_keys import ApiKeyRotator

class PriceIngestor:
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
            
        self.universe_file = self.config['universe']['source']
        if not os.path.isabs(self.universe_file):
            self.universe_file = os.path.join(base_dir, self.universe_file)
            
        self.lookback_days = self.config['data'].get('lookback_days', 60)

        self.price_source = (self.config.get("data", {}).get("sources", {}) or {}).get("prices", "stooq")
        providers = (self.config.get("data", {}).get("providers", {}) or {})
        td = providers.get("twelvedata", {}) if isinstance(providers, dict) else {}
        self.twelvedata_base_url = str(td.get("base_url", "https://api.twelvedata.com")).rstrip("/")
        self.twelvedata_outputsize = int(td.get("outputsize", 450) or 450)
        self.twelvedata_timeout = int(td.get("timeout_seconds", 20) or 20)
        self.twelvedata_keys = ApiKeyRotator(str(td.get("api_key_env", "TWELVEDATA_API_KEYS")), fallback_env_var="TWELVEDATA_API_KEY")

        av = providers.get("alphavantage", {}) if isinstance(providers, dict) else {}
        self.alphavantage_base_url = str(av.get("base_url", "https://www.alphavantage.co")).rstrip("/")
        self.alphavantage_timeout = int(av.get("timeout_seconds", 20) or 20)
        self.alphavantage_outputsize = str(av.get("outputsize", "full") or "full").strip().lower()
        self.alphavantage_keys = ApiKeyRotator(str(av.get("api_key_env", "ALPHAVANTAGE_API_KEYS")), fallback_env_var="ALPHAVANTAGE_API_KEY")
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, date)
            )
        ''')
        conn.commit()
        conn.close()

    def get_latest_date_for_symbol(self, symbol: str):
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return None
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT MAX(date) FROM prices WHERE symbol=?", (symbol,)).fetchone()
            return row[0] if row and row[0] else None
        finally:
            conn.close()

    def get_latest_market_date(self):
        """Global latest date across all symbols in prices."""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
            return row[0] if row and row[0] else None
        finally:
            conn.close()

    def fetch_stooq_data(self, symbol):
        """Fetch historical daily OHLCV from Stooq"""
        # Stooq ticker for US is SYMBOL.US (lowercase is safest).
        # Some symbols with share classes use different separators on Stooq (e.g. BRK-B vs BRK.B),
        # so we try a few variants.
        symbol_clean = str(symbol or "").strip().lower()
        if not symbol_clean:
            return None

        candidates = [symbol_clean]
        if "." in symbol_clean:
            candidates.append(symbol_clean.replace(".", "-"))
            candidates.append(symbol_clean.replace(".", "_"))
        if "-" in symbol_clean:
            candidates.append(symbol_clean.replace("-", "."))
            candidates.append(symbol_clean.replace("-", "_"))

        # De-dupe while preserving order.
        seen = set()
        tickers = []
        for c in candidates:
            if c and c not in seen:
                tickers.append(c)
                seen.add(c)
        
        try:
            for tkr in tickers:
                stooq_symbol = f"{tkr}.us"
                url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&f=sd2t2ohlcv&h&e=csv"
                logger.info(f"Downloading {symbol} from Stooq (s={stooq_symbol})...")
                response = requests.get(url, timeout=15)
                response.raise_for_status()

                if "No data" in response.text or len(response.text) < 50:
                    continue

                df = pd.read_csv(StringIO(response.text))
                df.columns = [c.lower() for c in df.columns]
                # Preserve canonical symbol as provided by universe file (usually uppercase).
                df['symbol'] = str(symbol or "").strip().upper()
                return df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]

            logger.warning(f"No data found for {symbol} (tried: {', '.join([f'{t}.us' for t in tickers])})")
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} from Stooq: {e}")
            return None

    def fetch_stooq_latest(self, symbol):
        """
        Fetch the latest daily OHLCV row from Stooq (quote endpoint).
        Much faster than downloading full history.
        """
        symbol_clean = str(symbol or "").strip().lower()
        if not symbol_clean:
            return None

        candidates = [symbol_clean]
        if "." in symbol_clean:
            candidates.append(symbol_clean.replace(".", "-"))
            candidates.append(symbol_clean.replace(".", "_"))
        if "-" in symbol_clean:
            candidates.append(symbol_clean.replace("-", "."))
            candidates.append(symbol_clean.replace("-", "_"))

        seen = set()
        tickers = []
        for c in candidates:
            if c and c not in seen:
                tickers.append(c)
                seen.add(c)

        for tkr in tickers:
            stooq_symbol = f"{tkr}.us"
            url = f"https://stooq.com/q/l/?s={stooq_symbol}&f=sd2t2ohlcv&h&e=csv"
            try:
                logger.info(f"Downloading latest {symbol} from Stooq (s={stooq_symbol})...")
                response = requests.get(url, timeout=15, headers={"User-Agent": "trading_bot"})
                response.raise_for_status()
                text = response.text.strip()
                if "No data" in text or len(text.splitlines()) < 2:
                    continue
                df = pd.read_csv(StringIO(text))
                df.columns = [c.lower() for c in df.columns]
                # Expected columns: symbol,date,time,open,high,low,close,volume
                if "date" not in df.columns:
                    continue
                out = pd.DataFrame({
                    "symbol": [str(symbol or "").strip().upper()],
                    "date": [str(df.iloc[0]["date"])],
                    "open": [float(df.iloc[0].get("open"))],
                    "high": [float(df.iloc[0].get("high"))],
                    "low": [float(df.iloc[0].get("low"))],
                    "close": [float(df.iloc[0].get("close"))],
                    "volume": [int(float(df.iloc[0].get("volume") or 0))],
                })
                return out
            except Exception:
                continue
        logger.warning(f"No latest quote found for {symbol} (tried: {', '.join([f'{t}.us' for t in tickers])})")
        return None

    def fetch_twelvedata_daily(self, symbol, outputsize=None):
        """
        Fetch daily OHLCV from TwelveData (limited window).
        Uses rotating keys from TWELVEDATA_API_KEYS to spread free-tier rate limits.
        """
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return None

        outsz = int(outputsize or self.twelvedata_outputsize or 450)
        keys = self.twelvedata_keys.keys()
        if not keys:
            return None

        for _ in range(len(keys)):
            key = self.twelvedata_keys.next_key()
            if not key:
                break
            params = {
                "symbol": symbol,
                "interval": "1day",
                "format": "JSON",
                "outputsize": outsz,
                "apikey": key,
            }
            try:
                r = requests.get(
                    f"{self.twelvedata_base_url}/time_series",
                    params=params,
                    timeout=self.twelvedata_timeout,
                    headers={"User-Agent": "trading_bot"},
                )
                if r.status_code == 429:
                    continue
                r.raise_for_status()
                payload = r.json()
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue
            if payload.get("status") == "error":
                # Common rate-limit message also comes through here.
                logger.warning(f"TwelveData error for {symbol}: {payload.get('message', 'Unknown error')}")
                continue
            values = payload.get("values") or []
            if not values:
                continue
            df = pd.DataFrame(values)
            if "datetime" not in df.columns:
                continue
            df = df.rename(columns={"datetime": "date"})
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = 0
            df["symbol"] = symbol
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df[["symbol", "date", "open", "high", "low", "close", "volume"]].sort_values("date")

        return None

    def fetch_alphavantage_daily(self, symbol, outputsize=None):
        """
        Alpha Vantage Daily Adjusted (free-tier friendly, but low RPM).
        Requires ALPHAVANTAGE_API_KEYS (comma-separated) or ALPHAVANTAGE_API_KEY.
        """
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return None
        keys = self.alphavantage_keys.keys()
        if not keys:
            return None
        outsz = (outputsize or self.alphavantage_outputsize or "full")
        outsz = str(outsz).strip().lower()
        if outsz not in {"compact", "full"}:
            outsz = "full"

        for _ in range(len(keys)):
            key = self.alphavantage_keys.next_key()
            if not key:
                break
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": outsz,
                "apikey": key,
            }
            try:
                r = requests.get(
                    f"{self.alphavantage_base_url}/query",
                    params=params,
                    timeout=self.alphavantage_timeout,
                    headers={"User-Agent": "trading_bot"},
                )
                if r.status_code == 429:
                    continue
                r.raise_for_status()
                payload = r.json()
            except Exception:
                continue

            series = payload.get("Time Series (Daily)")
            if not isinstance(series, dict) or not series:
                # Could be throttled or invalid symbol.
                continue

            rows = []
            for date_str, vals in series.items():
                try:
                    rows.append({
                        "symbol": symbol,
                        "date": date_str,
                        "open": float(vals.get("1. open")),
                        "high": float(vals.get("2. high")),
                        "low": float(vals.get("3. low")),
                        "close": float(vals.get("4. close")),
                        "volume": int(float(vals.get("6. volume") or 0)),
                    })
                except Exception:
                    continue
            if not rows:
                continue
            df = pd.DataFrame(rows).sort_values("date")
            return df[["symbol", "date", "open", "high", "low", "close", "volume"]]
        return None

    def ingest_universe(self):
        universe_df = pd.read_csv(self.universe_file)
        # For efficiency, only ingest what we don't have or update last N days
        # Simple implementation: ingest all for now
        
        conn = sqlite3.connect(self.db_path)
        
        for ticker in universe_df['ticker']:
            df = self.fetch_stooq_data(ticker)
            if df is not None and not df.empty:
                df.to_sql('prices', conn, if_exists='append', index=False, method=self._sqlite_upsert)
                logger.info(f"Ingested {len(df)} rows for {ticker}")
            
            # Rate limiting
            time.sleep(1.0) 
            
        conn.close()

    def _sqlite_upsert(self, table, conn, keys, data_iter):
        """Helper for upserting into SQLite"""
        from sqlite3 import IntegrityError
        
        data = [dict(zip(keys, row)) for row in data_iter]
        cursor = conn.cursor()
        
        # Proper UPSERT syntax for SQLite 3.24+
        query = f"""
            INSERT INTO {table.name} ({', '.join(keys)})
            VALUES ({', '.join(['?' for _ in keys])})
            ON CONFLICT(symbol, date) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume
        """
        cursor.executemany(query, [tuple(d.values()) for d in data])
        conn.commit()

if __name__ == "__main__":
    ingestor = PriceIngestor()
    # To avoid huge initial download for testing, just do first 5
    universe_df = pd.read_csv(ingestor.universe_file).head(5)
    
    conn = sqlite3.connect(ingestor.db_path)
    for ticker in universe_df['ticker']:
        df = ingestor.fetch_stooq_data(ticker)
        if df is not None and not df.empty:
            ingestor._sqlite_upsert(type('Table', (), {'name': 'prices'}), conn, df.columns.tolist(), df.values.tolist())
            logger.info(f"Ingested {len(df)} rows for {ticker}")
        time.sleep(0.5)
    conn.close()
