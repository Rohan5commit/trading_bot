#!/usr/bin/env python3
"""Test FMP API for earnings data."""

import os
import requests
from datetime import datetime
from pathlib import Path

# Load .env file
env_path = Path("backtesting/.env")
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

api_key = os.getenv("FMP_API_KEY")
print(f"FMP API Key: {api_key[:5]}...{api_key[-5:] if api_key else 'Not found'}")

# Test earnings calendar API
symbols = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]

for symbol in symbols:
    print(f"\n{symbol}:")
    
    # Try earnings surprises endpoint
    url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{symbol}?apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                print(f"  Found {len(data)} earnings records")
                for item in data[:3]:
                    print(f"    Date: {item.get('date')}, Surprise: {item.get('actualEarningResult')} vs {item.get('estimatedEarning')}")
            else:
                print(f"  No data or unexpected format: {type(data)}")
        else:
            print(f"  Error: {response.text[:200]}")
    except Exception as e:
        print(f"  Exception: {e}")
    
    # Try earnings calendar endpoint
    url2 = f"https://financialmodelingprep.com/api/v3/earnings_calendar/{symbol}?apikey={api_key}"
    try:
        response = requests.get(url2, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"  Calendar endpoint: {len(data) if isinstance(data, list) else 'N/A'} records")
    except Exception as e:
        pass

# Try general earnings calendar
print("\n\nGeneral Earnings Calendar (2022):")
url = f"https://financialmodelingprep.com/api/v3/earnings_calendar?from=2022-01-01&to=2022-06-30&apikey={api_key}"
try:
    response = requests.get(url, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list):
            print(f"Total earnings events: {len(data)}")
            # Count by symbol
            symbol_counts = {}
            for item in data[:100]:  # First 100
                sym = item.get('symbol')
                if sym in symbols:
                    symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
            print(f"Events for target symbols: {symbol_counts}")
        else:
            print(f"Unexpected format: {type(data)}")
except Exception as e:
    print(f"Exception: {e}")
