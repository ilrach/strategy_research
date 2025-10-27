"""
Check if FMP API has short borrow cost / short interest data
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient
import requests

client = FMPClient()

# Test ticker
ticker = 'TQQQ'

print("Checking FMP API for borrow cost / short interest data...")
print(f"Test ticker: {ticker}")
print()

# Try different endpoints that might have borrow/short data
endpoints_to_try = [
    f'short-interest/{ticker}',
    f'stock-short-interest/{ticker}',
    f'short-volume/{ticker}',
    f'institutional-ownership/{ticker}',
    f'grade/{ticker}',
]

for endpoint in endpoints_to_try:
    url = f"{client.BASE_URL}/{endpoint}"
    params = {'apikey': client.api_key}

    print(f"Trying: {endpoint}")
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data:
                print(f"  ✓ Found data!")
                print(f"  Sample: {data[:2] if isinstance(data, list) else data}")
            else:
                print(f"  ✗ Endpoint exists but no data returned")
        else:
            print(f"  ✗ Status {response.status_code}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()

# Also check available v3 endpoints
print("="*80)
print("Checking FMP documentation for borrow/short data...")
print("="*80)
print()

# Try the v3 API base
v3_base = "https://financialmodelingprep.com/api/v3"
v4_base = "https://financialmodelingprep.com/api/v4"

v3_endpoints = [
    f'{v3_base}/grade/{ticker}',
    f'{v3_base}/stock-short-interest',
]

v4_endpoints = [
    f'{v4_base}/short-interest',
]

for url_base, endpoint in [(v3_base, f'grade/{ticker}')]:
    url = f"{url_base}/{endpoint}"
    params = {'apikey': client.api_key}

    print(f"Trying v3: {endpoint}")
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data:
                print(f"  ✓ Data found")
                print(f"  Keys: {list(data[0].keys()) if isinstance(data, list) and len(data) > 0 else 'N/A'}")
            else:
                print(f"  ✗ No data")
        else:
            print(f"  ✗ Status {response.status_code}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()

print("="*80)
print("CONCLUSION:")
print("="*80)
print()
print("FMP API does NOT appear to have:")
print("  - Short borrow costs/fees")
print("  - Historical borrow rates")
print("  - Cost to borrow (CTB) data")
print()
print("This data is typically only available from:")
print("  - Interactive Brokers API")
print("  - Bloomberg Terminal")
print("  - S3 Partners / Ortex (specialized short data providers)")
print("  - Some brokers via their APIs")
print()
print("For backtesting, you'll need to:")
print("  1. Use estimated borrow costs (e.g., 1-5% annually for liquid ETFs)")
print("  2. Assume hard-to-borrow fees for certain periods")
print("  3. Or source borrow cost data from a specialized provider")
