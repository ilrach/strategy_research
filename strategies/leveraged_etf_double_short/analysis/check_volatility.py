"""
Check what volatilities we're actually calculating vs the table
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

client = FMPClient()

# Check TQQQ as example
print("Checking TQQQ volatility calculation...")
tqqq = pd.DataFrame(client.get_historical_prices('TQQQ', '2024-01-01', '2025-10-29'))
tqqq['date'] = pd.to_datetime(tqqq['date'])
tqqq = tqqq.set_index('date').sort_index()

tqqq['ret'] = tqqq['adjClose'].pct_change()
tqqq['vol_30d'] = tqqq['ret'].rolling(30).std() * np.sqrt(252)

# Show recent values
print("\nRecent TQQQ 30-day volatility values:")
print(tqqq[['adjClose', 'ret', 'vol_30d']].tail(40))

print(f"\nMean 30-day vol in 2024-2025: {tqqq['vol_30d'].mean():.2%}")
print(f"Median: {tqqq['vol_30d'].median():.2%}")
print(f"Min: {tqqq['vol_30d'].min():.2%}")
print(f"Max: {tqqq['vol_30d'].max():.2%}")

print("\n" + "="*80)
print("Expected from table: 55.9%")
print("="*80)
