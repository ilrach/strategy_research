"""
Debug why all pairs pass filter 100% of time
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

client = FMPClient()

EDGE_REQUIREMENT = 20.0
leverage = 3
k = np.sqrt(0.01 / (leverage**2 / 2))  # 0.0471 for 3x

print(f"Testing TQQQ/SQQQ with 20% edge requirement")
print(f"Leverage: {leverage}x")
print(f"k = {k:.4f}")
print()

# Load TQQQ
tqqq = pd.DataFrame(client.get_historical_prices('TQQQ', '2024-01-01', '2025-10-29'))
tqqq['date'] = pd.to_datetime(tqqq['date'])
tqqq = tqqq[['date', 'adjClose']].rename(columns={'adjClose': 'p_TQQQ'}).set_index('date').sort_index()

# Load borrow costs
tqqq_borrow = pd.read_csv('../iborrowdesk_letf_rates/TQQQ.csv', parse_dates=['time'])
tqqq_borrow = tqqq_borrow.rename(columns={'time': 'date', 'fee': 'b_TQQQ'})
tqqq_borrow = tqqq_borrow[['date', 'b_TQQQ']].set_index('date')

sqqq_borrow = pd.read_csv('../iborrowdesk_letf_rates/SQQQ.csv', parse_dates=['time'])
sqqq_borrow = sqqq_borrow.rename(columns={'time': 'date', 'fee': 'b_SQQQ'})
sqqq_borrow = sqqq_borrow[['date', 'b_SQQQ']].set_index('date')

# Merge
df = tqqq.join(tqqq_borrow, how='left')
df = df.join(sqqq_borrow, how='left')
df['b_TQQQ'] = df['b_TQQQ'].ffill()
df['b_SQQQ'] = df['b_SQQQ'].ffill()
df = df.dropna()

# Calculate
df['ret'] = df['p_TQQQ'].pct_change()
df['vol_30d'] = df['ret'].rolling(30).std() * np.sqrt(252)
df['avg_borrow'] = (df['b_TQQQ'] + df['b_SQQQ']) / 2

# Filter logic (FIXED)
df['threshold'] = (df['vol_30d'] * 100 / k) - (df['avg_borrow'] + EDGE_REQUIREMENT)
df['passes_filter'] = df['threshold'] > 0

df = df.dropna()

print(f"Total days: {len(df)}")
print(f"Days passing filter: {df['passes_filter'].sum()}")
print(f"Pass rate: {df['passes_filter'].mean()*100:.1f}%")
print()

print("Sample of days (recent):")
print(df[['vol_30d', 'avg_borrow', 'threshold', 'passes_filter']].tail(20))
print()

print("Statistics:")
print(f"Avg vol: {df['vol_30d'].mean():.2%}")
print(f"Avg borrow: {df['avg_borrow'].mean():.2f}%")
print(f"Required vol to pass: {(df['avg_borrow'].mean() + EDGE_REQUIREMENT) * k / 100:.2%}")
print()

# Show breakdown by vol ranges
print("Pass rate by volatility range:")
for vol_min in [0, 30, 40, 50, 60, 70, 80, 90, 100]:
    vol_max = vol_min + 10 if vol_min < 100 else 200
    mask = (df['vol_30d'] >= vol_min/100) & (df['vol_30d'] < vol_max/100)
    if mask.sum() > 0:
        pass_rate = df[mask]['passes_filter'].mean() * 100
        print(f"  {vol_min:3d}%-{vol_max:3d}%: {mask.sum():4d} days, {pass_rate:5.1f}% pass")
