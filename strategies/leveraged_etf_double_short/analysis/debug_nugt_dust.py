"""
Debug NUGT/DUST calculation to understand discrepancy with Portfolio Visualizer
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

client = FMPClient()

# Fetch data
print("Fetching NUGT prices...")
nugt = pd.DataFrame(client.get_historical_prices('NUGT', '2017-05-01', '2017-05-31'))
nugt['date'] = pd.to_datetime(nugt['date'])
print("Fetching DUST prices...")
dust = pd.DataFrame(client.get_historical_prices('DUST', '2017-05-01', '2017-05-31'))
dust['date'] = pd.to_datetime(dust['date'])
print("Fetching BIL prices...")
bil = pd.DataFrame(client.get_historical_prices('BIL', '2017-05-01', '2017-05-31'))
bil['date'] = pd.to_datetime(bil['date'])

# Load borrow costs
nugt_borrow = pd.read_csv('../iborrowdesk_letf_rates/NUGT.csv', parse_dates=['time'])
nugt_borrow = nugt_borrow.rename(columns={'time': 'date', 'fee': 'borrow_nugt'})
nugt_borrow = nugt_borrow[['date', 'borrow_nugt']].set_index('date')

dust_borrow = pd.read_csv('../iborrowdesk_letf_rates/DUST.csv', parse_dates=['time'])
dust_borrow = dust_borrow.rename(columns={'time': 'date', 'fee': 'borrow_dust'})
dust_borrow = dust_borrow[['date', 'borrow_dust']].set_index('date')

# Merge
df = nugt[['date', 'adjClose']].copy()
df = df.rename(columns={'adjClose': 'p_nugt'})
df = df.merge(dust[['date', 'adjClose']].rename(columns={'adjClose': 'p_dust'}), on='date')
df = df.merge(bil[['date', 'adjClose']].rename(columns={'adjClose': 'p_bil'}), on='date')
df = df.set_index('date')

# Merge borrow costs
df = df.join(nugt_borrow)
df = df.join(dust_borrow)
df['borrow_nugt'] = df['borrow_nugt'].ffill()
df['borrow_dust'] = df['borrow_dust'].ffill()

# Calculate returns
df['nugt_ret'] = df['p_nugt'].pct_change()
df['dust_ret'] = df['p_dust'].pct_change()
df['bil_ret'] = df['p_bil'].pct_change()

df = df.dropna()

# Calculate strategy returns
df['cash_return'] = 2.0 * (df['bil_ret'] - 0.01/252)
df['short_return'] = -0.5 * df['nugt_ret'] - 0.5 * df['dust_ret']
df['borrow_cost'] = -0.5 * (df['borrow_nugt']/100/252) - 0.5 * (df['borrow_dust']/100/252)
df['total_return'] = df['cash_return'] + df['short_return'] + df['borrow_cost']

print("\nFirst 20 days of May 2017:")
print(df[['p_nugt', 'p_dust', 'nugt_ret', 'dust_ret', 'bil_ret',
         'cash_return', 'short_return', 'borrow_cost', 'total_return']].head(20).to_string())

print(f"\n\nMay 2017 Summary:")
print(f"Avg NUGT return: {df['nugt_ret'].mean()*100:.4f}%")
print(f"Avg DUST return: {df['dust_ret'].mean()*100:.4f}%")
print(f"Avg BIL return: {df['bil_ret'].mean()*100:.4f}%")
print(f"Avg cash return: {df['cash_return'].mean()*100:.4f}%")
print(f"Avg short return: {df['short_return'].mean()*100:.4f}%")
print(f"Avg borrow cost: {df['borrow_cost'].mean()*100:.4f}%")
print(f"Avg total return: {df['total_return'].mean()*100:.4f}%")
print(f"\nCumulative return for May: {(df['total_return'] + 1).prod() - 1:.4%}")
