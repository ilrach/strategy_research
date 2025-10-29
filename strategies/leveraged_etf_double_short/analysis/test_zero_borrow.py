"""
Test NUGT/DUST with ZERO borrow costs to match Portfolio Visualizer
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

client = FMPClient()

# Use same date range as Portfolio Visualizer: 2010-12-08 to 2025-10-27
start_date = '2010-12-08'
end_date = '2025-10-27'

print("Fetching NUGT prices...")
nugt = pd.DataFrame(client.get_historical_prices('NUGT', start_date, end_date))
nugt['date'] = pd.to_datetime(nugt['date'])

print("Fetching DUST prices...")
dust = pd.DataFrame(client.get_historical_prices('DUST', start_date, end_date))
dust['date'] = pd.to_datetime(dust['date'])

print("Fetching BIL prices...")
bil = pd.DataFrame(client.get_historical_prices('BIL', start_date, end_date))
bil['date'] = pd.to_datetime(bil['date'])

# Merge
df = nugt[['date', 'adjClose']].copy()
df = df.rename(columns={'adjClose': 'p_nugt'})
df = df.merge(dust[['date', 'adjClose']].rename(columns={'adjClose': 'p_dust'}), on='date')
df = df.merge(bil[['date', 'adjClose']].rename(columns={'adjClose': 'p_bil'}), on='date')
df = df.set_index('date')
df = df.sort_index()

print(f"\nData range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Trading days: {len(df)}")

# Calculate returns
df['nugt_ret'] = df['p_nugt'].pct_change()
df['dust_ret'] = df['p_dust'].pct_change()
df['bil_ret'] = df['p_bil'].pct_change()

df = df.dropna()

# Calculate strategy returns with ZERO borrow cost
# Test 1: With 100bps haircut
df['cash_return_haircut'] = 2.0 * (df['bil_ret'] - 0.01/252)  # 200% cash @ BIL - 100bps
# Test 2: Without haircut (like Portfolio Visualizer)
df['cash_return_no_haircut'] = 2.0 * df['bil_ret']  # 200% cash @ BIL

df['short_return'] = -0.5 * df['nugt_ret'] - 0.5 * df['dust_ret']
df['borrow_cost'] = 0.0  # ZERO borrow cost

# First test with haircut
df['total_return'] = df['cash_return_haircut'] + df['short_return'] + df['borrow_cost']

# Build equity curve
df['equity'] = (1 + df['total_return']).cumprod() * 10000

# Calculate metrics
final_equity = df['equity'].iloc[-1]
total_return = (final_equity / 10000) - 1

years = len(df) / 252
cagr = (final_equity / 10000) ** (1/years) - 1

returns = df['total_return']
volatility = returns.std() * np.sqrt(252)
sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0

# Max drawdown
cumulative = df['equity'] / 10000
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_dd = drawdown.min()

# Calmar ratio
calmar = cagr / abs(max_dd) if max_dd != 0 else 0

print("\n" + "="*80)
print("RESULTS: NUGT/DUST with ZERO borrow costs")
print("="*80)
print(f"Total Return:  {total_return:>10.2%}")
print(f"CAGR:          {cagr:>10.2%}")
print(f"Volatility:    {volatility:>10.2%}")
print(f"Sharpe Ratio:  {sharpe:>10.2f}")
print(f"Max Drawdown:  {max_dd:>10.2%}")
print(f"Calmar Ratio:  {calmar:>10.2f}")
print(f"\nYears:         {years:>10.2f}")

print("\n" + "="*80)
print("Portfolio Visualizer Results (for comparison):")
print("="*80)
print(f"CAGR:          {'7.71%':>10}")
print(f"Sharpe:        {'1.15':>10}")
print(f"Max DD:        {'-4.33%':>10}")
print(f"Calmar:        {'1.78':>10}")

# Now test without haircut
df['total_return'] = df['cash_return_no_haircut'] + df['short_return'] + df['borrow_cost']
df['equity'] = (1 + df['total_return']).cumprod() * 10000

final_equity = df['equity'].iloc[-1]
total_return = (final_equity / 10000) - 1
cagr = (final_equity / 10000) ** (1/years) - 1
returns = df['total_return']
volatility = returns.std() * np.sqrt(252)
sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0
cumulative = df['equity'] / 10000
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_dd = drawdown.min()
calmar = cagr / abs(max_dd) if max_dd != 0 else 0

print("\n" + "="*80)
print("RESULTS: NUGT/DUST with NO haircut on cash")
print("="*80)
print(f"Total Return:  {total_return:>10.2%}")
print(f"CAGR:          {cagr:>10.2%}")
print(f"Volatility:    {volatility:>10.2%}")
print(f"Sharpe Ratio:  {sharpe:>10.2f}")
print(f"Max Drawdown:  {max_dd:>10.2%}")
print(f"Calmar Ratio:  {calmar:>10.2f}")
