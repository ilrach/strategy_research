"""
Compare strategy performance with and without borrow costs
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# Load both equity curves
print("Loading equity curves...")

# Without borrow costs (from ultra_simple_backtest.py - need to regenerate for TQQQ/SQQQ only)
# For now, let's compute it inline

from datetime import datetime, timedelta
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

client = FMPClient()

end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)
from_date = start_date.strftime('%Y-%m-%d')
to_date = end_date.strftime('%Y-%m-%d')

print("Fetching price data...")
tqqq_data = client.get_historical_prices('TQQQ', from_date=from_date, to_date=to_date)
sqqq_data = client.get_historical_prices('SQQQ', from_date=from_date, to_date=to_date)
bil_data = client.get_historical_prices('BIL', from_date=from_date, to_date=to_date)

df_tqqq = pd.DataFrame(tqqq_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_bull'})
df_sqqq = pd.DataFrame(sqqq_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_bear'})
df_bil = pd.DataFrame(bil_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_bil'})

for df in [df_tqqq, df_sqqq, df_bil]:
    df['date'] = pd.to_datetime(df['date'])

df = df_tqqq.set_index('date').join(df_sqqq.set_index('date')).join(df_bil.set_index('date'))
df = df.sort_index()

# Calculate returns WITHOUT borrow costs
df['bil_ret'] = df['p_bil'].pct_change()
df['bull_ret'] = df['p_bull'].pct_change()
df['bear_ret'] = df['p_bear'].pct_change()

df['return_no_borrow'] = 2 * df['bil_ret'] - 0.5 * df['bull_ret'] - 0.5 * df['bear_ret']

equity_no_borrow = 100000
equity_curve_no_borrow = [equity_no_borrow]

for i in range(1, len(df)):
    ret = df.iloc[i]['return_no_borrow']
    if pd.notna(ret):
        equity_no_borrow = equity_no_borrow * (1 + ret)
    equity_curve_no_borrow.append(equity_no_borrow)

df['equity_no_borrow'] = equity_curve_no_borrow

# Load WITH borrow costs
df_with_borrow = pd.read_csv('backtest_with_borrow_costs_equity.csv')
df_with_borrow['date'] = pd.to_datetime(df_with_borrow['date'])
df_with_borrow = df_with_borrow.set_index('date')

# Merge
df_compare = df[['equity_no_borrow']].join(df_with_borrow[['equity']], how='inner')
df_compare = df_compare.rename(columns={'equity': 'equity_with_borrow'})

# Calculate cumulative returns in %
df_compare['return_no_borrow_pct'] = (df_compare['equity_no_borrow'] / 100000 - 1) * 100
df_compare['return_with_borrow_pct'] = (df_compare['equity_with_borrow'] / 100000 - 1) * 100

# Calculate metrics for both
print("\n" + "="*80)
print("COMPARISON: WITH vs WITHOUT BORROW COSTS")
print("="*80)
print()

# No borrow
final_no = df_compare['equity_no_borrow'].iloc[-1]
total_ret_no = (final_no / 100000 - 1)
years = len(df_compare) / 365.25
cagr_no = (final_no / 100000) ** (1/years) - 1

returns_no = df_compare['equity_no_borrow'].pct_change().dropna()
vol_no = returns_no.std() * np.sqrt(252)
sharpe_no = (returns_no.mean() * 252) / vol_no if vol_no > 0 else 0

cumulative_no = (1 + returns_no).cumprod()
running_max_no = cumulative_no.expanding().max()
dd_no = (cumulative_no - running_max_no) / running_max_no
mdd_no = dd_no.min()

# With borrow
final_with = df_compare['equity_with_borrow'].iloc[-1]
total_ret_with = (final_with / 100000 - 1)
cagr_with = (final_with / 100000) ** (1/years) - 1

returns_with = df_compare['equity_with_borrow'].pct_change().dropna()
vol_with = returns_with.std() * np.sqrt(252)
sharpe_with = (returns_with.mean() * 252) / vol_with if vol_with > 0 else 0

cumulative_with = (1 + returns_with).cumprod()
running_max_with = cumulative_with.expanding().max()
dd_with = (cumulative_with - running_max_with) / running_max_with
mdd_with = dd_with.min()

# Print comparison
print(f"{'Metric':<20} {'Without Borrow':<20} {'With Borrow':<20} {'Impact':<20}")
print("-"*80)
print(f"{'Total Return':<20} {total_ret_no*100:>18.2f}% {total_ret_with*100:>18.2f}% {(total_ret_with-total_ret_no)*100:>18.2f}%")
print(f"{'CAGR':<20} {cagr_no*100:>18.2f}% {cagr_with*100:>18.2f}% {(cagr_with-cagr_no)*100:>18.2f}%")
print(f"{'Volatility':<20} {vol_no*100:>18.2f}% {vol_with*100:>18.2f}% {(vol_with-vol_no)*100:>18.2f}%")
print(f"{'Sharpe Ratio':<20} {sharpe_no:>18.2f} {sharpe_with:>18.2f} {sharpe_with-sharpe_no:>18.2f}")
print(f"{'Max Drawdown':<20} {mdd_no*100:>18.2f}% {mdd_with*100:>18.2f}% {(mdd_with-mdd_no)*100:>18.2f}%")
print()

print("Borrow Cost Impact:")
print(f"  TQQQ borrow cost: 0.69% annually")
print(f"  SQQQ borrow cost: 1.83% annually")
print(f"  Total cost: 2.52% annually")
print(f"  CAGR reduction: {(cagr_no-cagr_with)*100:.2f}%")
print()

# Create plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Cumulative returns
ax1.plot(df_compare.index, df_compare['return_no_borrow_pct'], label='Without Borrow Costs', linewidth=2, color='#2E86AB')
ax1.plot(df_compare.index, df_compare['return_with_borrow_pct'], label='With Borrow Costs', linewidth=2, color='#C73E1D')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
ax1.set_title('TQQQ/SQQQ Double Short Strategy: Impact of Borrow Costs', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Difference (borrow cost impact)
df_compare['diff'] = df_compare['return_no_borrow_pct'] - df_compare['return_with_borrow_pct']
ax2.fill_between(df_compare.index, 0, df_compare['diff'], alpha=0.5, color='#F18F01', label='Borrow Cost Impact')
ax2.plot(df_compare.index, df_compare['diff'], linewidth=1.5, color='#F18F01')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Return Difference (%)', fontsize=11)
ax2.set_title('Cumulative Impact of Borrow Costs Over Time', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../plots/leveraged_etf/borrow_cost_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plot saved to: ../../plots/leveraged_etf/borrow_cost_comparison.png")
print()
print("Analysis complete!")
