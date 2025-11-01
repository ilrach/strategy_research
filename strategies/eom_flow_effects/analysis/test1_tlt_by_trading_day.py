"""
TEST 1: TLT Returns by Trading Day of Month

Analyze TLT (20+ Year Treasury Bond ETF) absolute returns grouped by trading day of month.
This tests for end-of-month flow effects and other intra-month patterns.

Trading day of month:
- Day 1 = first trading day of the month
- Day 2 = second trading day of the month
- Day 20 = 20th trading day (typically near month end)
- etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(0, '/Users/bryn/projects/strategy_research')
from fmp_client import FMPClient

sns.set_style("darkgrid")

print("="*80)
print("TEST 1: TLT RETURNS BY TRADING DAY OF MONTH")
print("="*80)
print()

# Load TLT data
client = FMPClient()
print("Loading TLT data...")
tlt_data = client.get_historical_prices('TLT', '2002-07-26', '2025-10-29')  # TLT inception
tlt_df = pd.DataFrame(tlt_data)
tlt_df['date'] = pd.to_datetime(tlt_df['date'])
tlt_df = tlt_df.set_index('date').sort_index()

# Calculate returns
tlt_df['return'] = tlt_df['adjClose'].pct_change()
tlt_df = tlt_df.dropna()

print(f"Loaded {len(tlt_df)} trading days from {tlt_df.index[0].date()} to {tlt_df.index[-1].date()}")
print()

# Add year and month columns
tlt_df['year'] = tlt_df.index.year
tlt_df['month'] = tlt_df.index.month
tlt_df['year_month'] = tlt_df.index.to_period('M')

# Calculate trading day of month (forward only: 1, 2, 3, ...)
tlt_df['trading_day'] = tlt_df.groupby('year_month').cumcount() + 1

print("Sample of data with trading day labels:")
print(tlt_df[['adjClose', 'return', 'trading_day']].head(25))
print()

# Group by trading day
day_stats = tlt_df.groupby('trading_day')['return'].agg([
    ('count', 'count'),
    ('mean_return', 'mean'),
    ('median_return', 'median'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
])
day_stats['mean_return_bps'] = day_stats['mean_return'] * 10000
day_stats['annualized_return'] = (1 + day_stats['mean_return']) ** 252 - 1

print("="*80)
print("TRADING DAY STATISTICS")
print("="*80)
print(f"\n{'Day':>4} {'Count':>6} {'Mean Ret (bps)':>15} {'Median Ret (bps)':>17} {'Ann. Return':>12}")
print("-"*80)
for day in sorted(day_stats.index):
    row = day_stats.loc[day]
    print(f"{day:>4} {row['count']:>6.0f} {row['mean_return_bps']:>15.2f} "
          f"{row['median_return']*10000:>17.2f} {row['annualized_return']:>11.2%}")

# Overall statistics
print("\n" + "="*80)
print("OVERALL TLT STATISTICS")
print("="*80)
overall_mean = tlt_df['return'].mean()
overall_std = tlt_df['return'].std()
overall_sharpe = (overall_mean / overall_std) * np.sqrt(252) if overall_std > 0 else 0
overall_ann_ret = (1 + overall_mean) ** 252 - 1

print(f"Mean daily return:     {overall_mean*10000:>8.2f} bps")
print(f"Median daily return:   {tlt_df['return'].median()*10000:>8.2f} bps")
print(f"Daily std dev:         {overall_std*10000:>8.2f} bps")
print(f"Annualized return:     {overall_ann_ret:>8.2%}")
print(f"Annualized volatility: {overall_std * np.sqrt(252):>8.2%}")
print(f"Sharpe ratio:          {overall_sharpe:>8.2f}")

# Save results
day_stats.to_csv('../data/test1_trading_day_stats.csv')
print(f"\nResults saved to: ../data/test1_trading_day_stats.csv")

# Visualization - Single plot showing all trading days
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

days_to_plot = day_stats.index[day_stats.index <= 23]
ax.bar(days_to_plot, day_stats.loc[days_to_plot, 'mean_return_bps'],
       color=['green' if x > 0 else 'red' for x in day_stats.loc[days_to_plot, 'mean_return_bps']],
       alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax.axhline(y=overall_mean*10000, color='blue', linestyle='--', linewidth=2, alpha=0.5,
           label=f'Overall Mean ({overall_mean*10000:.2f} bps)')
ax.set_xlabel('Trading Day of Month', fontsize=12)
ax.set_ylabel('Mean Return (bps)', fontsize=12)
ax.set_title(f'TLT Mean Return by Trading Day of Month (2002-2025)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()
ax.set_xticks(range(1, 24))

plt.tight_layout()
plt.savefig('../plots/test1_tlt_by_trading_day.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/test1_tlt_by_trading_day.png")
