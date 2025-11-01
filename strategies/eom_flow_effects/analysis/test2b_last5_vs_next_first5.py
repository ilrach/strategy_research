"""
TEST 2b: SPY/TLT Ratio Returns - Last 5 Days vs Next Month's First 5 Days

Analyzes the SPY/TLT ratio returns:
- X-axis: Cumulative return of SPY/TLT ratio during last 5 trading days of month
- Y-axis: Cumulative return of SPY/TLT ratio during first 5 trading days of NEXT month

This tests whether the relative performance in the last 5 days reverses in the
first 5 days of the following month.
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
print("TEST 2b: SPY/TLT RATIO - LAST 5 DAYS VS NEXT MONTH'S FIRST 5 DAYS")
print("="*80)
print()

# Load data
client = FMPClient()
print("Loading SPY and TLT data...")

spy_data = client.get_historical_prices('SPY', '2002-07-26', '2025-10-29')
spy_df = pd.DataFrame(spy_data)
spy_df['date'] = pd.to_datetime(spy_df['date'])
spy_df = spy_df.set_index('date').sort_index()
spy_df['spy_return'] = spy_df['adjClose'].pct_change()

tlt_data = client.get_historical_prices('TLT', '2002-07-26', '2025-10-29')
tlt_df = pd.DataFrame(tlt_data)
tlt_df['date'] = pd.to_datetime(tlt_df['date'])
tlt_df = tlt_df.set_index('date').sort_index()
tlt_df['tlt_return'] = tlt_df['adjClose'].pct_change()

# Merge on common dates
df = pd.DataFrame({
    'spy_return': spy_df['spy_return'],
    'tlt_return': tlt_df['tlt_return']
}).dropna()

print(f"Loaded {len(df)} trading days from {df.index[0].date()} to {df.index[-1].date()}")
print()

# Calculate ratio return (SPY/TLT)
df['ratio_return'] = np.log(1 + df['spy_return']) - np.log(1 + df['tlt_return'])

# Add year and month columns
df['year'] = df.index.year
df['month'] = df.index.month
df['year_month'] = df.index.to_period('M')

# Calculate trading day of month
df['trading_day'] = df.groupby('year_month').cumcount() + 1

# Calculate last trading day of each month
df['days_in_month'] = df.groupby('year_month')['trading_day'].transform('max')

# Determine if we're in the last 5 or first 5 trading days
df['last_5_threshold'] = (df['days_in_month'] - 4).clip(lower=19)
df['is_last_5'] = df['trading_day'] >= df['last_5_threshold']
df['is_first_5'] = df['trading_day'] <= 5

# For each month, calculate cumulative returns for last 5 days
# Then match with next month's first 5 days
monthly_stats = []

periods = sorted(df['year_month'].unique())

for i in range(len(periods) - 1):
    current_period = periods[i]
    next_period = periods[i + 1]

    # Current month's last 5 days
    current_month = df[df['year_month'] == current_period]
    last_5 = current_month[current_month['is_last_5']]

    # Next month's first 5 days
    next_month = df[df['year_month'] == next_period]
    first_5 = next_month[next_month['is_first_5']]

    if len(last_5) > 0 and len(first_5) > 0:
        last_5_cum_ret = last_5['ratio_return'].sum()
        first_5_cum_ret = first_5['ratio_return'].sum()

        monthly_stats.append({
            'current_month': current_period,
            'next_month': next_period,
            'last_5_return': last_5_cum_ret,
            'next_first_5_return': first_5_cum_ret,
            'last_5_count': len(last_5),
            'first_5_count': len(first_5)
        })

monthly_df = pd.DataFrame(monthly_stats)

print(f"Analyzed {len(monthly_df)} consecutive month pairs")
print()

# Summary statistics
print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nLast 5 Days of Month:")
print(f"  Mean cumulative return:   {monthly_df['last_5_return'].mean()*100:>8.3f}%")
print(f"  Median cumulative return: {monthly_df['last_5_return'].median()*100:>8.3f}%")
print(f"  Std dev:                  {monthly_df['last_5_return'].std()*100:>8.3f}%")

print(f"\nFirst 5 Days of Next Month:")
print(f"  Mean cumulative return:   {monthly_df['next_first_5_return'].mean()*100:>8.3f}%")
print(f"  Median cumulative return: {monthly_df['next_first_5_return'].median()*100:>8.3f}%")
print(f"  Std dev:                  {monthly_df['next_first_5_return'].std()*100:>8.3f}%")

# Correlation
correlation = monthly_df['last_5_return'].corr(monthly_df['next_first_5_return'])
print(f"\nCorrelation: {correlation:>8.3f}")

# Save results
monthly_df.to_csv('../data/test2b_last5_vs_next_first5.csv', index=False)
print(f"\nResults saved to: ../data/test2b_last5_vs_next_first5.csv")

# Visualization - Scatter plot
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Convert to percentage for plotting
x = monthly_df['last_5_return'] * 100
y = monthly_df['next_first_5_return'] * 100

ax.scatter(x, y, alpha=0.5, s=30, color='steelblue', edgecolor='black', linewidth=0.5)

# Add reference lines
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

# Add regression line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
        label=f'Regression: y={z[0]:.3f}x + {z[1]:.3f}')

# Add mean lines
ax.axhline(y=y.mean(), color='green', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'Mean Next First 5: {y.mean():.3f}%')
ax.axvline(x=x.mean(), color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'Mean Last 5: {x.mean():.3f}%')

ax.set_xlabel('Last 5 Days Cumulative Return (%) [Current Month]', fontsize=12, fontweight='bold')
ax.set_ylabel('First 5 Days Cumulative Return (%) [Next Month]', fontsize=12, fontweight='bold')
ax.set_title(f'SPY/TLT Ratio Returns: Last 5 Days vs Next Month First 5 Days\n'
             f'Correlation: {correlation:.3f} | N={len(monthly_df)} month pairs',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/test2b_last5_vs_next_first5.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/test2b_last5_vs_next_first5.png")
