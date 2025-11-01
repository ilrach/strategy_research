"""
Exploratory Analysis: Spread Volatility by Trading Day of Month

Question: Does the standard deviation of SPY-TLT spread returns increase
as the month progresses?

Hypothesis: Volatility should increase toward month-end due to rebalancing flows.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.insert(0, '/Users/bryn/projects/strategy_research')
from fmp_client import FMPClient

sns.set_style("darkgrid")

print("="*80)
print("EXPLORATORY: SPREAD VOLATILITY BY TRADING DAY OF MONTH")
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

# Calculate spread return (SPY - TLT)
df['spread_return'] = df['spy_return'] - df['tlt_return']

# Add month tracking
df['year_month'] = df.index.to_period('M')
df['trading_day'] = df.groupby('year_month').cumcount() + 1

# Calculate statistics by trading day
print("Calculating volatility by trading day of month...")
print()

# Group by trading day and calculate std dev
volatility_by_day = df.groupby('trading_day')['spread_return'].agg([
    ('std', 'std'),
    ('mean', 'mean'),
    ('count', 'count')
]).reset_index()

# Only keep days with sufficient observations (at least 50 months)
volatility_by_day = volatility_by_day[volatility_by_day['count'] >= 50]

print("="*80)
print("SPREAD VOLATILITY BY TRADING DAY")
print("="*80)
print(f"\n{'Day':>4} {'Std Dev':>10} {'Mean':>10} {'N Obs':>8}")
print("-"*80)
for _, row in volatility_by_day.iterrows():
    print(f"{row['trading_day']:>4.0f} {row['std']*100:>9.3f}% {row['mean']*100:>9.3f}% {row['count']:>8.0f}")

# Calculate average volatility for different periods
first_5_vol = volatility_by_day[volatility_by_day['trading_day'] <= 5]['std'].mean()
middle_vol = volatility_by_day[(volatility_by_day['trading_day'] > 5) & (volatility_by_day['trading_day'] <= 18)]['std'].mean()
last_5_vol = volatility_by_day[volatility_by_day['trading_day'] > 18]['std'].mean()

print()
print("="*80)
print("VOLATILITY BY PERIOD")
print("="*80)
print(f"First 5 days:    {first_5_vol*100:.3f}%")
print(f"Middle days 6-18: {middle_vol*100:.3f}%")
print(f"Last 5 days (19+): {last_5_vol*100:.3f}%")
print()
print(f"Last 5 / First 5 ratio: {last_5_vol/first_5_vol:.2f}x")
print(f"Last 5 / Middle ratio:  {last_5_vol/middle_vol:.2f}x")

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Bar chart of standard deviation
ax1 = axes[0]
bars = ax1.bar(volatility_by_day['trading_day'], volatility_by_day['std'] * 100,
               color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

# Color code the bars
for i, bar in enumerate(bars):
    day = volatility_by_day.iloc[i]['trading_day']
    if day <= 5:
        bar.set_color('green')
        bar.set_alpha(0.6)
    elif day > 18:
        bar.set_color('red')
        bar.set_alpha(0.6)

# Add reference lines for averages
ax1.axhline(y=first_5_vol*100, color='green', linestyle='--', linewidth=2,
            alpha=0.7, label=f'First 5 avg: {first_5_vol*100:.3f}%')
ax1.axhline(y=middle_vol*100, color='blue', linestyle='--', linewidth=2,
            alpha=0.7, label=f'Middle avg: {middle_vol*100:.3f}%')
ax1.axhline(y=last_5_vol*100, color='red', linestyle='--', linewidth=2,
            alpha=0.7, label=f'Last 5 avg: {last_5_vol*100:.3f}%')

ax1.set_xlabel('Trading Day of Month', fontsize=12, fontweight='bold')
ax1.set_ylabel('Standard Deviation of Spread Return (%)', fontsize=12, fontweight='bold')
ax1.set_title('SPY-TLT Spread Volatility by Trading Day of Month\n'
              f'Higher volatility = More variable relative performance',
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xlim(0, volatility_by_day['trading_day'].max() + 1)

# Add vertical lines to separate periods
ax1.axvline(x=5.5, color='black', linestyle=':', linewidth=1, alpha=0.5)
ax1.axvline(x=18.5, color='black', linestyle=':', linewidth=1, alpha=0.5)

# Text annotations
ax1.text(3, ax1.get_ylim()[1]*0.95, 'First 5', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
ax1.text(12, ax1.get_ylim()[1]*0.95, 'Middle', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
ax1.text(21, ax1.get_ylim()[1]*0.95, 'Last 5', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

# Line plot showing the trend more clearly
ax2 = axes[1]
ax2.plot(volatility_by_day['trading_day'], volatility_by_day['std'] * 100,
         marker='o', linewidth=2, markersize=6, color='darkblue', alpha=0.7)

# Fill between to show periods
ax2.fill_between(volatility_by_day[volatility_by_day['trading_day'] <= 5]['trading_day'],
                  0,
                  volatility_by_day[volatility_by_day['trading_day'] <= 5]['std'] * 100,
                  color='green', alpha=0.2, label='First 5 days')
ax2.fill_between(volatility_by_day[volatility_by_day['trading_day'] > 18]['trading_day'],
                  0,
                  volatility_by_day[volatility_by_day['trading_day'] > 18]['std'] * 100,
                  color='red', alpha=0.2, label='Last 5 days')

ax2.set_xlabel('Trading Day of Month', fontsize=12, fontweight='bold')
ax2.set_ylabel('Standard Deviation of Spread Return (%)', fontsize=12, fontweight='bold')
ax2.set_title('Trend View: Does Volatility Increase Toward Month-End?',
              fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, volatility_by_day['trading_day'].max() + 1)

plt.tight_layout()
plt.savefig('../plots/exploratory_spread_volatility.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/exploratory_spread_volatility.png")

# Additional analysis: rolling window to smooth
print()
print("="*80)
print("ROLLING 3-DAY AVERAGE VOLATILITY")
print("="*80)
volatility_by_day['rolling_std'] = volatility_by_day['std'].rolling(window=3, center=True).mean()
print(f"\n{'Day':>4} {'Std Dev':>10} {'3-Day Avg':>12}")
print("-"*80)
for _, row in volatility_by_day.iterrows():
    if pd.notna(row['rolling_std']):
        print(f"{row['trading_day']:>4.0f} {row['std']*100:>9.3f}% {row['rolling_std']*100:>11.3f}%")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print()
if last_5_vol > first_5_vol:
    print(f"✓ Volatility INCREASES toward month-end:")
    print(f"  Last 5 days are {last_5_vol/first_5_vol:.2f}x more volatile than first 5 days")
    print(f"  This suggests larger, more variable relative performance swings at EOM")
else:
    print(f"✗ Volatility does NOT increase toward month-end:")
    print(f"  Last 5 days are only {last_5_vol/first_5_vol:.2f}x the volatility of first 5 days")
