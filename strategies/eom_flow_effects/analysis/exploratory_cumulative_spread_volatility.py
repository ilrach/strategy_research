"""
Exploratory Analysis: Cumulative Spread Volatility by Trading Day of Month

Question: Does the standard deviation of cumulative SPY-TLT spread returns
(from start of month to each day) increase as the month progresses?

Hypothesis: Cumulative volatility should increase as the month goes on,
showing increasing dispersion in relative performance outcomes.
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
print("EXPLORATORY: CUMULATIVE SPREAD VOLATILITY BY TRADING DAY")
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

# Calculate cumulative spread return within each month
print("Calculating cumulative spread returns by month...")
df['cum_spread_return'] = df.groupby('year_month')['spread_return'].cumsum()

print()

# For each trading day, calculate std dev of cumulative returns up to that day
print("Calculating volatility of cumulative returns by trading day...")
print()

cumulative_vol_by_day = []

for day in range(1, 24):  # Up to day 23
    # Get all observations at this trading day
    day_data = df[df['trading_day'] == day]['cum_spread_return']

    if len(day_data) >= 50:  # Need sufficient observations
        cumulative_vol_by_day.append({
            'trading_day': day,
            'std': day_data.std(),
            'mean': day_data.mean(),
            'count': len(day_data)
        })

vol_df = pd.DataFrame(cumulative_vol_by_day)

print("="*80)
print("CUMULATIVE SPREAD VOLATILITY BY TRADING DAY")
print("="*80)
print(f"\n{'Day':>4} {'Cum Std Dev':>12} {'Cum Mean':>10} {'N Obs':>8}")
print("-"*80)
for _, row in vol_df.iterrows():
    print(f"{row['trading_day']:>4.0f} {row['std']*100:>11.3f}% {row['mean']*100:>9.3f}% {row['count']:>8.0f}")

# Calculate statistics for different periods
first_5_vol = vol_df[vol_df['trading_day'] <= 5]['std'].iloc[-1]  # Day 5 cumulative vol
day_18_vol = vol_df[vol_df['trading_day'] == 18]['std'].iloc[0]  # Day 18 cumulative vol
last_5_vol = vol_df[vol_df['trading_day'] >= 19]['std'].iloc[-1]  # Last available day cumulative vol

print()
print("="*80)
print("CUMULATIVE VOLATILITY AT KEY POINTS")
print("="*80)
print(f"After day 5:     {first_5_vol*100:.3f}%")
print(f"After day 18:    {day_18_vol*100:.3f}%")
print(f"After day 19+:   {last_5_vol*100:.3f}%")
print()
print(f"Day 18 / Day 5 ratio:   {day_18_vol/first_5_vol:.2f}x")
print(f"Day 19+ / Day 18 ratio: {last_5_vol/day_18_vol:.2f}x")

# Calculate the rate of volatility growth
vol_df['vol_squared'] = vol_df['std'] ** 2  # Variance
vol_df['volatility_growth_rate'] = vol_df['std'].pct_change()

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# Bar chart of cumulative standard deviation
ax1 = axes[0]
bars = ax1.bar(vol_df['trading_day'], vol_df['std'] * 100,
               color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

# Color code the bars
for i, bar in enumerate(bars):
    day = vol_df.iloc[i]['trading_day']
    if day <= 5:
        bar.set_color('green')
        bar.set_alpha(0.6)
    elif day > 18:
        bar.set_color('red')
        bar.set_alpha(0.6)

ax1.set_xlabel('Trading Day of Month', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cumulative Standard Deviation (%)', fontsize=12, fontweight='bold')
ax1.set_title('Cumulative SPY-TLT Spread Volatility: Dispersion from Month Start\n'
              'Each bar shows std dev of cumulative return from day 1 to that day',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xlim(0, vol_df['trading_day'].max() + 1)

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

# Line plot showing the trend
ax2 = axes[1]
ax2.plot(vol_df['trading_day'], vol_df['std'] * 100,
         marker='o', linewidth=2.5, markersize=8, color='darkblue', alpha=0.7)

# Fill between to show periods
ax2.fill_between(vol_df[vol_df['trading_day'] <= 5]['trading_day'],
                  0,
                  vol_df[vol_df['trading_day'] <= 5]['std'] * 100,
                  color='green', alpha=0.2, label='First 5 days')
ax2.fill_between(vol_df[vol_df['trading_day'] > 18]['trading_day'],
                  0,
                  vol_df[vol_df['trading_day'] > 18]['std'] * 100,
                  color='red', alpha=0.2, label='Last 5 days')

# Add theoretical square root of time line (if volatility grows as sqrt(t))
# Start from day 1 volatility
day1_vol = vol_df[vol_df['trading_day'] == 1]['std'].iloc[0]
theoretical_sqrt = day1_vol * np.sqrt(vol_df['trading_day'])
ax2.plot(vol_df['trading_day'], theoretical_sqrt * 100,
         linestyle='--', linewidth=2, color='orange', alpha=0.7,
         label=f'√t scaling (random walk)')

ax2.set_xlabel('Trading Day of Month', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Standard Deviation (%)', fontsize=12, fontweight='bold')
ax2.set_title('Growth in Cumulative Volatility vs √t Expectation\n'
              'Does dispersion grow as expected for random walk?',
              fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, vol_df['trading_day'].max() + 1)

# Variance plot (volatility squared) - should be linear for random walk
ax3 = axes[2]
ax3.plot(vol_df['trading_day'], vol_df['vol_squared'] * 10000,
         marker='s', linewidth=2, markersize=6, color='purple', alpha=0.7,
         label='Actual variance')

# Linear fit for variance
from numpy.polynomial import Polynomial
p = Polynomial.fit(vol_df['trading_day'], vol_df['vol_squared'] * 10000, 1)
ax3.plot(vol_df['trading_day'], p(vol_df['trading_day']),
         linestyle='--', linewidth=2, color='red', alpha=0.7,
         label='Linear fit (expected for random walk)')

ax3.set_xlabel('Trading Day of Month', fontsize=12, fontweight='bold')
ax3.set_ylabel('Variance (bps²)', fontsize=12, fontweight='bold')
ax3.set_title('Variance Growth: Linear = Random Walk Behavior\n'
              'Variance = Std Dev²',
              fontsize=14, fontweight='bold')
ax3.legend(loc='upper left', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, vol_df['trading_day'].max() + 1)

plt.tight_layout()
plt.savefig('../plots/exploratory_cumulative_spread_volatility.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: ../plots/exploratory_cumulative_spread_volatility.png")

# Statistical test: is the growth close to sqrt(t)?
# Calculate correlation between actual volatility and sqrt(t) scaling
theoretical_sqrt_vals = day1_vol * np.sqrt(vol_df['trading_day'])
correlation = np.corrcoef(vol_df['std'], theoretical_sqrt_vals)[0, 1]

print()
print("="*80)
print("RANDOM WALK TEST")
print("="*80)
print(f"Correlation between actual volatility and √t scaling: {correlation:.4f}")
print(f"R-squared: {correlation**2:.4f}")
print()
if correlation > 0.98:
    print("✓ Very close to √t scaling - behaves like a random walk")
elif correlation > 0.95:
    print("≈ Close to √t scaling - mostly random walk behavior")
else:
    print("✗ Does NOT follow √t scaling - non-random walk behavior")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print()
print(f"Cumulative volatility grows from {vol_df['std'].iloc[0]*100:.3f}% on day 1")
print(f"to {vol_df['std'].iloc[-1]*100:.3f}% by day {vol_df['trading_day'].iloc[-1]:.0f}")
print(f"Growth factor: {vol_df['std'].iloc[-1]/vol_df['std'].iloc[0]:.2f}x")
print()
print("This cumulative dispersion means:")
print("- Early in the month, relative performance outcomes are tightly clustered")
print("- By month-end, there's much wider dispersion in SPY vs TLT outcomes")
print("- This growing uncertainty creates larger potential moves to reverse")
