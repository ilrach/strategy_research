"""
Exploratory Analysis: End-of-Week Reversal Pattern

Question: Is there a reversal pattern in SPY/TLT spread returns from one week to the next?

Hypothesis: If SPY outperforms TLT during Monday-Friday, does TLT outperform
in the following Monday-Friday? (Similar to our EOM findings)

X-axis: Cumulative spread return from Monday through Friday (week N)
Y-axis: Cumulative spread return from Monday through Friday (week N+1)
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
print("EXPLORATORY: END-OF-WEEK REVERSAL PATTERN")
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

# Add day of week
df['day_of_week'] = df.index.dayofweek  # Monday=0, Friday=4
df['is_friday'] = df['day_of_week'] == 4

# Identify weeks (using week ending Friday)
# We'll group by year and week number, but only consider weeks ending on Friday
df['year'] = df.index.year
df['week_num'] = df.index.isocalendar().week

print("Identifying complete weeks (Monday-Friday)...")
print()

# For each Friday, look back to get the week's cumulative return
weekly_data = []

fridays = df[df['is_friday']].index

for i, friday in enumerate(fridays):
    # Get all days in this week (up to and including Friday)
    week_start = friday - pd.Timedelta(days=4)  # Go back to Monday

    # Get the trading days in this week (up to Friday)
    week_mask = (df.index >= week_start) & (df.index <= friday)
    week_days = df[week_mask]

    # Only proceed if we have at least 3 trading days
    if len(week_days) >= 3:
        week_cum_return = week_days['spread_return'].sum()

        # Now get the next week's Friday
        if i + 1 < len(fridays):
            next_friday = fridays[i + 1]
            next_week_start = next_friday - pd.Timedelta(days=4)

            # Get next week's trading days
            next_week_mask = (df.index >= next_week_start) & (df.index <= next_friday)
            next_week_days = df[next_week_mask]

            if len(next_week_days) >= 3:
                next_week_cum_return = next_week_days['spread_return'].sum()

                weekly_data.append({
                    'friday': friday,
                    'next_friday': next_friday,
                    'week_return': week_cum_return,
                    'next_week_return': next_week_cum_return,
                    'week_days': len(week_days),
                    'next_week_days': len(next_week_days)
                })

weekly_df = pd.DataFrame(weekly_data)

print(f"Analyzed {len(weekly_df)} consecutive week pairs (Friday to Friday)")
print(f"Date range: {weekly_df['friday'].min().date()} to {weekly_df['friday'].max().date()}")
print()

# Summary statistics
print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nCurrent Week (Mon-Fri):")
print(f"  Mean cumulative return:   {weekly_df['week_return'].mean()*100:>8.3f}%")
print(f"  Median cumulative return: {weekly_df['week_return'].median()*100:>8.3f}%")
print(f"  Std dev:                  {weekly_df['week_return'].std()*100:>8.3f}%")

print(f"\nNext Week (Mon-Fri):")
print(f"  Mean cumulative return:   {weekly_df['next_week_return'].mean()*100:>8.3f}%")
print(f"  Median cumulative return: {weekly_df['next_week_return'].median()*100:>8.3f}%")
print(f"  Std dev:                  {weekly_df['next_week_return'].std()*100:>8.3f}%")

# Correlation
correlation = weekly_df['week_return'].corr(weekly_df['next_week_return'])
print(f"\nCorrelation: {correlation:>8.3f}")

# Regression
z = np.polyfit(weekly_df['week_return'], weekly_df['next_week_return'], 1)
slope, intercept = z[0], z[1]
print(f"Regression slope: {slope:>8.3f}")
print(f"Regression intercept: {intercept*100:>8.3f}%")

# Save results
weekly_df.to_csv('../data/exploratory_weekly_reversal.csv', index=False)
print(f"\nResults saved to: ../data/exploratory_weekly_reversal.csv")

# Visualization - Scatter plot
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Convert to percentage for plotting
x = weekly_df['week_return'] * 100
y = weekly_df['next_week_return'] * 100

ax.scatter(x, y, alpha=0.5, s=30, color='steelblue', edgecolor='black', linewidth=0.5)

# Add reference lines
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

# Add regression line
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line / 100 * 100 + intercept * 100  # Convert back to percentage
ax.plot(x_line, y_line, "r--", alpha=0.8, linewidth=2,
        label=f'Regression: y={slope:.3f}x + {intercept*100:.3f}%')

# Add mean lines
ax.axhline(y=y.mean(), color='green', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'Mean Next Week: {y.mean():.3f}%')
ax.axvline(x=x.mean(), color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'Mean Current Week: {x.mean():.3f}%')

ax.set_xlabel('Current Week (Mon-Fri) Cumulative Spread Return (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Next Week (Mon-Fri) Cumulative Spread Return (%)', fontsize=12, fontweight='bold')
ax.set_title(f'SPY/TLT Spread: Week-to-Week Pattern (Friday-ending weeks)\\n'
             f'Correlation: {correlation:.3f} | N={len(weekly_df)} week pairs | Slope: {slope:.3f}',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/exploratory_weekly_reversal.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/exploratory_weekly_reversal.png")

# Additional analysis: Quartile breakdown
print()
print("="*80)
print("QUARTILE ANALYSIS")
print("="*80)

# Split current week returns into quartiles
weekly_df['week_quartile'] = pd.qcut(weekly_df['week_return'], q=4, labels=['Q1 (worst)', 'Q2', 'Q3', 'Q4 (best)'])

print("\nNext week returns by current week performance quartile:")
print(f"{'Quartile':<15} {'N':>6} {'Mean Next Week':>15} {'Median Next Week':>17}")
print("-"*80)
for quartile in ['Q1 (worst)', 'Q2', 'Q3', 'Q4 (best)']:
    q_data = weekly_df[weekly_df['week_quartile'] == quartile]['next_week_return']
    print(f"{quartile:<15} {len(q_data):>6} {q_data.mean()*100:>14.3f}% {q_data.median()*100:>16.3f}%")

# Test for reversal: compare Q1 vs Q4 next week returns
q1_next = weekly_df[weekly_df['week_quartile'] == 'Q1 (worst)']['next_week_return']
q4_next = weekly_df[weekly_df['week_quartile'] == 'Q4 (best)']['next_week_return']

print()
print("="*80)
print("REVERSAL TEST")
print("="*80)
print(f"\nAfter worst performing weeks (Q1):")
print(f"  Mean next week return: {q1_next.mean()*100:>8.3f}%")
print(f"\nAfter best performing weeks (Q4):")
print(f"  Mean next week return: {q4_next.mean()*100:>8.3f}%")
print(f"\nDifference (Q1 - Q4): {(q1_next.mean() - q4_next.mean())*100:>8.3f}%")

if q1_next.mean() > q4_next.mean():
    print("\n✓ REVERSAL DETECTED: Worst weeks followed by better performance")
else:
    print("\n✗ NO REVERSAL: Best weeks continue outperforming")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print()
if correlation < -0.1:
    print(f"Strong negative correlation ({correlation:.3f}) suggests week-to-week REVERSAL")
    print("Similar to EOM effect - relative performance tends to reverse week over week")
elif correlation > 0.1:
    print(f"Positive correlation ({correlation:.3f}) suggests week-to-week MOMENTUM")
    print("Unlike EOM effect - relative performance tends to continue week over week")
else:
    print(f"Near-zero correlation ({correlation:.3f}) suggests NO predictable pattern")
    print("Week-to-week spread returns appear independent")
