"""
Exploratory Analysis: End-of-Quarter vs End-of-Month Reversal

Question: Is the reversal effect stronger at quarter-ends (Mar/Jun/Sep/Dec)
compared to regular month-ends?

Hypothesis: Quarter-end rebalancing is more synchronized and larger in magnitude,
creating stronger reversal patterns.

Similar to Test 2a, but split by:
- EOQ months (March, June, September, December)
- Non-EOQ months (all others)

X-axis: Cumulative ratio return from day 1 to day before last 5
Y-axis: Cumulative ratio return during last 5 days
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
print("EXPLORATORY: END-OF-QUARTER VS END-OF-MONTH REVERSAL")
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

# Calculate ratio return (SPY/TLT) - log returns
df['ratio_return'] = np.log(1 + df['spy_return']) - np.log(1 + df['tlt_return'])

# Add year and month columns
df['year'] = df.index.year
df['month'] = df.index.month
df['year_month'] = df.index.to_period('M')

# Identify quarter-end months
df['is_eoq_month'] = df['month'].isin([3, 6, 9, 12])

# Calculate trading day of month
df['trading_day'] = df.groupby('year_month').cumcount() + 1

# Calculate last trading day of each month
df['days_in_month'] = df.groupby('year_month')['trading_day'].transform('max')

# Determine if we're in the last 5 trading days
df['last_5_threshold'] = (df['days_in_month'] - 4).clip(lower=19)
df['is_last_5'] = df['trading_day'] >= df['last_5_threshold']

print("Analyzing reversal patterns by month type...")
print()

# For each month, calculate cumulative returns for first days vs last 5 days
monthly_stats_eoq = []
monthly_stats_non_eoq = []

periods = sorted(df['year_month'].unique())

for period in periods:
    group = df[df['year_month'] == period]

    # Get the days before last 5 and the last 5 days
    first_days = group[~group['is_last_5']]
    last_5_days = group[group['is_last_5']]

    if len(first_days) > 0 and len(last_5_days) > 0:
        first_days_cum_ret = first_days['ratio_return'].sum()
        last_5_cum_ret = last_5_days['ratio_return'].sum()

        month_data = {
            'period': period,
            'month': period.month,
            'year': period.year,
            'first_days_return': first_days_cum_ret,
            'last_5_return': last_5_cum_ret,
            'first_days_count': len(first_days),
            'last_5_count': len(last_5_days)
        }

        # Split by EOQ vs non-EOQ
        if period.month in [3, 6, 9, 12]:
            monthly_stats_eoq.append(month_data)
        else:
            monthly_stats_non_eoq.append(month_data)

eoq_df = pd.DataFrame(monthly_stats_eoq)
non_eoq_df = pd.DataFrame(monthly_stats_non_eoq)

print(f"EOQ months (Mar/Jun/Sep/Dec): {len(eoq_df)} months")
print(f"Non-EOQ months: {len(non_eoq_df)} months")
print()

# Calculate correlations
corr_eoq = eoq_df['first_days_return'].corr(eoq_df['last_5_return'])
corr_non_eoq = non_eoq_df['first_days_return'].corr(non_eoq_df['last_5_return'])

# Summary statistics
print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
print()
print("EOQ MONTHS (March, June, September, December):")
print(f"  First days mean:       {eoq_df['first_days_return'].mean()*100:>8.3f}%")
print(f"  First days std:        {eoq_df['first_days_return'].std()*100:>8.3f}%")
print(f"  Last 5 days mean:      {eoq_df['last_5_return'].mean()*100:>8.3f}%")
print(f"  Last 5 days std:       {eoq_df['last_5_return'].std()*100:>8.3f}%")
print(f"  Correlation:           {corr_eoq:>8.3f}")

print()
print("NON-EOQ MONTHS:")
print(f"  First days mean:       {non_eoq_df['first_days_return'].mean()*100:>8.3f}%")
print(f"  First days std:        {non_eoq_df['first_days_return'].std()*100:>8.3f}%")
print(f"  Last 5 days mean:      {non_eoq_df['last_5_return'].mean()*100:>8.3f}%")
print(f"  Last 5 days std:       {non_eoq_df['last_5_return'].std()*100:>8.3f}%")
print(f"  Correlation:           {corr_non_eoq:>8.3f}")

print()
print("="*80)
print("REVERSAL STRENGTH COMPARISON")
print("="*80)
print(f"EOQ correlation:       {corr_eoq:>8.3f}")
print(f"Non-EOQ correlation:   {corr_non_eoq:>8.3f}")
print(f"Difference:            {corr_eoq - corr_non_eoq:>8.3f}")
print()
if corr_eoq < corr_non_eoq:
    print("✓ EOQ months show STRONGER reversal effect")
    print(f"  ({abs(corr_eoq):.3f} vs {abs(corr_non_eoq):.3f} in absolute terms)")
else:
    print("✗ Non-EOQ months show stronger (or equal) reversal effect")

# Save results
combined_df = pd.concat([
    eoq_df.assign(month_type='EOQ'),
    non_eoq_df.assign(month_type='Non-EOQ')
])
combined_df.to_csv('../data/exploratory_eoq_vs_eom.csv', index=False)
print(f"\nResults saved to: ../data/exploratory_eoq_vs_eom.csv")

# Visualization - Two scatter plots side by side
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# EOQ months
ax1 = axes[0]
x_eoq = eoq_df['first_days_return'] * 100
y_eoq = eoq_df['last_5_return'] * 100

ax1.scatter(x_eoq, y_eoq, alpha=0.6, s=50, color='red', edgecolor='black', linewidth=0.5)

# Add reference lines
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

# Add regression line
z_eoq = np.polyfit(x_eoq, y_eoq, 1)
p_eoq = np.poly1d(z_eoq)
x_line_eoq = np.linspace(x_eoq.min(), x_eoq.max(), 100)
ax1.plot(x_line_eoq, p_eoq(x_line_eoq), "darkred", linestyle='--', alpha=0.8, linewidth=2.5,
        label=f'Regression: y={z_eoq[0]:.3f}x + {z_eoq[1]:.3f}')

# Add mean lines
ax1.axhline(y=y_eoq.mean(), color='orange', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'Mean Last 5: {y_eoq.mean():.3f}%')
ax1.axvline(x=x_eoq.mean(), color='purple', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'Mean First Days: {x_eoq.mean():.3f}%')

ax1.set_xlabel('First Days Cumulative Return (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Last 5 Days Cumulative Return (%)', fontsize=12, fontweight='bold')
ax1.set_title(f'EOQ MONTHS (Mar/Jun/Sep/Dec)\\n'
             f'Correlation: {corr_eoq:.3f} | N={len(eoq_df)} months',
             fontsize=14, fontweight='bold', color='darkred')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Non-EOQ months
ax2 = axes[1]
x_non_eoq = non_eoq_df['first_days_return'] * 100
y_non_eoq = non_eoq_df['last_5_return'] * 100

ax2.scatter(x_non_eoq, y_non_eoq, alpha=0.6, s=50, color='steelblue', edgecolor='black', linewidth=0.5)

# Add reference lines
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

# Add regression line
z_non_eoq = np.polyfit(x_non_eoq, y_non_eoq, 1)
p_non_eoq = np.poly1d(z_non_eoq)
x_line_non_eoq = np.linspace(x_non_eoq.min(), x_non_eoq.max(), 100)
ax2.plot(x_line_non_eoq, p_non_eoq(x_line_non_eoq), "darkblue", linestyle='--', alpha=0.8, linewidth=2.5,
        label=f'Regression: y={z_non_eoq[0]:.3f}x + {z_non_eoq[1]:.3f}')

# Add mean lines
ax2.axhline(y=y_non_eoq.mean(), color='green', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'Mean Last 5: {y_non_eoq.mean():.3f}%')
ax2.axvline(x=x_non_eoq.mean(), color='purple', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'Mean First Days: {x_non_eoq.mean():.3f}%')

ax2.set_xlabel('First Days Cumulative Return (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Last 5 Days Cumulative Return (%)', fontsize=12, fontweight='bold')
ax2.set_title(f'NON-EOQ MONTHS (All Others)\\n'
             f'Correlation: {corr_non_eoq:.3f} | N={len(non_eoq_df)} months',
             fontsize=14, fontweight='bold', color='darkblue')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle(f'SPY/TLT Ratio: EOQ vs Non-EOQ Reversal Comparison\\n'
             f'EOQ Correlation: {corr_eoq:.3f} | Non-EOQ Correlation: {corr_non_eoq:.3f}',
             fontsize=16, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig('../plots/exploratory_eoq_vs_eom.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/exploratory_eoq_vs_eom.png")

# Additional analysis: Combined plot showing both
fig2, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot both on same axes with different colors
ax.scatter(x_eoq, y_eoq, alpha=0.6, s=60, color='red',
           edgecolor='darkred', linewidth=1, label=f'EOQ months (corr={corr_eoq:.3f})', marker='o')
ax.scatter(x_non_eoq, y_non_eoq, alpha=0.4, s=40, color='steelblue',
           edgecolor='darkblue', linewidth=0.8, label=f'Non-EOQ months (corr={corr_non_eoq:.3f})', marker='s')

# Add reference lines
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

# Add regression lines
ax.plot(x_line_eoq, p_eoq(x_line_eoq), "darkred", linestyle='-', alpha=0.8, linewidth=2.5,
        label=f'EOQ regression (slope={z_eoq[0]:.3f})')
ax.plot(x_line_non_eoq, p_non_eoq(x_line_non_eoq), "darkblue", linestyle='--', alpha=0.8, linewidth=2.5,
        label=f'Non-EOQ regression (slope={z_non_eoq[0]:.3f})')

ax.set_xlabel('First Days Cumulative Return (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Last 5 Days Cumulative Return (%)', fontsize=13, fontweight='bold')
ax.set_title(f'SPY/TLT Ratio Reversal: Quarter-End vs Regular Months\\n'
             f'Stronger negative correlation at EOQ suggests larger synchronized rebalancing',
             fontsize=15, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/exploratory_eoq_vs_eom_combined.png', dpi=300, bbox_inches='tight')
print(f"Combined plot saved to: ../plots/exploratory_eoq_vs_eom_combined.png")

# Regression slope comparison
print()
print("="*80)
print("REGRESSION SLOPE ANALYSIS")
print("="*80)
print(f"EOQ slope:       {z_eoq[0]:>8.3f}")
print(f"Non-EOQ slope:   {z_non_eoq[0]:>8.3f}")
print(f"Ratio (EOQ/Non): {z_eoq[0]/z_non_eoq[0]:>8.3f}x")
print()
print("More negative slope = stronger reversal effect")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print()
if abs(corr_eoq) > abs(corr_non_eoq) * 1.1:  # At least 10% stronger
    print(f"✓ EOQ months show MEANINGFULLY STRONGER reversal")
    print(f"  Correlation: {corr_eoq:.3f} vs {corr_non_eoq:.3f}")
    print(f"  This supports the hypothesis of larger quarter-end rebalancing flows")
elif abs(corr_eoq) > abs(corr_non_eoq):
    print(f"≈ EOQ months show SLIGHTLY STRONGER reversal")
    print(f"  Correlation: {corr_eoq:.3f} vs {corr_non_eoq:.3f}")
    print(f"  Effect is present but not dramatic")
else:
    print(f"✗ EOQ months DO NOT show stronger reversal")
    print(f"  Correlation: {corr_eoq:.3f} vs {corr_non_eoq:.3f}")
    print(f"  Regular month-end effects appear similar to quarter-end")
