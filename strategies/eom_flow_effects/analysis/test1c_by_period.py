"""
TEST 1c: TLT Returns by Trading Day - Across 5-Year Periods

Show the same trading day analysis as TEST 1, but broken down by 5-year periods
to visualize how the EOM effect has evolved over time.
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
print("TEST 1c: TLT RETURNS BY TRADING DAY - ACROSS 5-YEAR PERIODS")
print("="*80)
print()

# Load TLT data
client = FMPClient()
print("Loading TLT data...")
tlt_data = client.get_historical_prices('TLT', '2002-07-26', '2025-10-29')
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

# Calculate trading day of month
tlt_df['trading_day'] = tlt_df.groupby('year_month').cumcount() + 1

# Define 5-year buckets
year_min = tlt_df['year'].min()
year_max = tlt_df['year'].max()
buckets = []
bucket_start = year_min

while bucket_start <= year_max:
    bucket_end = min(bucket_start + 4, year_max)
    buckets.append((bucket_start, bucket_end))
    bucket_start += 5

print(f"Created {len(buckets)} 5-year buckets:")
for start, end in buckets:
    print(f"  {start}-{end}")
print()

# Calculate stats for each bucket
bucket_stats = {}

for bucket_start, bucket_end in buckets:
    bucket_name = f"{bucket_start}-{bucket_end}"
    bucket_df = tlt_df[(tlt_df['year'] >= bucket_start) & (tlt_df['year'] <= bucket_end)]

    if len(bucket_df) > 0:
        day_stats = bucket_df.groupby('trading_day')['return'].agg([
            ('count', 'count'),
            ('mean_return', 'mean'),
        ])
        day_stats['mean_return_bps'] = day_stats['mean_return'] * 10000
        bucket_stats[bucket_name] = day_stats

        print(f"{bucket_name}: {len(bucket_df)} days")

# Create visualization - One subplot per period
n_buckets = len(buckets)
n_cols = 2
n_rows = int(np.ceil(n_buckets / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
axes = axes.flatten() if n_buckets > 1 else [axes]

for idx, (bucket_start, bucket_end) in enumerate(buckets):
    bucket_name = f"{bucket_start}-{bucket_end}"
    ax = axes[idx]

    if bucket_name in bucket_stats:
        stats = bucket_stats[bucket_name]
        days_to_plot = stats.index[stats.index <= 23]

        ax.bar(days_to_plot, stats.loc[days_to_plot, 'mean_return_bps'],
               color=['green' if x > 0 else 'red' for x in stats.loc[days_to_plot, 'mean_return_bps']],
               alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

        # Calculate period mean for reference line
        period_df = tlt_df[(tlt_df['year'] >= bucket_start) & (tlt_df['year'] <= bucket_end)]
        period_mean = period_df['return'].mean() * 10000
        ax.axhline(y=period_mean, color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Period Mean ({period_mean:.2f} bps)')

        ax.set_xlabel('Trading Day of Month', fontsize=11)
        ax.set_ylabel('Mean Return (bps)', fontsize=11)
        ax.set_title(f'{bucket_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)
        ax.set_xticks(range(1, 24))
        ax.set_xticklabels(range(1, 24), fontsize=9)

# Hide any unused subplots
for idx in range(len(buckets), len(axes)):
    axes[idx].axis('off')

plt.suptitle('TLT Mean Return by Trading Day of Month - Across 5-Year Periods',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('../plots/test1c_by_period.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: ../plots/test1c_by_period.png")

# Save detailed stats to CSV
all_stats = []
for bucket_name, stats in bucket_stats.items():
    stats_copy = stats.copy()
    stats_copy['period'] = bucket_name
    stats_copy['trading_day'] = stats_copy.index
    all_stats.append(stats_copy)

if all_stats:
    combined_stats = pd.concat(all_stats)
    combined_stats.to_csv('../data/test1c_by_period_stats.csv', index=False)
    print(f"Stats saved to: ../data/test1c_by_period_stats.csv")
