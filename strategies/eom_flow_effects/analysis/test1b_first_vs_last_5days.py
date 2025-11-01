"""
TEST 1b: First 5 Days vs Last 5 Days by Period

Compare the first 5 trading days vs last 5 trading days of each month,
broken down by 5-year periods to see if the EOM effect is consistent.
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
print("TEST 1b: FIRST 5 DAYS VS LAST 5 DAYS")
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

# Calculate stats for first 5 vs last 5 days for each bucket
bucket_results = []

for bucket_start, bucket_end in buckets:
    bucket_name = f"{bucket_start}-{bucket_end}"
    bucket_df = tlt_df[(tlt_df['year'] >= bucket_start) & (tlt_df['year'] <= bucket_end)]

    if len(bucket_df) > 0:
        # First 5 days
        first_5 = bucket_df[bucket_df['trading_day'] <= 5]
        first_5_mean = first_5['return'].mean() * 10000  # bps
        first_5_count = len(first_5)

        # Last 5 days
        last_5 = bucket_df[bucket_df['trading_day'] >= 19]
        last_5_mean = last_5['return'].mean() * 10000  # bps
        last_5_count = len(last_5)

        bucket_results.append({
            'period': bucket_name,
            'first_5_mean': first_5_mean,
            'last_5_mean': last_5_mean,
            'first_5_count': first_5_count,
            'last_5_count': last_5_count,
            'bucket_start': bucket_start,
            'bucket_end': bucket_end
        })

        print(f"{bucket_name}:")
        print(f"  First 5 days:  {first_5_mean:>7.2f} bps  (n={first_5_count})")
        print(f"  Last 5 days:   {last_5_mean:>7.2f} bps  (n={last_5_count})")
        print(f"  Difference:    {last_5_mean - first_5_mean:>7.2f} bps")
        print()

# Create visualization
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

x = np.arange(len(bucket_results))
width = 0.35

first_5_vals = [r['first_5_mean'] for r in bucket_results]
last_5_vals = [r['last_5_mean'] for r in bucket_results]
labels = [r['period'] for r in bucket_results]

bars1 = ax.bar(x - width/2, first_5_vals, width, label='First 5 Days',
               color='lightcoral', edgecolor='black', linewidth=1, alpha=0.8)
bars2 = ax.bar(x + width/2, last_5_vals, width, label='Last 5 Days',
               color='lightgreen', edgecolor='black', linewidth=1, alpha=0.8)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Time Period', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Return (bps)', fontsize=14, fontweight='bold')
ax.set_title('TLT: First 5 Days vs Last 5 Days of Month by Period', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=10, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('../plots/test1b_first_vs_last_5days.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/test1b_first_vs_last_5days.png")
