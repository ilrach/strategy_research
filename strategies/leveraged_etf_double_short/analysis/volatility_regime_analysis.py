"""
Realized Volatility Regime Analysis for Double Short TQQQ/SQQQ Strategy

Bucket daily returns by SPY realized volatility decile to see if strategy performs better
in higher volatility environments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

client = FMPClient()

# Fetch data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)

from_date = start_date.strftime('%Y-%m-%d')
to_date = end_date.strftime('%Y-%m-%d')

print("Fetching data...")
print(f"Period: {from_date} to {to_date}")
print()

# Fetch TQQQ, SQQQ, BIL, and SPY
print("  TQQQ...")
tqqq_data = client.get_historical_prices('TQQQ', from_date=from_date, to_date=to_date)
print("  SQQQ...")
sqqq_data = client.get_historical_prices('SQQQ', from_date=from_date, to_date=to_date)
print("  BIL...")
bil_data = client.get_historical_prices('BIL', from_date=from_date, to_date=to_date)
print("  SPY...")
spy_data = client.get_historical_prices('SPY', from_date=from_date, to_date=to_date)
print()

# Convert to DataFrames
df_tqqq = pd.DataFrame(tqqq_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_tqqq'})
df_sqqq = pd.DataFrame(sqqq_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_sqqq'})
df_bil = pd.DataFrame(bil_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_bil'})
df_spy = pd.DataFrame(spy_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_spy'})

# Convert dates
for df in [df_tqqq, df_sqqq, df_bil, df_spy]:
    df['date'] = pd.to_datetime(df['date'])

# Merge all data
df = df_tqqq.set_index('date').join(df_sqqq.set_index('date'))
df = df.join(df_bil.set_index('date'))
df = df.join(df_spy.set_index('date'))
df = df.sort_index()

print(f"\nData loaded: {len(df)} days")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
print()

# Calculate daily returns
df['tqqq_ret'] = df['p_tqqq'].pct_change()
df['sqqq_ret'] = df['p_sqqq'].pct_change()
df['bil_ret'] = df['p_bil'].pct_change()
df['spy_ret'] = df['p_spy'].pct_change()

# Calculate rolling 20-day realized volatility of SPY
# Using 20-day window (approximately 1 month)
df['spy_rvol'] = df['spy_ret'].rolling(window=20).std() * np.sqrt(252) * 100  # annualized %

# Strategy return: 2×BIL - 0.5×TQQQ - 0.5×SQQQ
df['strategy_ret'] = 2 * df['bil_ret'] - 0.5 * df['tqqq_ret'] - 0.5 * df['sqqq_ret']

# Drop NaN rows (first 20 days due to rolling vol calculation)
df = df.dropna()

print(f"Valid days with returns: {len(df)}")
print()

# Calculate realized vol deciles
print("="*80)
print("VOLATILITY REGIME ANALYSIS (SPY 20-Day Realized Vol Deciles)")
print("="*80)
print()

df['rvol_decile'] = pd.qcut(df['spy_rvol'], q=10, labels=False, duplicates='drop') + 1

# Summary statistics by realized vol decile
print("SPY Realized Volatility Decile Ranges:")
for decile in sorted(df['rvol_decile'].unique()):
    decile_data = df[df['rvol_decile'] == decile]
    rvol_min = decile_data['spy_rvol'].min()
    rvol_max = decile_data['spy_rvol'].max()
    rvol_mean = decile_data['spy_rvol'].mean()
    print(f"  Decile {int(decile):2d}: {rvol_min:5.2f}% - {rvol_max:5.2f}% (mean: {rvol_mean:5.2f}%)")

print()
print("="*80)
print("STRATEGY PERFORMANCE BY VOLATILITY DECILE")
print("="*80)
print()

# Calculate metrics by decile
decile_stats = []

for decile in sorted(df['rvol_decile'].unique()):
    decile_data = df[df['rvol_decile'] == decile]

    # Strategy metrics
    mean_daily_ret = decile_data['strategy_ret'].mean() * 100  # in %
    annualized_ret = mean_daily_ret * 252  # approximate annualized
    volatility = decile_data['strategy_ret'].std() * np.sqrt(252) * 100  # annualized %
    sharpe = (mean_daily_ret * 252) / volatility if volatility > 0 else 0

    # Win rate
    win_rate = (decile_data['strategy_ret'] > 0).sum() / len(decile_data) * 100

    # Realized vol stats
    rvol_min = decile_data['spy_rvol'].min()
    rvol_max = decile_data['spy_rvol'].max()
    rvol_mean = decile_data['spy_rvol'].mean()

    # Number of days
    n_days = len(decile_data)

    decile_stats.append({
        'Decile': int(decile),
        'RVol Range': f"{rvol_min:.1f}%-{rvol_max:.1f}%",
        'RVol Mean': rvol_mean,
        'Days': n_days,
        'Avg Daily Ret (bps)': mean_daily_ret * 100,  # basis points
        'Ann. Return (%)': annualized_ret,
        'Volatility (%)': volatility,
        'Sharpe': sharpe,
        'Win Rate (%)': win_rate,
        'mean_daily_ret': mean_daily_ret,  # for plotting
    })

stats_df = pd.DataFrame(decile_stats)

# Print table
print(stats_df[['Decile', 'RVol Range', 'Days', 'Avg Daily Ret (bps)', 'Ann. Return (%)',
                'Volatility (%)', 'Sharpe', 'Win Rate (%)']].to_string(index=False))

print()
print("="*80)
print("KEY FINDINGS")
print("="*80)
print()

# Compare low vs high vol regimes
low_vol_deciles = stats_df[stats_df['Decile'] <= 3]['Ann. Return (%)'].mean()
high_vol_deciles = stats_df[stats_df['Decile'] >= 8]['Ann. Return (%)'].mean()

print(f"Low volatility (deciles 1-3): {low_vol_deciles:.2f}% annualized")
print(f"High volatility (deciles 8-10): {high_vol_deciles:.2f}% annualized")
print(f"Difference: {high_vol_deciles - low_vol_deciles:+.2f}%")
print()

best_decile = stats_df.loc[stats_df['Ann. Return (%)'].idxmax()]
worst_decile = stats_df.loc[stats_df['Ann. Return (%)'].idxmin()]

print(f"Best performing decile: {int(best_decile['Decile'])} (RVol {best_decile['RVol Range']})")
print(f"  Annualized return: {best_decile['Ann. Return (%)']:.2f}%")
print(f"  Sharpe ratio: {best_decile['Sharpe']:.2f}")
print()

print(f"Worst performing decile: {int(worst_decile['Decile'])} (RVol {worst_decile['RVol Range']})")
print(f"  Annualized return: {worst_decile['Ann. Return (%)']:.2f}%")
print(f"  Sharpe ratio: {worst_decile['Sharpe']:.2f}")
print()

# Generate plots
print("="*80)
print("GENERATING PLOTS")
print("="*80)
print()

# 1. Returns by realized vol decile
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Annualized return by decile
axes[0, 0].bar(stats_df['Decile'], stats_df['Ann. Return (%)'], color='#2E86AB', alpha=0.7)
axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[0, 0].set_xlabel('SPY Realized Vol Decile (1=Low Vol, 10=High Vol)', fontsize=11)
axes[0, 0].set_ylabel('Annualized Return (%)', fontsize=11)
axes[0, 0].set_title('Strategy Returns by SPY Realized Volatility Decile', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Sharpe ratio by decile
axes[0, 1].bar(stats_df['Decile'], stats_df['Sharpe'], color='#F18F01', alpha=0.7)
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[0, 1].set_xlabel('SPY Realized Vol Decile', fontsize=11)
axes[0, 1].set_ylabel('Sharpe Ratio', fontsize=11)
axes[0, 1].set_title('Sharpe Ratio by Realized Volatility Decile', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Volatility by decile
axes[1, 0].bar(stats_df['Decile'], stats_df['Volatility (%)'], color='#C73E1D', alpha=0.7)
axes[1, 0].set_xlabel('SPY Realized Vol Decile', fontsize=11)
axes[1, 0].set_ylabel('Strategy Annualized Volatility (%)', fontsize=11)
axes[1, 0].set_title('Strategy Volatility by Realized Volatility Decile', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Win rate by decile
axes[1, 1].bar(stats_df['Decile'], stats_df['Win Rate (%)'], color='#06A77D', alpha=0.7)
axes[1, 1].axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
axes[1, 1].set_xlabel('SPY Realized Vol Decile', fontsize=11)
axes[1, 1].set_ylabel('Win Rate (%)', fontsize=11)
axes[1, 1].set_title('Win Rate by Realized Volatility Decile', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../plots/leveraged_etf/rvol_regime_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: rvol_regime_analysis.png")

# 2. Scatter plot: SPY realized vol vs daily return
fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(df['spy_rvol'], df['strategy_ret'] * 100, alpha=0.3, s=20, c=df['spy_rvol'], cmap='coolwarm')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('SPY 20-Day Realized Volatility (% ann.)', fontsize=12)
ax.set_ylabel('Daily Strategy Return (%)', fontsize=12)
ax.set_title('Daily Returns vs SPY Realized Volatility', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('SPY Realized Vol (%)', fontsize=11)

# Add trend line
z = np.polyfit(df['spy_rvol'], df['strategy_ret'] * 100, 1)
p = np.poly1d(z)
ax.plot(df['spy_rvol'].sort_values(), p(df['spy_rvol'].sort_values()), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('../../plots/leveraged_etf/rvol_vs_returns_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: rvol_vs_returns_scatter.png")

# Save detailed stats to CSV
stats_df.to_csv('rvol_decile_stats.csv', index=False)
print()
print(f"Detailed statistics saved to: rvol_decile_stats.csv")
print()
print("Analysis complete!")
