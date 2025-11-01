"""
TEST 1d: TLT End-of-Month Flow Strategy

Trading Strategy:
- LONG TLT during last 5 trading days of each month (days 19-23, or last 5 available)
- SHORT TLT during first 5 trading days of each month (days 1-5)
- FLAT all other times

Performance metrics: CAGR, Sharpe, Max Drawdown, Calmar
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
print("TEST 1d: TLT END-OF-MONTH FLOW STRATEGY")
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

# Calculate last trading day of each month
tlt_df['days_in_month'] = tlt_df.groupby('year_month')['trading_day'].transform('max')

# Determine if we're in the last 5 trading days
# Last 5 days = max(19, days_in_month - 4) through days_in_month
tlt_df['last_5_threshold'] = (tlt_df['days_in_month'] - 4).clip(lower=19)
tlt_df['is_last_5'] = tlt_df['trading_day'] >= tlt_df['last_5_threshold']

# Strategy positions:
# - SHORT during first 5 days (days 1-5)
# - LONG during last 5 days
# - FLAT otherwise
def get_position(row):
    """
    Determine position for each day.
    Returns: 1 (long), -1 (short), 0 (flat)
    """
    if row['trading_day'] <= 5:
        return -1  # Short first 5 days
    elif row['is_last_5']:
        return 1  # Long last 5 days
    else:
        return 0  # Flat otherwise

tlt_df['position'] = tlt_df.apply(get_position, axis=1)

# Calculate strategy returns
# Strategy return = position * daily return
tlt_df['strategy_return'] = tlt_df['position'] * tlt_df['return']

# Calculate cumulative returns
tlt_df['cum_return'] = (1 + tlt_df['strategy_return']).cumprod()
tlt_df['cum_bh_return'] = (1 + tlt_df['return']).cumprod()  # Buy & hold for comparison

# Calculate drawdown
tlt_df['running_max'] = tlt_df['cum_return'].cummax()
tlt_df['drawdown'] = (tlt_df['cum_return'] - tlt_df['running_max']) / tlt_df['running_max']

print("Strategy Summary:")
print("-" * 80)
print(f"Position breakdown:")
position_counts = tlt_df['position'].value_counts().sort_index()
for pos, count in position_counts.items():
    pos_name = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}[pos]
    pct = count / len(tlt_df) * 100
    print(f"  {pos_name:>5}: {count:>5} days ({pct:>5.1f}%)")
print()

# Performance metrics
total_days = len(tlt_df)
years = total_days / 252

final_value = tlt_df['cum_return'].iloc[-1]
cagr = (final_value ** (1 / years)) - 1

returns = tlt_df['strategy_return']
mean_return = returns.mean()
std_return = returns.std()
sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

max_dd = tlt_df['drawdown'].min()
calmar = cagr / abs(max_dd) if max_dd != 0 else 0

# Buy & hold comparison
bh_final = tlt_df['cum_bh_return'].iloc[-1]
bh_cagr = (bh_final ** (1 / years)) - 1
bh_returns = tlt_df['return']
bh_sharpe = (bh_returns.mean() / bh_returns.std()) * np.sqrt(252)
bh_running_max = tlt_df['cum_bh_return'].cummax()
bh_dd = ((tlt_df['cum_bh_return'] - bh_running_max) / bh_running_max).min()
bh_calmar = bh_cagr / abs(bh_dd) if bh_dd != 0 else 0

print("="*80)
print("PERFORMANCE METRICS")
print("="*80)
print(f"\n{'Metric':<25} {'Strategy':>15} {'Buy & Hold':>15}")
print("-"*80)
print(f"{'Total Return':<25} {(final_value - 1):>14.2%} {(bh_final - 1):>14.2%}")
print(f"{'CAGR':<25} {cagr:>14.2%} {bh_cagr:>14.2%}")
print(f"{'Sharpe Ratio':<25} {sharpe:>14.2f} {bh_sharpe:>14.2f}")
print(f"{'Max Drawdown':<25} {max_dd:>14.2%} {bh_dd:>14.2%}")
print(f"{'Calmar Ratio':<25} {calmar:>14.2f} {bh_calmar:>14.2f}")
print(f"{'Annual Volatility':<25} {std_return * np.sqrt(252):>14.2%} {bh_returns.std() * np.sqrt(252):>14.2%}")
print(f"{'Total Days':<25} {total_days:>15} {total_days:>15}")
print(f"{'Years':<25} {years:>14.1f} {years:>14.1f}")

# Save results
results = pd.DataFrame({
    'metric': ['Total Return', 'CAGR', 'Sharpe', 'Max DD', 'Calmar', 'Volatility'],
    'strategy': [(final_value - 1), cagr, sharpe, max_dd, calmar, std_return * np.sqrt(252)],
    'buy_hold': [(bh_final - 1), bh_cagr, bh_sharpe, bh_dd, bh_calmar, bh_returns.std() * np.sqrt(252)]
})
results.to_csv('../data/test1d_performance.csv', index=False)
print(f"\nResults saved to: ../data/test1d_performance.csv")

# Save daily data
tlt_df[['adjClose', 'return', 'trading_day', 'position', 'strategy_return',
        'cum_return', 'drawdown']].to_csv('../data/test1d_daily_data.csv')
print(f"Daily data saved to: ../data/test1d_daily_data.csv")

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

# Equity curve
ax1 = axes[0]
ax1.plot(tlt_df.index, tlt_df['cum_return'], label='Strategy', linewidth=2, color='darkblue')
ax1.plot(tlt_df.index, tlt_df['cum_bh_return'], label='Buy & Hold TLT',
         linewidth=1.5, color='gray', alpha=0.6, linestyle='--')
ax1.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
ax1.set_title(f'TLT End-of-Month Flow Strategy\n'
              f'CAGR: {cagr:.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%} | Calmar: {calmar:.2f}',
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Drawdown
ax2 = axes[1]
ax2.fill_between(tlt_df.index, tlt_df['drawdown'] * 100, 0,
                  color='red', alpha=0.3, label='Drawdown')
ax2.plot(tlt_df.index, tlt_df['drawdown'] * 100, color='darkred', linewidth=1)
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([max_dd * 100 * 1.1, 5])  # Add some padding

plt.tight_layout()
plt.savefig('../plots/test1d_tlt_strategy.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/test1d_tlt_strategy.png")

# Print sample of positions to verify logic
print("\n" + "="*80)
print("SAMPLE POSITIONS (First 60 trading days)")
print("="*80)
print(f"{'Date':<12} {'Day':>4} {'Last5?':>7} {'Pos':>5} {'Return':>8} {'Strat Ret':>10}")
print("-"*80)
for i in range(min(60, len(tlt_df))):
    row = tlt_df.iloc[i]
    pos_name = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}[row['position']]
    print(f"{row.name.date()} {row['trading_day']:>4} "
          f"{'YES' if row['is_last_5'] else 'NO':>7} {pos_name:>5} "
          f"{row['return']*100:>7.2f}% {row['strategy_return']*100:>9.2f}%")
