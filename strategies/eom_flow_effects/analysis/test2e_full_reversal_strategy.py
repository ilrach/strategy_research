"""
TEST 2e: Full Reversal Strategy (Last 5 + Inverted First 5)

Trading Strategy:
- Calculate SPY/TLT ratio return from day 1 to day before last 5 (signal period)
- LAST 5 DAYS of current month:
  * If SPY outperforms TLT: SHORT ratio (short SPY, long TLT)
  * If TLT outperforms SPY: LONG ratio (long SPY, short TLT)
- FIRST 5 DAYS of next month (INVERTED):
  * If SPY outperformed TLT last month: LONG ratio (long SPY, short TLT)
  * If TLT outperformed SPY last month: SHORT ratio (short SPY, long TLT)
- FLAT all other times

This captures both the month-end reversal AND the subsequent re-reversal.
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
print("TEST 2e: FULL REVERSAL STRATEGY (LAST 5 + INVERTED FIRST 5)")
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

# Calculate ratio return (SPY/TLT) - log returns
df['ratio_return'] = np.log(1 + df['spy_return']) - np.log(1 + df['tlt_return'])

# For each month, calculate cumulative ratio return up to last 5 days
# Store the signal for both last 5 of current month and first 5 of next month
df['signal'] = 0.0

periods = sorted(df['year_month'].unique())

for i, period in enumerate(periods):
    group = df[df['year_month'] == period]

    # Get the days before last 5
    first_days = group[~group['is_last_5']]

    if len(first_days) > 0:
        # Cumulative ratio return in first days
        cum_ratio_return = first_days['ratio_return'].sum()

        # LAST 5 DAYS: Set signal for reversal
        # If SPY outperformed (positive ratio), signal is -1 (short ratio)
        # If TLT outperformed (negative ratio), signal is +1 (long ratio)
        last_5_idx = group[group['is_last_5']].index
        if len(last_5_idx) > 0:
            last_5_signal = -1 if cum_ratio_return > 0 else 1
            df.loc[last_5_idx, 'signal'] = last_5_signal

            # FIRST 5 DAYS OF NEXT MONTH: Invert the signal
            # If we shorted the ratio in last 5, we long it in first 5 of next month
            if i + 1 < len(periods):
                next_period = periods[i + 1]
                next_group = df[df['year_month'] == next_period]
                first_5_next_idx = next_group[next_group['is_first_5']].index
                if len(first_5_next_idx) > 0:
                    # Inverted signal
                    df.loc[first_5_next_idx, 'signal'] = -last_5_signal

# Strategy return is trading the ratio
# signal=1: long ratio (long SPY, short TLT)
# signal=-1: short ratio (short SPY, long TLT)
df['strategy_return'] = df['signal'] * df['ratio_return']

# Calculate cumulative returns
df['cum_return'] = (1 + df['strategy_return']).cumprod()
df['cum_ratio_return'] = (1 + df['ratio_return']).cumprod()  # Buy & hold ratio for comparison

# Calculate drawdown
df['running_max'] = df['cum_return'].cummax()
df['drawdown'] = (df['cum_return'] - df['running_max']) / df['running_max']

print("Strategy Summary:")
print("-" * 80)
print(f"Signal breakdown:")
signal_counts = df['signal'].value_counts().sort_index()
for sig, count in signal_counts.items():
    sig_name = {-1: 'SHORT Ratio (Short SPY, Long TLT)', 0: 'FLAT', 1: 'LONG Ratio (Long SPY, Short TLT)'}[sig]
    pct = count / len(df) * 100
    print(f"  {sig_name}: {count:>5} days ({pct:>5.1f}%)")
print()

# Performance metrics
total_days = len(df)
years = total_days / 252

final_value = df['cum_return'].iloc[-1]
cagr = (final_value ** (1 / years)) - 1

returns = df['strategy_return']
mean_return = returns.mean()
std_return = returns.std()
sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

max_dd = df['drawdown'].min()
calmar = cagr / abs(max_dd) if max_dd != 0 else 0

# Buy & hold ratio comparison
bh_final = df['cum_ratio_return'].iloc[-1]
bh_cagr = (bh_final ** (1 / years)) - 1
bh_returns = df['ratio_return']
bh_sharpe = (bh_returns.mean() / bh_returns.std()) * np.sqrt(252) if bh_returns.std() > 0 else 0
bh_running_max = df['cum_ratio_return'].cummax()
bh_dd = ((df['cum_ratio_return'] - bh_running_max) / bh_running_max).min()
bh_calmar = bh_cagr / abs(bh_dd) if bh_dd != 0 else 0

print("="*80)
print("PERFORMANCE METRICS")
print("="*80)
print(f"\n{'Metric':<25} {'Strategy':>15} {'B&H Ratio':>15}")
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
    'buy_hold_ratio': [(bh_final - 1), bh_cagr, bh_sharpe, bh_dd, bh_calmar, bh_returns.std() * np.sqrt(252)]
})
results.to_csv('../data/test2e_performance.csv', index=False)
print(f"\nResults saved to: ../data/test2e_performance.csv")

# Save daily data
df[['spy_return', 'tlt_return', 'ratio_return', 'trading_day', 'is_last_5', 'is_first_5',
    'signal', 'strategy_return', 'cum_return', 'drawdown']].to_csv('../data/test2e_daily_data.csv')
print(f"Daily data saved to: ../data/test2e_daily_data.csv")

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

# Equity curve
ax1 = axes[0]
ax1.plot(df.index, df['cum_return'], label='Strategy', linewidth=2, color='darkblue')
ax1.plot(df.index, df['cum_ratio_return'], label='Buy & Hold SPY/TLT Ratio',
         linewidth=1.5, color='gray', alpha=0.6, linestyle='--')
ax1.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
ax1.set_title(f'Full Reversal Strategy (Last 5 Days + Inverted First 5 Days)\n'
              f'CAGR: {cagr:.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%} | Calmar: {calmar:.2f}',
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Drawdown
ax2 = axes[1]
ax2.fill_between(df.index, df['drawdown'] * 100, 0,
                  color='red', alpha=0.3, label='Drawdown')
ax2.plot(df.index, df['drawdown'] * 100, color='darkred', linewidth=1)
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([max_dd * 100 * 1.1, 5])  # Add some padding

plt.tight_layout()
plt.savefig('../plots/test2e_full_reversal_strategy.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/test2e_full_reversal_strategy.png")

# Print sample to verify logic
print("\n" + "="*80)
print("SAMPLE SIGNALS (First 80 trading days)")
print("="*80)
print(f"{'Date':<12} {'Day':>4} {'Last5?':>7} {'First5?':>8} {'Signal':>7} {'Ratio Ret':>10} {'Strat Ret':>10}")
print("-"*80)
for i in range(min(80, len(df))):
    row = df.iloc[i]
    sig_name = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}[row['signal']]
    print(f"{row.name.date()} {row['trading_day']:>4} "
          f"{'YES' if row['is_last_5'] else 'NO':>7} "
          f"{'YES' if row['is_first_5'] else 'NO':>8} "
          f"{sig_name:>7} "
          f"{row['ratio_return']*100:>9.3f}% {row['strategy_return']*100:>9.3f}%")
