"""
TEST 2f: Long-Only Full Reversal Strategy

Trading Strategy:
- Calculate SPY/TLT ratio return from day 1 to day before last 5 (signal period)
- LAST 5 DAYS of current month:
  * If SPY outperforms TLT: LONG TLT (the underperformer)
  * If TLT outperforms SPY: LONG SPY (the underperformer)
- FIRST 5 DAYS of next month (INVERTED - go with the previous winner):
  * If SPY outperformed TLT last month: LONG SPY
  * If TLT outperformed SPY last month: LONG TLT
- FLAT all other times

This is a long-only version of test2e, capturing both moves without shorting.
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
print("TEST 2f: LONG-ONLY FULL REVERSAL STRATEGY")
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

# For each month, determine positions
df['position'] = 'FLAT'  # Can be 'FLAT', 'LONG_SPY', or 'LONG_TLT'

periods = sorted(df['year_month'].unique())

for i, period in enumerate(periods):
    group = df[df['year_month'] == period]

    # Get the days before last 5
    first_days = group[~group['is_last_5']]

    if len(first_days) > 0:
        # Cumulative ratio return in first days
        cum_ratio_return = first_days['ratio_return'].sum()

        # LAST 5 DAYS: Long the underperformer
        # If SPY outperformed (positive ratio), go LONG TLT
        # If TLT outperformed (negative ratio), go LONG SPY
        last_5_idx = group[group['is_last_5']].index
        if len(last_5_idx) > 0:
            if cum_ratio_return > 0:
                df.loc[last_5_idx, 'position'] = 'LONG_TLT'
            else:
                df.loc[last_5_idx, 'position'] = 'LONG_SPY'

            # FIRST 5 DAYS OF NEXT MONTH: Long the previous winner (inverted)
            # If SPY outperformed last month, go LONG SPY
            # If TLT outperformed last month, go LONG TLT
            if i + 1 < len(periods):
                next_period = periods[i + 1]
                next_group = df[df['year_month'] == next_period]
                first_5_next_idx = next_group[next_group['is_first_5']].index
                if len(first_5_next_idx) > 0:
                    if cum_ratio_return > 0:
                        # SPY was the winner, go long it
                        df.loc[first_5_next_idx, 'position'] = 'LONG_SPY'
                    else:
                        # TLT was the winner, go long it
                        df.loc[first_5_next_idx, 'position'] = 'LONG_TLT'

# Calculate strategy returns based on position
df['strategy_return'] = 0.0
df.loc[df['position'] == 'LONG_SPY', 'strategy_return'] = df.loc[df['position'] == 'LONG_SPY', 'spy_return']
df.loc[df['position'] == 'LONG_TLT', 'strategy_return'] = df.loc[df['position'] == 'LONG_TLT', 'tlt_return']

# Calculate cumulative returns
df['cum_return'] = (1 + df['strategy_return']).cumprod()

# For comparison: 60/40 SPY/TLT portfolio
df['portfolio_return'] = 0.6 * df['spy_return'] + 0.4 * df['tlt_return']
df['cum_portfolio_return'] = (1 + df['portfolio_return']).cumprod()

# Calculate drawdown
df['running_max'] = df['cum_return'].cummax()
df['drawdown'] = (df['cum_return'] - df['running_max']) / df['running_max']

print("Strategy Summary:")
print("-" * 80)
print(f"Position breakdown:")
position_counts = df['position'].value_counts()
for pos, count in position_counts.items():
    pct = count / len(df) * 100
    print(f"  {pos:>10}: {count:>5} days ({pct:>5.1f}%)")
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

# 60/40 portfolio comparison
portfolio_final = df['cum_portfolio_return'].iloc[-1]
portfolio_cagr = (portfolio_final ** (1 / years)) - 1
portfolio_returns = df['portfolio_return']
portfolio_sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
portfolio_running_max = df['cum_portfolio_return'].cummax()
portfolio_dd = ((df['cum_portfolio_return'] - portfolio_running_max) / portfolio_running_max).min()
portfolio_calmar = portfolio_cagr / abs(portfolio_dd) if portfolio_dd != 0 else 0

print("="*80)
print("PERFORMANCE METRICS")
print("="*80)
print(f"\n{'Metric':<25} {'Strategy':>15} {'60/40 Portfolio':>15}")
print("-"*80)
print(f"{'Total Return':<25} {(final_value - 1):>14.2%} {(portfolio_final - 1):>14.2%}")
print(f"{'CAGR':<25} {cagr:>14.2%} {portfolio_cagr:>14.2%}")
print(f"{'Sharpe Ratio':<25} {sharpe:>14.2f} {portfolio_sharpe:>14.2f}")
print(f"{'Max Drawdown':<25} {max_dd:>14.2%} {portfolio_dd:>14.2%}")
print(f"{'Calmar Ratio':<25} {calmar:>14.2f} {portfolio_calmar:>14.2f}")
print(f"{'Annual Volatility':<25} {std_return * np.sqrt(252):>14.2%} {portfolio_returns.std() * np.sqrt(252):>14.2%}")
print(f"{'Total Days':<25} {total_days:>15} {total_days:>15}")
print(f"{'Years':<25} {years:>14.1f} {years:>14.1f}")

# Save results
results = pd.DataFrame({
    'metric': ['Total Return', 'CAGR', 'Sharpe', 'Max DD', 'Calmar', 'Volatility'],
    'strategy': [(final_value - 1), cagr, sharpe, max_dd, calmar, std_return * np.sqrt(252)],
    'portfolio_60_40': [(portfolio_final - 1), portfolio_cagr, portfolio_sharpe, portfolio_dd, portfolio_calmar, portfolio_returns.std() * np.sqrt(252)]
})
results.to_csv('../data/test2f_performance.csv', index=False)
print(f"\nResults saved to: ../data/test2f_performance.csv")

# Save daily data
df[['spy_return', 'tlt_return', 'ratio_return', 'trading_day', 'is_last_5', 'is_first_5',
    'position', 'strategy_return', 'cum_return', 'drawdown']].to_csv('../data/test2f_daily_data.csv')
print(f"Daily data saved to: ../data/test2f_daily_data.csv")

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

# Equity curve
ax1 = axes[0]
ax1.plot(df.index, df['cum_return'], label='Strategy', linewidth=2, color='darkblue')
ax1.plot(df.index, df['cum_portfolio_return'], label='60/40 SPY/TLT Portfolio',
         linewidth=1.5, color='gray', alpha=0.6, linestyle='--')
ax1.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
ax1.set_title(f'Long-Only Full Reversal Strategy (Last 5 + Inverted First 5)\n'
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
plt.savefig('../plots/test2f_long_only_full_reversal.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/test2f_long_only_full_reversal.png")

# Print sample to verify logic
print("\n" + "="*80)
print("SAMPLE POSITIONS (First 80 trading days)")
print("="*80)
print(f"{'Date':<12} {'Day':>4} {'Last5?':>7} {'First5?':>8} {'Position':>10} {'SPY Ret':>9} {'TLT Ret':>9} {'Strat Ret':>10}")
print("-"*80)
for i in range(min(80, len(df))):
    row = df.iloc[i]
    print(f"{row.name.date()} {row['trading_day']:>4} "
          f"{'YES' if row['is_last_5'] else 'NO':>7} "
          f"{'YES' if row['is_first_5'] else 'NO':>8} "
          f"{row['position']:>10} "
          f"{row['spy_return']*100:>8.2f}% {row['tlt_return']*100:>8.2f}% {row['strategy_return']*100:>9.2f}%")
