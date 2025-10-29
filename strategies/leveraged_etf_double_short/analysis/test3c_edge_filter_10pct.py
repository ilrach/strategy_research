"""
TEST 3c: Volatility-Filtered with 10% Edge Requirement

Only include pairs where volatility is high enough to overcome borrow costs PLUS 5% annual edge.

Filter criteria:
- Calculate 30-day historical volatility of long side (annualized)
- Identify leverage (2x or 3x)
- k = sqrt(0.01 / (leverage^2 / 2))
  - k ~= 7.07 for 2x leverage
  - k ~= 4.71 for 3x leverage
- Add 5% phantom borrow cost to required edge
- Include pair only if: (volatility / k) - (avg_borrow + 5.0) > 0

Equal weight across all pairs that pass the filter on each day.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

sns.set_style("darkgrid")

# Define pairs with their leverage
PAIRS = [
    ('TQQQ', 'SQQQ', 3),  # 3x
    ('SOXL', 'SOXS', 3),  # 3x
    ('SPXL', 'SPXS', 3),  # 3x
    ('UPRO', 'SPXU', 3),  # 3x
    ('TNA', 'TZA', 3),    # 3x
    ('TECL', 'TECS', 3),  # 3x
    ('QLD', 'QID', 2),    # 2x
    ('SSO', 'SDS', 2),    # 2x
    ('TMF', 'TMV', 3),    # 3x
    ('NUGT', 'DUST', 2),  # 2x
    ('YINN', 'YANG', 3),  # 3x
    ('BOIL', 'KOLD', 2),  # 2x
    ('UCO', 'SCO', 2),    # 2x
]

EDGE_REQUIREMENT = 10.0  # 5% additional edge requirement

client = FMPClient()

print("="*80)
print("TEST 3c: VOLATILITY-FILTERED WITH 5% EDGE REQUIREMENT")
print("="*80)
print(f"Max pairs: {len(PAIRS)}")
print(f"Edge requirement: {EDGE_REQUIREMENT}% additional hurdle")
print()

# Calculate k values for each leverage
k_values = {}
for leverage in [2, 3]:
    k = np.sqrt(100 / (leverage**2 / 2))
    k_values[leverage] = k
    print(f"k for {leverage}x leverage: {k:.2f} vol points per 1% borrow")

print()

# Load BIL
print("Loading BIL...")
bil_data = client.get_historical_prices('BIL', '2015-01-01', '2025-10-29')
bil_df = pd.DataFrame(bil_data)
bil_df['date'] = pd.to_datetime(bil_df['date'])
bil_df = bil_df[['date', 'adjClose']].rename(columns={'adjClose': 'bil'}).set_index('date').sort_index()

print(f"BIL: {len(bil_df)} rows from {bil_df.index[0].date()} to {bil_df.index[-1].date()}")

# Build dictionary of dataframes for each pair
pair_data = {}

for bull, bear, leverage in PAIRS:
    print(f"Loading {bull}/{bear} ({leverage}x)...")

    # Load prices
    bull_prices = pd.DataFrame(client.get_historical_prices(bull, '2015-01-01', '2025-10-29'))
    bull_prices['date'] = pd.to_datetime(bull_prices['date'])
    bull_prices = bull_prices[['date', 'adjClose']].rename(columns={'adjClose': f'p_{bull}'}).set_index('date')

    bear_prices = pd.DataFrame(client.get_historical_prices(bear, '2015-01-01', '2025-10-29'))
    bear_prices['date'] = pd.to_datetime(bear_prices['date'])
    bear_prices = bear_prices[['date', 'adjClose']].rename(columns={'adjClose': f'p_{bear}'}).set_index('date')

    # Load borrow costs
    bull_borrow_file = f'../iborrowdesk_letf_rates/{bull}.csv'
    bear_borrow_file = f'../iborrowdesk_letf_rates/{bear}.csv'

    if not os.path.exists(bull_borrow_file) or not os.path.exists(bear_borrow_file):
        print(f"  Skipping - missing borrow data")
        continue

    bull_borrow = pd.read_csv(bull_borrow_file, parse_dates=['time'])
    bull_borrow = bull_borrow.rename(columns={'time': 'date', 'fee': f'b_{bull}'})
    bull_borrow = bull_borrow[['date', f'b_{bull}']].set_index('date')

    bear_borrow = pd.read_csv(bear_borrow_file, parse_dates=['time'])
    bear_borrow = bear_borrow.rename(columns={'time': 'date', 'fee': f'b_{bear}'})
    bear_borrow = bear_borrow[['date', f'b_{bear}']].set_index('date')

    # Combine pair data
    pair_df = bull_prices.join(bear_prices, how='inner')
    pair_df = pair_df.join(bull_borrow, how='left')
    pair_df = pair_df.join(bear_borrow, how='left')

    # Remove duplicate dates
    pair_df = pair_df[~pair_df.index.duplicated(keep='first')]
    pair_df = pair_df.sort_index()

    # Forward fill borrow costs
    pair_df[f'b_{bull}'] = pair_df[f'b_{bull}'].ffill()
    pair_df[f'b_{bear}'] = pair_df[f'b_{bear}'].ffill()

    # Drop rows missing borrow costs
    pair_df = pair_df.dropna()

    # Calculate returns for bull (long side)
    pair_df[f'ret_{bull}'] = pair_df[f'p_{bull}'].pct_change()
    pair_df[f'ret_{bear}'] = pair_df[f'p_{bear}'].pct_change()

    # Calculate 30-day rolling volatility (annualized) of bull side
    pair_df[f'vol_{bull}'] = pair_df[f'ret_{bull}'].rolling(30).std() * np.sqrt(252)

    # Calculate average borrow cost between bull and bear
    pair_df['avg_borrow'] = (pair_df[f'b_{bull}'] + pair_df[f'b_{bear}']) / 2

    # Calculate threshold with 10% edge requirement
    k = k_values[leverage]
    # Must overcome: avg_borrow + EDGE_REQUIREMENT
    # Convert vol to percentage, filter: vol% / k > (avg_borrow% + edge%)
    pair_df['vol_pct'] = pair_df[f'vol_{bull}'] * 100
    pair_df['required_vol'] = k * (pair_df['avg_borrow'] + EDGE_REQUIREMENT)
    pair_df['threshold'] = pair_df['vol_pct'] - pair_df['required_vol']
    pair_df['passes_filter'] = pair_df['threshold'] > 0

    pair_df = pair_df.dropna()

    if len(pair_df) > 0:
        pair_data[(bull, bear, leverage)] = pair_df
        pct_pass = (pair_df['passes_filter'].sum() / len(pair_df)) * 100
        print(f"  {len(pair_df)} days from {pair_df.index[0].date()} to {pair_df.index[-1].date()}")
        print(f"  Passes filter: {pair_df['passes_filter'].sum()} days ({pct_pass:.1f}%)")

print(f"\n{len(pair_data)} pairs with complete data")

# Calculate BIL returns
bil_df['bil_ret'] = bil_df['bil'].pct_change()
bil_df = bil_df.dropna()

# Build portfolio returns day by day
portfolio_dates = []
portfolio_returns = []
pair_counts = []
pair_counts_filtered = []

for date in bil_df.index:
    # Find which pairs have data on this date
    available_pairs = []
    filtered_pairs = []

    for (bull, bear, leverage), pair_df in pair_data.items():
        if date in pair_df.index:
            available_pairs.append((bull, bear, leverage))
            if pair_df.loc[date, 'passes_filter']:
                filtered_pairs.append((bull, bear, leverage))

    if len(filtered_pairs) == 0:
        # No pairs pass filter - hold cash only
        cash_ret = 2.0 * (bil_df.loc[date, 'bil_ret'] - 0.01/252)
        portfolio_dates.append(date)
        portfolio_returns.append(cash_ret)
        pair_counts.append(len(available_pairs))
        pair_counts_filtered.append(0)
        continue

    n_pairs = len(filtered_pairs)
    weight_per_side = 0.5 / n_pairs

    # Calculate returns for this day
    cash_ret = 2.0 * (bil_df.loc[date, 'bil_ret'] - 0.01/252)

    short_ret = 0.0
    borrow_cost = 0.0

    for bull, bear, leverage in filtered_pairs:
        pair_df = pair_data[(bull, bear, leverage)]
        short_ret += -weight_per_side * (pair_df.loc[date, f'ret_{bull}'] + pair_df.loc[date, f'ret_{bear}'])
        borrow_cost += -weight_per_side * ((pair_df.loc[date, f'b_{bull}'] + pair_df.loc[date, f'b_{bear}'])/100/252)

    total_ret = cash_ret + short_ret + borrow_cost

    portfolio_dates.append(date)
    portfolio_returns.append(total_ret)
    pair_counts.append(len(available_pairs))
    pair_counts_filtered.append(n_pairs)

# Create results dataframe
results_df = pd.DataFrame({
    'date': portfolio_dates,
    'return': portfolio_returns,
    'n_pairs_available': pair_counts,
    'n_pairs_filtered': pair_counts_filtered
})
results_df = results_df.set_index('date')

print(f"\nPortfolio has {len(results_df)} trading days")
print(f"Date range: {results_df.index[0].date()} to {results_df.index[-1].date()}")
print(f"Avg pairs available: {results_df['n_pairs_available'].mean():.1f}")
print(f"Avg pairs after filter: {results_df['n_pairs_filtered'].mean():.1f}")
print(f"Days with 0 pairs: {(results_df['n_pairs_filtered'] == 0).sum()} ({(results_df['n_pairs_filtered'] == 0).mean()*100:.1f}%)")

# Calculate metrics
results_df['equity'] = (1 + results_df['return']).cumprod() * 100000

final = results_df['equity'].iloc[-1]
total_return = (final / 100000) - 1
years = len(results_df) / 252
cagr = (final / 100000) ** (1/years) - 1
vol = results_df['return'].std() * np.sqrt(252)
sharpe = (results_df['return'].mean() * 252) / vol if vol > 0 else 0

cumulative = results_df['equity'] / 100000
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_dd = drawdown.min()
calmar = cagr / abs(max_dd) if max_dd != 0 else 0
win_rate = (results_df['return'] > 0).mean()

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Total Return:  {total_return:>10.2%}")
print(f"CAGR:          {cagr:>10.2%}")
print(f"Volatility:    {vol:>10.2%}")
print(f"Sharpe Ratio:  {sharpe:>10.2f}")
print(f"Max Drawdown:  {max_dd:>10.2%}")
print(f"Calmar Ratio:  {calmar:>10.2f}")
print(f"Win Rate:      {win_rate:>10.2%}")

# Save summary
summary = pd.DataFrame([{
    'edge_requirement': EDGE_REQUIREMENT,
    'pairs_available_avg': results_df['n_pairs_available'].mean(),
    'pairs_filtered_avg': results_df['n_pairs_filtered'].mean(),
    'days_zero_pairs': (results_df['n_pairs_filtered'] == 0).sum(),
    'pct_zero_pairs': (results_df['n_pairs_filtered'] == 0).mean(),
    'total_return': total_return,
    'cagr': cagr,
    'volatility': vol,
    'sharpe': sharpe,
    'max_drawdown': max_dd,
    'calmar': calmar,
    'win_rate': win_rate,
    'years': years,
    'days': len(results_df),
    'start_date': results_df.index[0].strftime('%Y-%m-%d'),
    'end_date': results_df.index[-1].strftime('%Y-%m-%d'),
}])
summary.to_csv('../data/test3c_edge_filter_10pct_results.csv', index=False)
print(f"\nResults saved to: ../data/test3c_edge_filter_10pct_results.csv")

# Plot
fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])

# Equity curve
ax1.plot(results_df.index, results_df['equity'], linewidth=1.5, color='#2E86AB')
ax1.set_title(f'Volatility-Filtered Portfolio (10% Edge Required)\nCAGR: {cagr:.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}',
              fontsize=12, pad=15)
ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Drawdown
ax2.fill_between(results_df.index, drawdown * 100, 0, color='#A23B72', alpha=0.6)
ax2.set_ylabel('Drawdown (%)', fontsize=10)
ax2.grid(True, alpha=0.3)

# Number of pairs available
ax3.plot(results_df.index, results_df['n_pairs_available'], linewidth=1, color='#C0C0C0',
         drawstyle='steps-post', label='Available', alpha=0.5)
ax3.plot(results_df.index, results_df['n_pairs_filtered'], linewidth=1.5, color='#F18F01',
         drawstyle='steps-post', label='After Filter')
ax3.set_ylabel('# of Pairs', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(bottom=0)
ax3.legend(loc='upper left', fontsize=9)

# Filter ratio (what % of available pairs pass filter)
filter_ratio = (results_df['n_pairs_filtered'] / results_df['n_pairs_available'].replace(0, np.nan)) * 100
ax4.fill_between(results_df.index, filter_ratio, 0, color='#06A77D', alpha=0.6)
ax4.set_ylabel('Filter Pass %', fontsize=10)
ax4.set_xlabel('Date', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('../plots/test3c_edge_filter_10pct_portfolio.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/test3c_edge_filter_10pct_portfolio.png")
