"""
TEST 6: Top 5 Most Mean-Reverting Pairs

Based on trendiness analysis, test only the 5 most mean-reverting pairs:
1. YINN/YANG (trendiness: 0.301)
2. SOXL/SOXS (trendiness: 0.343)
3. BOIL/KOLD (trendiness: 0.364)
4. TMF/TMV (trendiness: 0.430)
5. NUGT/DUST (trendiness: 0.492)

Same as TEST 5 (no cash yield, 20% edge threshold, edge-proportional weighting)
but only using the most mean-reverting pairs.
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

# Top 5 most mean-reverting pairs (lowest trendiness scores)
PAIRS = [
    ('YINN', 'YANG', 3),  # 3x - trendiness: 0.301
    ('SOXL', 'SOXS', 3),  # 3x - trendiness: 0.343
    ('BOIL', 'KOLD', 2),  # 2x - trendiness: 0.364
    ('TMF', 'TMV', 3),    # 3x - trendiness: 0.430
    ('NUGT', 'DUST', 2),  # 2x - trendiness: 0.492
]

client = FMPClient()

EDGE_THRESHOLD = 20.0

print("="*80)
print("TEST 6: TOP 5 MOST MEAN-REVERTING PAIRS")
print("="*80)
print(f"Pairs: {len(PAIRS)} (YINN, SOXL, BOIL, TMF, NUGT)")
print(f"Edge threshold: {EDGE_THRESHOLD}%")
print(f"Weighting: Proportional to expected edge above threshold")
print(f"Cash yield: 0% (no BIL yield)")
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

    # Calculate returns
    pair_df[f'ret_{bull}'] = pair_df[f'p_{bull}'].pct_change()
    pair_df[f'ret_{bear}'] = pair_df[f'p_{bear}'].pct_change()

    # Calculate 30-day rolling volatility (annualized) of bull side
    pair_df[f'vol_{bull}'] = pair_df[f'ret_{bull}'].rolling(30).std() * np.sqrt(252)

    # Calculate average borrow cost between bull and bear
    pair_df['avg_borrow'] = (pair_df[f'b_{bull}'] + pair_df[f'b_{bear}']) / 2

    # Calculate expected edge
    k = k_values[leverage]
    pair_df['vol_pct'] = pair_df[f'vol_{bull}'] * 100
    pair_df['expected_edge'] = (pair_df['vol_pct'] / k) - pair_df['avg_borrow']

    # Set negative edges to zero
    pair_df['expected_edge'] = pair_df['expected_edge'].clip(lower=0)

    pair_df = pair_df.dropna()

    if len(pair_df) > 0:
        pair_data[(bull, bear, leverage)] = pair_df
        avg_edge = pair_df['expected_edge'].mean()
        days_positive_edge = (pair_df['expected_edge'] > 0).sum()
        pct_positive = (days_positive_edge / len(pair_df)) * 100
        print(f"  {len(pair_df)} days, avg edge: {avg_edge:.2f}%, positive: {days_positive_edge} days ({pct_positive:.1f}%)")

print(f"\n{len(pair_data)} pairs with complete data")

# Calculate BIL returns
bil_df['bil_ret'] = bil_df['bil'].pct_change()
bil_df = bil_df.dropna()

# Build portfolio returns day by day with edge-proportional weighting
portfolio_dates = []
portfolio_returns = []
pair_counts = []
total_edge_values = []

for date in bil_df.index:
    # Find which pairs have data and edge > threshold on this date
    available_pairs_with_edge = []
    edges = []

    for (bull, bear, leverage), pair_df in pair_data.items():
        if date in pair_df.index:
            edge = pair_df.loc[date, 'expected_edge']
            if edge > EDGE_THRESHOLD:
                available_pairs_with_edge.append((bull, bear, leverage))
                edges.append(edge)

    if len(available_pairs_with_edge) == 0:
        # No pairs with edge > threshold - hold cash only (earning 0%)
        cash_ret = 0.0
        portfolio_dates.append(date)
        portfolio_returns.append(cash_ret)
        pair_counts.append(0)
        total_edge_values.append(0)
        continue

    # Calculate weights proportional to edge
    total_edge = sum(edges)
    weights = [edge / total_edge for edge in edges]  # These sum to 1.0

    # Each pair gets -100% * weight total short (split -50% * weight on each side)
    # Calculate returns for this day
    cash_ret = 0.0  # Ablation: assume no yield on cash

    short_ret = 0.0
    borrow_cost = 0.0

    for i, (bull, bear, leverage) in enumerate(available_pairs_with_edge):
        pair_df = pair_data[(bull, bear, leverage)]
        weight = weights[i]
        weight_per_side = 0.5 * weight  # Each side gets half of the total pair weight

        short_ret += -weight_per_side * (pair_df.loc[date, f'ret_{bull}'] + pair_df.loc[date, f'ret_{bear}'])
        borrow_cost += -weight_per_side * ((pair_df.loc[date, f'b_{bull}'] + pair_df.loc[date, f'b_{bear}'])/100/252)

    total_ret = cash_ret + short_ret + borrow_cost

    portfolio_dates.append(date)
    portfolio_returns.append(total_ret)
    pair_counts.append(len(available_pairs_with_edge))
    total_edge_values.append(total_edge)

# Create results dataframe
results_df = pd.DataFrame({
    'date': portfolio_dates,
    'return': portfolio_returns,
    'n_pairs': pair_counts,
    'total_edge': total_edge_values
})
results_df = results_df.set_index('date')

print(f"\nPortfolio has {len(results_df)} trading days")
print(f"Date range: {results_df.index[0].date()} to {results_df.index[-1].date()}")
print(f"Avg pairs with positive edge: {results_df['n_pairs'].mean():.1f}")
print(f"Days with 0 pairs: {(results_df['n_pairs'] == 0).sum()} ({(results_df['n_pairs'] == 0).mean()*100:.1f}%)")
print(f"Avg total edge when active: {results_df[results_df['total_edge'] > 0]['total_edge'].mean():.2f}%")

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
    'pairs_avg': results_df['n_pairs'].mean(),
    'days_zero_pairs': (results_df['n_pairs'] == 0).sum(),
    'pct_zero_pairs': (results_df['n_pairs'] == 0).mean(),
    'avg_total_edge': results_df[results_df['total_edge'] > 0]['total_edge'].mean(),
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
summary.to_csv('../data/test6_top5_mean_reverting_results.csv', index=False)
print(f"\nResults saved to: ../data/test6_top5_mean_reverting_results.csv")

# Plot
fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])

# Equity curve
ax1.plot(results_df.index, results_df['equity'], linewidth=1.5, color='#2E86AB')
ax1.set_title(f'Edge-Proportional Weighting\nCAGR: {cagr:.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}',
              fontsize=12, pad=15)
ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Drawdown
ax2.fill_between(results_df.index, drawdown * 100, 0, color='#A23B72', alpha=0.6)
ax2.set_ylabel('Drawdown (%)', fontsize=10)
ax2.grid(True, alpha=0.3)

# Number of pairs
ax3.plot(results_df.index, results_df['n_pairs'], linewidth=1, color='#F18F01', drawstyle='steps-post')
ax3.set_ylabel('# of Pairs', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(bottom=0)

# Total edge over time
ax4.plot(results_df.index, results_df['total_edge'], linewidth=1, color='#06A77D', alpha=0.7)
ax4.set_ylabel('Total Edge (%)', fontsize=10)
ax4.set_xlabel('Date', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('../plots/test6_top5_mean_reverting_portfolio.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/test6_top5_mean_reverting_portfolio.png")
