"""
TEST 2b: Dynamic Equal Weight Portfolio

Use all available data from all pairs.
On each day, equal weight across all pairs that have data available.
Number of pairs increases over time as new ETFs are created.
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

# All pairs (excluding LABU/LABD - too recent)
PAIRS = [
    ('TQQQ', 'SQQQ'),
    ('SOXL', 'SOXS'),
    ('SPXL', 'SPXS'),
    ('UPRO', 'SPXU'),
    ('TNA', 'TZA'),
    ('TECL', 'TECS'),
    ('QLD', 'QID'),
    ('SSO', 'SDS'),
    ('TMF', 'TMV'),
    ('NUGT', 'DUST'),
    ('YINN', 'YANG'),
    ('BOIL', 'KOLD'),
    ('UCO', 'SCO'),
]

client = FMPClient()

def load_borrow_costs(ticker):
    """Load borrow cost data from CSV"""
    filepath = f'../iborrowdesk_letf_rates/{ticker}.csv'
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, parse_dates=['time'])
    df = df.rename(columns={'time': 'date', 'fee': f'borrow_{ticker}'})
    df = df[['date', f'borrow_{ticker}']].set_index('date')
    return df

print("="*80)
print("TEST 2b: DYNAMIC EQUAL WEIGHT PORTFOLIO")
print("="*80)
print(f"Max pairs: {len(PAIRS)}")
print("Strategy: Equal weight across all available pairs on each day")
print()

# Load BIL
print("Loading BIL...")
bil_data = client.get_historical_prices('BIL', '2010-01-01', '2025-10-29')
df = pd.DataFrame(bil_data)
df['date'] = pd.to_datetime(df['date'])
df = df[['date', 'adjClose']].rename(columns={'adjClose': 'bil'})
df = df.set_index('date').sort_index()

# Load each pair
for bull, bear in PAIRS:
    print(f"Loading {bull}/{bear}...")

    # Bull
    bull_prices = pd.DataFrame(client.get_historical_prices(bull, '2010-01-01', '2025-10-29'))
    bull_prices['date'] = pd.to_datetime(bull_prices['date'])
    bull_prices = bull_prices[['date', 'adjClose']].rename(columns={'adjClose': bull})
    bull_prices = bull_prices.set_index('date')

    # Bear
    bear_prices = pd.DataFrame(client.get_historical_prices(bear, '2010-01-01', '2025-10-29'))
    bear_prices['date'] = pd.to_datetime(bear_prices['date'])
    bear_prices = bear_prices[['date', 'adjClose']].rename(columns={'adjClose': bear})
    bear_prices = bear_prices.set_index('date')

    # Borrow costs
    bull_borrow = load_borrow_costs(bull)
    bear_borrow = load_borrow_costs(bear)

    if bull_borrow is not None:
        bull_borrow = bull_borrow.rename(columns={f'borrow_{bull}': f'borrow_{bull}'})
    if bear_borrow is not None:
        bear_borrow = bear_borrow.rename(columns={f'borrow_{bear}': f'borrow_{bear}'})

    # Merge
    df = df.join(bull_prices, how='outer')
    df = df.join(bear_prices, how='outer')
    if bull_borrow is not None:
        df = df.join(bull_borrow, how='outer')
    if bear_borrow is not None:
        df = df.join(bear_borrow, how='outer')

df = df.sort_index()

# Only require BIL
df = df[df['bil'].notna()].copy()

print(f"\nDate range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Trading days: {len(df)}")

# Calculate returns
df['bil_ret'] = df['bil'].pct_change()

for bull, bear in PAIRS:
    df[f'{bull}_ret'] = df[bull].pct_change()
    df[f'{bear}_ret'] = df[bear].pct_change()

# Forward fill borrow costs
for bull, bear in PAIRS:
    if f'borrow_{bull}' in df.columns:
        df[f'borrow_{bull}'] = df[f'borrow_{bull}'].ffill()
    if f'borrow_{bear}' in df.columns:
        df[f'borrow_{bear}'] = df[f'borrow_{bear}'].ffill()

# Count available pairs on each day
df['n_pairs'] = 0

for bull, bear in PAIRS:
    # Pair is available if we have both prices, both returns, and both borrow costs
    pair_available = (
        df[bull].notna() &
        df[bear].notna() &
        df[f'{bull}_ret'].notna() &
        df[f'{bear}_ret'].notna()
    )

    # Also need borrow costs
    if f'borrow_{bull}' in df.columns and f'borrow_{bear}' in df.columns:
        pair_available = pair_available & df[f'borrow_{bull}'].notna() & df[f'borrow_{bear}'].notna()
    else:
        pair_available = pd.Series(False, index=df.index)

    df[f'avail_{bull}_{bear}'] = pair_available
    df['n_pairs'] += pair_available.astype(int)

# Keep only days with at least one pair
df = df[df['n_pairs'] > 0].copy()
df = df[df['bil_ret'].notna()].copy()

print(f"Days with at least 1 pair: {len(df)}")
print(f"Min pairs on any day: {df['n_pairs'].min()}")
print(f"Max pairs on any day: {df['n_pairs'].max()}")
print(f"Avg pairs per day: {df['n_pairs'].mean():.1f}")

# Calculate portfolio returns with dynamic weighting
df['cash_ret'] = 2.0 * (df['bil_ret'] - 0.01/252)
df['short_ret'] = 0.0
df['borrow_cost'] = 0.0

for bull, bear in PAIRS:
    # Weight per side on each day = 0.5 / n_pairs_that_day
    available = df[f'avail_{bull}_{bear}']

    if available.sum() == 0:
        continue

    weight_per_side = 0.5 / df.loc[available, 'n_pairs']

    # Short returns
    df.loc[available, 'short_ret'] += -weight_per_side * df.loc[available, f'{bull}_ret']
    df.loc[available, 'short_ret'] += -weight_per_side * df.loc[available, f'{bear}_ret']

    # Borrow costs
    if f'borrow_{bull}' in df.columns and f'borrow_{bear}' in df.columns:
        df.loc[available, 'borrow_cost'] += -weight_per_side * (df.loc[available, f'borrow_{bull}']/100/252)
        df.loc[available, 'borrow_cost'] += -weight_per_side * (df.loc[available, f'borrow_{bear}']/100/252)

# Total return
df['total_ret'] = df['cash_ret'] + df['short_ret'] + df['borrow_cost']
df['equity'] = (1 + df['total_ret']).cumprod() * 100000

# Metrics
final = df['equity'].iloc[-1]
total_return = (final / 100000) - 1
years = len(df) / 252
cagr = (final / 100000) ** (1/years) - 1
vol = df['total_ret'].std() * np.sqrt(252)
sharpe = (df['total_ret'].mean() * 252) / vol if vol > 0 else 0

cumulative = df['equity'] / 100000
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_dd = drawdown.min()
calmar = cagr / abs(max_dd) if max_dd != 0 else 0
win_rate = (df['total_ret'] > 0).mean()

print("\n" + "="*80)
print(f"RESULTS")
print("="*80)
print(f"Total Return:  {total_return:>10.2%}")
print(f"CAGR:          {cagr:>10.2%}")
print(f"Volatility:    {vol:>10.2%}")
print(f"Sharpe Ratio:  {sharpe:>10.2f}")
print(f"Max Drawdown:  {max_dd:>10.2%}")
print(f"Calmar Ratio:  {calmar:>10.2f}")
print(f"Win Rate:      {win_rate:>10.2%}")

# Save
results = pd.DataFrame([{
    'pairs_min': df['n_pairs'].min(),
    'pairs_max': df['n_pairs'].max(),
    'pairs_avg': df['n_pairs'].mean(),
    'total_return': total_return,
    'cagr': cagr,
    'volatility': vol,
    'sharpe': sharpe,
    'max_drawdown': max_dd,
    'calmar': calmar,
    'win_rate': win_rate,
    'years': years,
    'days': len(df),
    'start_date': df.index[0].strftime('%Y-%m-%d'),
    'end_date': df.index[-1].strftime('%Y-%m-%d'),
}])
results.to_csv('../data/test2b_dynamic_equal_weight_results.csv', index=False)
print(f"\nResults saved to: ../data/test2b_dynamic_equal_weight_results.csv")

# Plot
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# Equity curve
ax1.plot(df.index, df['equity'], linewidth=1.5, color='#2E86AB')
ax1.set_title(f'Dynamic Equal Weight Portfolio\nCAGR: {cagr:.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}',
              fontsize=12, pad=15)
ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Drawdown
ax2.fill_between(df.index, drawdown * 100, 0, color='#A23B72', alpha=0.6)
ax2.set_ylabel('Drawdown (%)', fontsize=10)
ax2.grid(True, alpha=0.3)

# Number of pairs over time
ax3.plot(df.index, df['n_pairs'], linewidth=1, color='#F18F01', drawstyle='steps-post')
ax3.set_ylabel('# of Pairs', fontsize=10)
ax3.set_xlabel('Date', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('../plots/test2b_dynamic_equal_weight_portfolio.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/test2b_dynamic_equal_weight_portfolio.png")
