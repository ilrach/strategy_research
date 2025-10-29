"""
TEST 2: Equal Weight Portfolio (Simplified)

Only include days where ALL pairs have data to keep it simple
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

# Pairs (excluding LABU/LABD and TECL/TECS due to limited data)
PAIRS = [
    ('TQQQ', 'SQQQ'),
    ('SOXL', 'SOXS'),
    ('SPXL', 'SPXS'),
    ('UPRO', 'SPXU'),
    ('TNA', 'TZA'),
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
    df = df.rename(columns={'time': 'date', 'fee': 'borrow'})
    df = df[['date', 'borrow']].set_index('date')
    return df

print("="*80)
print("TEST 2: EQUAL WEIGHT PORTFOLIO")
print("="*80)
print(f"Pairs: {len(PAIRS)}")
print()

# Load BIL
print("Loading BIL...")
bil_data = client.get_historical_prices('BIL', '2015-01-01', '2025-10-29')
df = pd.DataFrame(bil_data)
df['date'] = pd.to_datetime(df['date'])
df = df[['date', 'adjClose']].rename(columns={'adjClose': 'bil'})
df = df.set_index('date').sort_index()

# Load each pair
for bull, bear in PAIRS:
    print(f"Loading {bull}/{bear}...")

    # Bull
    bull_prices = pd.DataFrame(client.get_historical_prices(bull, '2015-01-01', '2025-10-29'))
    bull_prices['date'] = pd.to_datetime(bull_prices['date'])
    bull_prices = bull_prices[['date', 'adjClose']].rename(columns={'adjClose': bull})
    bull_prices = bull_prices.set_index('date')

    # Bear
    bear_prices = pd.DataFrame(client.get_historical_prices(bear, '2015-01-01', '2025-10-29'))
    bear_prices['date'] = pd.to_datetime(bear_prices['date'])
    bear_prices = bear_prices[['date', 'adjClose']].rename(columns={'adjClose': bear})
    bear_prices = bear_prices.set_index('date')

    # Borrow costs
    bull_borrow = load_borrow_costs(bull)
    bear_borrow = load_borrow_costs(bear)

    if bull_borrow is not None:
        bull_borrow = bull_borrow.rename(columns={'borrow': f'borrow_{bull}'})
    if bear_borrow is not None:
        bear_borrow = bear_borrow.rename(columns={'borrow': f'borrow_{bear}'})

    # Merge
    df = df.join(bull_prices)
    df = df.join(bear_prices)
    df = df.join(bull_borrow)
    df = df.join(bear_borrow)

print(f"\nBefore dropna: {len(df)} rows")
df = df.dropna()
print(f"After dropna: {len(df)} rows")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

# Calculate returns
for ticker in ['bil'] + [t for pair in PAIRS for t in pair]:
    df[f'{ticker}_ret'] = df[ticker].pct_change()

# Forward fill borrow costs
for bull, bear in PAIRS:
    df[f'borrow_{bull}'] = df[f'borrow_{bull}'].ffill()
    df[f'borrow_{bear}'] = df[f'borrow_{bear}'].ffill()

df = df.dropna()

# Calculate portfolio returns
n_pairs = len(PAIRS)
weight_per_side = 0.5 / n_pairs

print(f"\nAllocation: {n_pairs} pairs, {weight_per_side*100:.2f}% per side")

# Cash return
df['cash_ret'] = 2.0 * (df['bil_ret'] - 0.01/252)

# Short returns
df['short_ret'] = 0.0
for bull, bear in PAIRS:
    df['short_ret'] += -weight_per_side * (df[f'{bull}_ret'] + df[f'{bear}_ret'])

# Borrow costs
df['borrow_cost'] = 0.0
for bull, bear in PAIRS:
    df['borrow_cost'] += -weight_per_side * ((df[f'borrow_{bull}'] + df[f'borrow_{bear}'])/100/252)

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
    'pairs': n_pairs,
    'total_return': total_return,
    'cagr': cagr,
    'volatility': vol,
    'sharpe': sharpe,
    'max_drawdown': max_dd,
    'calmar': calmar,
    'win_rate': win_rate,
    'years': years,
    'days': len(df)
}])
results.to_csv('../data/test2_equal_weight_results.csv', index=False)
print(f"\nResults saved to: ../data/test2_equal_weight_results.csv")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(df.index, df['equity'], linewidth=1.5, color='#2E86AB')
ax1.set_title(f'Equal Weight Portfolio: {n_pairs} Pairs\nCAGR: {cagr:.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}',
              fontsize=12, pad=15)
ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, linewidth=1)

ax2.fill_between(df.index, drawdown * 100, 0, color='#A23B72', alpha=0.6)
ax2.set_ylabel('Drawdown (%)', fontsize=10)
ax2.set_xlabel('Date', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/test2_equal_weight_portfolio.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/test2_equal_weight_portfolio.png")
