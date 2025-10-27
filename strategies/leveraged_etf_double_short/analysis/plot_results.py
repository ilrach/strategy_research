"""
Generate clean plots for the ultra-simple backtest results
All in return space (%)
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

# Recreate the backtest to get equity curves
PAIRS = [
    ('TQQQ', 'SQQQ'),
    ('SOXL', 'SOXS'),
    ('SPXL', 'SPXS'),
    ('UPRO', 'SPXU'),
    ('TNA', 'TZA'),
    ('TECL', 'TECS'),
    ('UDOW', 'SDOW'),
    ('LABU', 'LABD'),
    ('NUGT', 'DUST'),
    ('YINN', 'YANG'),
    ('FAS', 'FAZ'),
    ('GUSH', 'DRIP'),
    ('UCO', 'SCO'),
    ('BOIL', 'KOLD'),
    ('QLD', 'QID'),
    ('SSO', 'SDS'),
    ('UWM', 'TWM'),
    ('DDM', 'DXD'),
    ('TMF', 'TMV'),
    ('UBT', 'TBT'),
    ('DRN', 'DRV'),
    ('ERX', 'ERY'),
    ('USD', 'SSG'),
    ('WEBL', 'WEBS'),
    ('AGQ', 'ZSL'),
    ('UGL', 'GLL'),
]

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

client = FMPClient()

def get_equity_curve(bull, bear, from_date, to_date):
    """Get equity curve for a pair."""
    try:
        bull_data = client.get_historical_prices(bull, from_date=from_date, to_date=to_date)
        bear_data = client.get_historical_prices(bear, from_date=from_date, to_date=to_date)
        bil_data = client.get_historical_prices('BIL', from_date=from_date, to_date=to_date)

        if not bull_data or not bear_data or not bil_data:
            return None

        df_bull = pd.DataFrame(bull_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_bull'})
        df_bear = pd.DataFrame(bear_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_bear'})
        df_bil = pd.DataFrame(bil_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_bil'})

        df_bull['date'] = pd.to_datetime(df_bull['date'])
        df_bear['date'] = pd.to_datetime(df_bear['date'])
        df_bil['date'] = pd.to_datetime(df_bil['date'])

        df = df_bull.set_index('date').join(df_bear.set_index('date')).join(df_bil.set_index('date'))
        df = df.sort_index()

        if len(df) < 2:
            return None

        # Calculate returns
        df['bil_ret'] = df['p_bil'].pct_change()
        df['bull_ret'] = df['p_bull'].pct_change()
        df['bear_ret'] = df['p_bear'].pct_change()

        df['total_return'] = 2 * df['bil_ret'] - 0.5 * df['bull_ret'] - 0.5 * df['bear_ret']

        # Cumulative return (%)
        df['cumulative_return_pct'] = ((1 + df['total_return']).cumprod() - 1) * 100

        return df[['cumulative_return_pct']]

    except Exception as e:
        print(f"Error with {bull}/{bear}: {e}")
        return None


print("Fetching data and generating plots...")

end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)

from_date = start_date.strftime('%Y-%m-%d')
to_date = end_date.strftime('%Y-%m-%d')

# Collect all equity curves
equity_curves = {}
for bull, bear in PAIRS:
    print(f"  {bull}/{bear}...")
    ec = get_equity_curve(bull, bear, from_date, to_date)
    if ec is not None:
        equity_curves[f"{bull}/{bear}"] = ec

# Create portfolio
port_df = pd.DataFrame({k: v['cumulative_return_pct'] for k, v in equity_curves.items()})
port_df['Portfolio'] = port_df.mean(axis=1)

print(f"\nGenerating plots...")

# 1. Portfolio equity curve
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(port_df.index, port_df['Portfolio'], linewidth=2, color='#2E86AB', label='Equal-Weight Portfolio')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_title('Double Short Leveraged ETF Strategy - Portfolio Returns', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Return (%)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../plots/leveraged_etf/portfolio_returns.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: portfolio_returns.png")

# 2. Individual pairs (grid)
n_pairs = len(equity_curves)
n_cols = 4
n_rows = (n_pairs + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten() if n_pairs > 1 else [axes]

for i, (pair, ec) in enumerate(equity_curves.items()):
    ax = axes[i]
    ax.plot(ec.index, ec['cumulative_return_pct'], linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    final_return = ec['cumulative_return_pct'].iloc[-1]
    ax.set_title(f"{pair}\nReturn: {final_return:.1f}%", fontsize=10)
    ax.set_ylabel('Return (%)', fontsize=9)
    ax.grid(True, alpha=0.3)

# Hide empty subplots
for i in range(n_pairs, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('../../plots/leveraged_etf/individual_pairs_returns.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: individual_pairs_returns.png")

# 3. Performance comparison
results_df = pd.read_csv('ultra_simple_results.csv')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Total returns
sorted_by_return = results_df.sort_values('total_return', ascending=True)
axes[0, 0].barh(sorted_by_return['pair'], sorted_by_return['total_return'] * 100, color='#06A77D')
axes[0, 0].set_title('Total Return by Pair (%)', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Total Return (%)')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Sharpe ratios
sorted_by_sharpe = results_df.sort_values('sharpe', ascending=True)
axes[0, 1].barh(sorted_by_sharpe['pair'], sorted_by_sharpe['sharpe'], color='#F18F01')
axes[0, 1].set_title('Sharpe Ratio by Pair', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Sharpe Ratio')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Volatility
sorted_by_vol = results_df.sort_values('volatility', ascending=True)
axes[1, 0].barh(sorted_by_vol['pair'], sorted_by_vol['volatility'] * 100, color='#C73E1D')
axes[1, 0].set_title('Volatility by Pair (%)', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Annualized Volatility (%)')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Max drawdown
sorted_by_dd = results_df.sort_values('max_drawdown', ascending=True)
axes[1, 1].barh(sorted_by_dd['pair'], sorted_by_dd['max_drawdown'] * 100, color='#6A4C93')
axes[1, 1].set_title('Maximum Drawdown by Pair (%)', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Max Drawdown (%)')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../../plots/leveraged_etf/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: performance_comparison.png")

# 4. Drawdown chart for portfolio
port_returns = port_df['Portfolio'].pct_change()
cumulative = (1 + port_returns / 100).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max * 100

fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
ax.plot(drawdown.index, drawdown, color='red', linewidth=1)
ax.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Drawdown (%)', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../plots/leveraged_etf/portfolio_drawdown.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: portfolio_drawdown.png")

print("\nAll plots saved to: ../../plots/leveraged_etf/")
