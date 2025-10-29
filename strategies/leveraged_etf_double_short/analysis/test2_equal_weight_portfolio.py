"""
TEST 2: Equal Weight Portfolio of All Pairs

Strategy:
- Equal weight across all available pairs
- If N pairs, each pair gets -100%/N total short (split -50%/N bull, -50%/N bear)
- Total portfolio short = -100%
- 200% cash earning BIL - 100bps
- Apply actual borrow costs from iBorrowDesk
- Daily rebalancing (costless)

Metrics: CAGR, Sharpe, Max Drawdown, Calmar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

sns.set_style("darkgrid")

# Identify pairs from available borrow data
BORROW_DIR = '../iborrowdesk_letf_rates'

# Map tickers to their pairs (excluding LABU/LABD - insufficient data)
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
    filepath = f'{BORROW_DIR}/{ticker}.csv'
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, parse_dates=['time'])
    df = df.rename(columns={'time': 'date', 'fee': f'borrow_{ticker.lower()}'})
    df = df[['date', f'borrow_{ticker.lower()}']]
    df = df.set_index('date')
    return df


def main():
    start_date = '2015-01-01'
    end_date = '2025-10-29'

    print("="*80)
    print("TEST 2: EQUAL WEIGHT PORTFOLIO OF ALL PAIRS")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Strategy: Equal weight across {len(PAIRS)} pairs")
    print(f"Rebalance: Daily (costless)")
    print()

    # Build combined dataframe with all pairs
    print("Loading data for all pairs...")

    # Start with BIL
    print("Fetching BIL prices...")
    bil_data = client.get_historical_prices('BIL', start_date, end_date)
    bil_df = pd.DataFrame(bil_data)
    bil_df['date'] = pd.to_datetime(bil_df['date'])
    bil_df = bil_df[['date', 'adjClose']].rename(columns={'adjClose': 'p_bil'})
    bil_df = bil_df.set_index('date')

    df = bil_df.copy()

    # Load each pair
    pair_weights = []
    for bull, bear in PAIRS:
        print(f"\nLoading {bull}/{bear}...")

        # Fetch prices
        bull_data = client.get_historical_prices(bull, start_date, end_date)
        bear_data = client.get_historical_prices(bear, start_date, end_date)

        bull_df = pd.DataFrame(bull_data)
        bull_df['date'] = pd.to_datetime(bull_df['date'])
        bull_df = bull_df[['date', 'adjClose']].rename(columns={'adjClose': f'p_{bull.lower()}'})
        bull_df = bull_df.set_index('date')

        bear_df = pd.DataFrame(bear_data)
        bear_df['date'] = pd.to_datetime(bear_df['date'])
        bear_df = bear_df[['date', 'adjClose']].rename(columns={'adjClose': f'p_{bear.lower()}'})
        bear_df = bear_df.set_index('date')

        # Load borrow costs
        bull_borrow = load_borrow_costs(bull)
        bear_borrow = load_borrow_costs(bear)

        if bull_borrow is None or bear_borrow is None:
            print(f"  Skipping {bull}/{bear} - no borrow cost data")
            continue

        # Merge into main dataframe
        df = df.join(bull_df, how='outer')
        df = df.join(bear_df, how='outer')
        df = df.join(bull_borrow, how='outer')
        df = df.join(bear_borrow, how='outer')

        pair_weights.append((bull.lower(), bear.lower()))
        print(f"  Added {bull}/{bear}")

    df = df.sort_index()

    # Forward fill borrow costs
    for bull, bear in pair_weights:
        df[f'borrow_{bull}'] = df[f'borrow_{bull}'].ffill()
        df[f'borrow_{bear}'] = df[f'borrow_{bear}'].ffill()

    # Only require BIL data
    df = df.dropna(subset=['p_bil'])

    print(f"\n\nPortfolio composition: {len(pair_weights)} pairs (max)")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Trading days: {len(df)}")

    # Calculate returns for each instrument
    df['bil_ret'] = df['p_bil'].pct_change()

    for bull, bear in pair_weights:
        df[f'ret_{bull}'] = df[f'p_{bull}'].pct_change()
        df[f'ret_{bear}'] = df[f'p_{bear}'].pct_change()

    # Calculate portfolio return with dynamic weighting
    # On each day, count how many pairs have complete data
    # Each available pair gets equal weight that sums to -100% total

    df['n_pairs_available'] = 0
    for bull, bear in pair_weights:
        # Pair is available if we have prices and borrow costs for both sides
        pair_available = (
            df[f'p_{bull}'].notna() &
            df[f'p_{bear}'].notna() &
            df[f'borrow_{bull}'].notna() &
            df[f'borrow_{bear}'].notna() &
            df[f'ret_{bull}'].notna() &
            df[f'ret_{bear}'].notna()
        )
        df[f'available_{bull}_{bear}'] = pair_available.astype(int)
        df['n_pairs_available'] += pair_available.astype(int)

    # Drop days with no pairs available
    df = df[df['n_pairs_available'] > 0].copy()

    print(f"Days with data: {len(df)}")
    print(f"Avg pairs per day: {df['n_pairs_available'].mean():.1f}")

    # Cash return: 200% * (BIL - 100bps)
    df['cash_return'] = 2.0 * (df['bil_ret'] - 0.01/252)

    # Short returns and borrow costs: sum across available pairs with dynamic weights
    df['short_return'] = 0.0
    df['borrow_cost'] = 0.0

    for bull, bear in pair_weights:
        # Weight per side = 0.5 / n_pairs_available for that day
        weight_per_side = 0.5 / df['n_pairs_available']
        available = df[f'available_{bull}_{bear}'] == 1

        df.loc[available, 'short_return'] += -weight_per_side[available] * df.loc[available, f'ret_{bull}']
        df.loc[available, 'short_return'] += -weight_per_side[available] * df.loc[available, f'ret_{bear}']
        df.loc[available, 'borrow_cost'] += -weight_per_side[available] * (df.loc[available, f'borrow_{bull}']/100/252)
        df.loc[available, 'borrow_cost'] += -weight_per_side[available] * (df.loc[available, f'borrow_{bear}']/100/252)

    # Total return
    df['total_return'] = df['cash_return'] + df['short_return'] + df['borrow_cost']

    # Build equity curve
    df['equity'] = (1 + df['total_return']).cumprod() * 100000

    # Calculate metrics
    final_equity = df['equity'].iloc[-1]
    total_return = (final_equity / 100000) - 1

    years = len(df) / 252
    cagr = (final_equity / 100000) ** (1/years) - 1 if years > 0 else 0

    returns = df['total_return']
    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0

    # Max drawdown
    cumulative = df['equity'] / 100000
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns)

    # Print results
    max_pairs = len(pair_weights)
    avg_pairs = df['n_pairs_available'].mean()
    print("\n" + "="*80)
    print(f"RESULTS: EQUAL WEIGHT PORTFOLIO ({max_pairs} pairs max, {avg_pairs:.1f} avg)")
    print("="*80)
    print(f"Total Return:  {total_return:>10.2%}")
    print(f"CAGR:          {cagr:>10.2%}")
    print(f"Volatility:    {volatility:>10.2%}")
    print(f"Sharpe Ratio:  {sharpe:>10.2f}")
    print(f"Max Drawdown:  {max_dd:>10.2%}")
    print(f"Calmar Ratio:  {calmar:>10.2f}")
    print(f"Win Rate:      {win_rate:>10.2%}")

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Equity curve
    ax1.plot(df.index, df['equity'], linewidth=1.5, color='#2E86AB')
    ax1.set_title(f'Equal Weight Portfolio: {max_pairs} Pairs (avg {avg_pairs:.1f})\nCAGR: {cagr:.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}',
                  fontsize=12, pad=15)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Drawdown
    ax2.fill_between(df.index, drawdown * 100, 0, color='#A23B72', alpha=0.6)
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = '../plots/test2_equal_weight_portfolio.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")

    # Save results
    results = {
        'pairs_max': max_pairs,
        'pairs_avg': avg_pairs,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'years': years,
        'days': len(df),
        'start_date': df.index[0].strftime('%Y-%m-%d'),
        'end_date': df.index[-1].strftime('%Y-%m-%d'),
    }

    results_df = pd.DataFrame([results])
    results_path = '../data/test2_equal_weight_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Results saved: {results_path}")


if __name__ == '__main__':
    main()
