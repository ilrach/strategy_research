"""
TEST 1: Individual Pair Backtests

Strategy:
- Short -50% bull, -50% bear (total -100% short)
- 200% cash earning BIL - 100bps
- Apply actual borrow costs from iBorrowDesk
- Daily rebalancing (costless)

Metrics: CAGR, Sharpe, Max Drawdown, Calmar
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

# Identify pairs from available borrow data
BORROW_DIR = '../iborrowdesk_letf_rates'

# Map tickers to their pairs
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
    ('LABU', 'LABD'),
    ('NUGT', 'DUST'),
    ('YINN', 'YANG'),
    ('BOIL', 'KOLD'),
    ('UCO', 'SCO'),
]

client = FMPClient()


def load_borrow_costs(ticker):
    """Load borrow cost data from CSV"""
    filepath = os.path.join(BORROW_DIR, f'{ticker}.csv')

    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['time'])
    df = df[['date', 'fee']].sort_values('date')
    df = df.set_index('date')

    # Handle duplicate dates by taking mean
    df = df.groupby(df.index)['fee'].mean()

    return df


def backtest_pair(bull, bear, start_date, end_date):
    """
    Backtest single pair with:
    - -50% short bull, -50% short bear
    - 200% cash earning BIL - 100bps
    - Borrow costs from iBorrowDesk
    - Weekly rebalancing
    """
    print(f"\n{'='*80}")
    print(f"Backtesting {bull}/{bear}")
    print(f"{'='*80}")

    # Fetch price data
    print("Fetching price data...")
    bull_data = client.get_historical_prices(bull, from_date=start_date, to_date=end_date)
    bear_data = client.get_historical_prices(bear, from_date=start_date, to_date=end_date)
    bil_data = client.get_historical_prices('BIL', from_date=start_date, to_date=end_date)

    if not bull_data or not bear_data or not bil_data:
        print(f"ERROR: Missing price data")
        return None

    # Create dataframes
    df_bull = pd.DataFrame(bull_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_bull'})
    df_bear = pd.DataFrame(bear_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_bear'})
    df_bil = pd.DataFrame(bil_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_bil'})

    for df in [df_bull, df_bear, df_bil]:
        df['date'] = pd.to_datetime(df['date'])

    # Merge
    df = df_bull.set_index('date').join(df_bear.set_index('date')).join(df_bil.set_index('date'))
    df = df.sort_index()

    # Load borrow costs
    print("Loading borrow costs...")
    borrow_bull = load_borrow_costs(bull)
    borrow_bear = load_borrow_costs(bear)

    if borrow_bull is None or borrow_bear is None:
        print(f"ERROR: Missing borrow cost data")
        return None

    # Merge borrow costs (forward fill for missing dates)
    df = df.join(borrow_bull.rename('borrow_bull'))
    df = df.join(borrow_bear.rename('borrow_bear'))
    df['borrow_bull'] = df['borrow_bull'].fillna(method='ffill')
    df['borrow_bear'] = df['borrow_bear'].fillna(method='ffill')

    # Drop any rows still missing data
    df = df.dropna()

    if len(df) < 2:
        print(f"ERROR: Insufficient data after merge")
        return None

    print(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Trading days: {len(df)}")
    print(f"Avg borrow: {bull} {df['borrow_bull'].mean():.2f}%, {bear} {df['borrow_bear'].mean():.2f}%")

    # Calculate daily returns
    df['bull_ret'] = df['p_bull'].pct_change()
    df['bear_ret'] = df['p_bear'].pct_change()
    df['bil_ret'] = df['p_bil'].pct_change()

    # Drop NaN rows
    df = df.dropna(subset=['bull_ret', 'bear_ret', 'bil_ret'])

    # Daily returns with costless rebalancing (maintain constant -50%/-50% weights)
    # Strategy returns:
    # Cash: 200% * (BIL - 1%)
    # Shorts: -50% * bull_ret - 50% * bear_ret
    # Borrow: -50% * (borrow_bull/252/100) - 50% * (borrow_bear/252/100)

    df['cash_return'] = 2.0 * (df['bil_ret'] - 0.01/252)  # 200% cash, minus 100bps annually
    df['short_return'] = -0.5 * df['bull_ret'] - 0.5 * df['bear_ret']
    df['borrow_cost'] = -0.5 * (df['borrow_bull']/100/252) - 0.5 * (df['borrow_bear']/100/252)

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
    print(f"\n{'='*80}")
    print(f"RESULTS: {bull}/{bear}")
    print(f"{'='*80}")
    print(f"Total Return:    {total_return*100:>8.2f}%")
    print(f"CAGR:            {cagr*100:>8.2f}%")
    print(f"Volatility:      {volatility*100:>8.2f}%")
    print(f"Sharpe Ratio:    {sharpe:>8.2f}")
    print(f"Max Drawdown:    {max_dd*100:>8.2f}%")
    print(f"Calmar Ratio:    {calmar:>8.2f}")
    print(f"Win Rate:        {win_rate*100:>8.2f}%")

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Equity curve
    ax1.plot(df.index, (df['equity']/100000 - 1) * 100, linewidth=2, color='#2E86AB')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax1.set_title(f'{bull}/{bear} Double Short Strategy\nCAGR: {cagr*100:.2f}% | Sharpe: {sharpe:.2f} | Max DD: {max_dd*100:.2f}%',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend([f'{bull}/{bear}'], loc='upper left')

    # Drawdown
    ax2.fill_between(df.index, 0, drawdown * 100, alpha=0.5, color='#C73E1D')
    ax2.plot(df.index, drawdown * 100, linewidth=1.5, color='#C73E1D')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_filename = f'../plots/test1_{bull}_{bear}_backtest.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {plot_filename}")

    return {
        'pair': f'{bull}/{bear}',
        'bull': bull,
        'bear': bear,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'years': years,
        'days': len(df),
        'start_date': df.index[0],
        'end_date': df.index[-1],
        'avg_borrow_bull': df['borrow_bull'].mean(),
        'avg_borrow_bear': df['borrow_bear'].mean(),
    }


def main():
    # Backtest period: longest available
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2015-01-01'  # Start from 2015 to capture max history

    print("="*80)
    print("TEST 1: INDIVIDUAL PAIR BACKTESTS")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Strategy: -50% bull / -50% bear, 200% cash @ BIL-100bps")
    print(f"Rebalance: Daily (costless)")
    print()

    results = []

    for bull, bear in PAIRS:
        result = backtest_pair(bull, bear, start_date, end_date)
        if result:
            results.append(result)

    if not results:
        print("\nNo successful backtests!")
        return

    # Create summary table
    print(f"\n{'='*80}")
    print("SUMMARY: ALL PAIRS")
    print(f"{'='*80}\n")

    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values('sharpe', ascending=False)

    print(summary_df[['pair', 'cagr', 'volatility', 'sharpe', 'max_drawdown', 'calmar', 'win_rate']].to_string(index=False))

    # Save results
    summary_df.to_csv('../data/test1_individual_pairs_results.csv', index=False)
    print(f"\n\nResults saved to: ../data/test1_individual_pairs_results.csv")


if __name__ == '__main__':
    main()
