"""
Simple Double Short Backtest with BIL Returns

Logic:
1. Start with $100k
2. Short $50k of bull ETF, short $50k of bear ETF
3. Now have $200k cash ($100k original + $100k from short proceeds)
4. Earn BIL (T-Bill) returns on the $200k cash
5. Rebalance weekly to maintain 50/50 short positions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

# Leveraged pairs
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

client = FMPClient()

def backtest_pair(bull: str, bear: str, from_date: str, to_date: str):
    """
    Backtest a single pair.

    Returns dict with results or None if data insufficient.
    """
    # Fetch data
    try:
        bull_data = client.get_historical_prices(bull, from_date=from_date, to_date=to_date)
        bear_data = client.get_historical_prices(bear, from_date=from_date, to_date=to_date)
        bil_data = client.get_historical_prices('BIL', from_date=from_date, to_date=to_date)

        if not bull_data or not bear_data or not bil_data:
            return None

        # Convert to DataFrames
        df_bull = pd.DataFrame(bull_data)[['date', 'adjClose']]
        df_bear = pd.DataFrame(bear_data)[['date', 'adjClose']]
        df_bil = pd.DataFrame(bil_data)[['date', 'adjClose']]

        df_bull['date'] = pd.to_datetime(df_bull['date'])
        df_bear['date'] = pd.to_datetime(df_bear['date'])
        df_bil['date'] = pd.to_datetime(df_bil['date'])

        df_bull = df_bull.set_index('date').sort_index()
        df_bear = df_bear.set_index('date').sort_index()
        df_bil = df_bil.set_index('date').sort_index()

        # Merge all three (inner join - only common dates)
        df = df_bull.join(df_bear, how='inner', lsuffix='_bull', rsuffix='_bear')
        df = df.join(df_bil, how='inner')
        df.columns = ['price_bull', 'price_bear', 'price_bil']

        if len(df) < 2:
            return None

    except Exception as e:
        print(f"Error fetching {bull}/{bear}: {e}")
        return None

    # Initial setup
    initial_capital = 100000

    # Day 0: Short $50k of each
    p0_bull = df.iloc[0]['price_bull']
    p0_bear = df.iloc[0]['price_bear']
    p0_bil = df.iloc[0]['price_bil']

    shares_bull_short = -50000 / p0_bull  # Negative = short
    shares_bear_short = -50000 / p0_bear  # Negative = short
    shares_bil = 200000 / p0_bil  # Positive = long (cash position)

    # Track entry prices for P&L calculation
    entry_bull = p0_bull
    entry_bear = p0_bear

    last_rebalance = df.index[0]
    rebalances = 0

    equity_curve = []

    for i, (date, row) in enumerate(df.iterrows()):
        p_bull = row['price_bull']
        p_bear = row['price_bear']
        p_bil = row['price_bil']

        # Calculate values
        value_bull_short = shares_bull_short * p_bull  # Negative
        value_bear_short = shares_bear_short * p_bear  # Negative
        value_bil = shares_bil * p_bil  # Positive

        # P&L on shorts: (entry - current) * shares, but shares are negative
        # So: entry_price * shares - current_price * shares
        pnl_bull = shares_bull_short * (entry_bull - p_bull)
        pnl_bear = shares_bear_short * (entry_bear - p_bear)

        # Equity = BIL value + P&L from shorts
        equity = value_bil + pnl_bull + pnl_bear

        equity_curve.append({'date': date, 'equity': equity})

        # Weekly rebalance
        if i > 0 and (date - last_rebalance).days >= 7:
            # Close all positions at current equity, reopen
            # Current equity becomes our new capital base

            # Short 50% of equity in each
            shares_bull_short = -(equity * 0.5) / p_bull
            shares_bear_short = -(equity * 0.5) / p_bear

            # All equity goes into BIL (equity + short proceeds)
            # Short proceeds = abs(50% equity) + abs(50% equity) = equity
            # Total BIL = equity (our capital) + equity (from shorts) = 2 * equity
            shares_bil = (2 * equity) / p_bil

            # Reset entry prices for new shorts
            entry_bull = p_bull
            entry_bear = p_bear

            last_rebalance = date
            rebalances += 1

    # Convert to DataFrame
    eq_df = pd.DataFrame(equity_curve).set_index('date')
    eq_df['returns'] = eq_df['equity'].pct_change()

    # Calculate metrics
    final_equity = eq_df['equity'].iloc[-1]
    total_return = (final_equity / initial_capital - 1)

    days = len(eq_df)
    years = days / 365.25
    cagr = (final_equity / initial_capital) ** (1/years) - 1 if years > 0 else 0

    volatility = eq_df['returns'].std() * np.sqrt(252)
    sharpe = (eq_df['returns'].mean() * 252) / volatility if volatility > 0 else 0

    # Max drawdown
    cumulative = (1 + eq_df['returns']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    win_rate = (eq_df['returns'] > 0).sum() / len(eq_df['returns'].dropna())

    return {
        'pair': f"{bull}/{bear}",
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'rebalances': rebalances,
        'days': days,
        'equity_curve': eq_df
    }


def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3 years

    print("="*80)
    print("SIMPLE DOUBLE SHORT BACKTEST")
    print("="*80)
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Strategy: Short $50k bull + $50k bear, hold $200k in BIL")
    print(f"Rebalancing: Weekly")
    print()

    results = []

    for i, (bull, bear) in enumerate(PAIRS):
        print(f"[{i+1}/{len(PAIRS)}] {bull}/{bear}...", end=" ")

        result = backtest_pair(
            bull, bear,
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d')
        )

        if result:
            results.append(result)
            print(f"Return: {result['total_return']*100:+.2f}% | Sharpe: {result['sharpe']:.2f}")
        else:
            print("SKIP")

    print()
    print("="*80)
    print(f"Successfully backtested {len(results)} pairs")
    print("="*80)
    print()

    # Create summary table
    summary = []
    for r in results:
        summary.append({
            'Pair': r['pair'],
            'Return': f"{r['total_return']*100:.2f}%",
            'CAGR': f"{r['cagr']*100:.2f}%",
            'Volatility': f"{r['volatility']*100:.2f}%",
            'Sharpe': f"{r['sharpe']:.2f}",
            'Max DD': f"{r['max_drawdown']*100:.2f}%",
            'Win Rate': f"{r['win_rate']*100:.1f}%",
            'Rebalances': r['rebalances']
        })

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    print()

    # Portfolio-level (equal weight)
    print("="*80)
    print("EQUAL-WEIGHT PORTFOLIO")
    print("="*80)

    # Merge all equity curves
    eq_curves = []
    for r in results:
        ec = r['equity_curve'][['equity']].copy()
        ec.columns = [r['pair']]
        eq_curves.append(ec)

    portfolio_df = eq_curves[0]
    for ec in eq_curves[1:]:
        portfolio_df = portfolio_df.join(ec, how='inner')

    portfolio_df['portfolio'] = portfolio_df.mean(axis=1)
    portfolio_df['returns'] = portfolio_df['portfolio'].pct_change()

    final = portfolio_df['portfolio'].iloc[-1]
    port_return = (final / 100000 - 1)
    port_years = len(portfolio_df) / 365.25
    port_cagr = (final / 100000) ** (1/port_years) - 1
    port_vol = portfolio_df['returns'].std() * np.sqrt(252)
    port_sharpe = (portfolio_df['returns'].mean() * 252) / port_vol if port_vol > 0 else 0

    cumulative = (1 + portfolio_df['returns']).cumprod()
    running_max = cumulative.expanding().max()
    dd = (cumulative - running_max) / running_max
    port_mdd = dd.min()

    port_wr = (portfolio_df['returns'] > 0).sum() / len(portfolio_df['returns'].dropna())

    print(f"Total Return: {port_return*100:.2f}%")
    print(f"CAGR: {port_cagr*100:.2f}%")
    print(f"Volatility: {port_vol*100:.2f}%")
    print(f"Sharpe: {port_sharpe:.2f}")
    print(f"Max Drawdown: {port_mdd*100:.2f}%")
    print(f"Win Rate: {port_wr*100:.2f}%")
    print()

    # Save results
    results_df = pd.DataFrame([{
        'pair': r['pair'],
        'total_return': r['total_return'],
        'cagr': r['cagr'],
        'volatility': r['volatility'],
        'sharpe': r['sharpe'],
        'max_drawdown': r['max_drawdown'],
        'win_rate': r['win_rate'],
        'rebalances': r['rebalances']
    } for r in results])

    results_df.to_csv('simple_backtest_results.csv', index=False)
    print(f"Results saved to: simple_backtest_results.csv")


if __name__ == '__main__':
    main()
