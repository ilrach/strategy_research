"""
Ultra-simple backtest: Just give portfolio 2x BIL daily returns

Logic:
- We have $100k
- We short $50k of each ETF, which gives us $200k total cash
- That $200k earns BIL returns
- So each day: portfolio_return = 2 * BIL_daily_return
- Rebalance weekly to maintain 50/50 shorts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

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
    """Backtest a pair."""
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

        # Calculate daily returns
        df['bil_ret'] = df['p_bil'].pct_change()
        df['bull_ret'] = df['p_bull'].pct_change()
        df['bear_ret'] = df['p_bear'].pct_change()

        # Portfolio return each day:
        # = 2x BIL return (cash on $200k)
        # + short bull return (= -1 * bull return, on 50% of portfolio)
        # + short bear return (= -1 * bear return, on 50% of portfolio)
        df['cash_ret'] = 2 * df['bil_ret']
        df['short_bull_ret'] = -0.5 * df['bull_ret']  # Short 50% in bull
        df['short_bear_ret'] = -0.5 * df['bear_ret']  # Short 50% in bear

        df['total_return'] = df['cash_ret'] + df['short_bull_ret'] + df['short_bear_ret']

        # Build equity curve
        equity = 100000
        equity_curve = [equity]

        for i in range(1, len(df)):
            daily_return = df.iloc[i]['total_return']
            equity = equity * (1 + daily_return)
            equity_curve.append(equity)

        df['equity'] = equity_curve

        # Calculate metrics
        final_equity = df['equity'].iloc[-1]
        total_return = (final_equity / 100000 - 1)

        years = len(df) / 365.25
        cagr = (final_equity / 100000) ** (1/years) - 1 if years > 0 else 0

        returns = df['equity'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        win_rate = (returns > 0).sum() / len(returns)

        return {
            'pair': f"{bull}/{bear}",
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'days': len(df),
            'equity_curve': df[['equity']]
        }

    except Exception as e:
        print(f"Error with {bull}/{bear}: {e}")
        return None


def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)

    print("="*80)
    print("ULTRA-SIMPLE BACKTEST: 2x BIL + Double Short Returns")
    print("="*80)
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Strategy: Daily return = 2×BIL - 0.5×Bull - 0.5×Bear")
    print()

    results = []

    for i, (bull, bear) in enumerate(PAIRS):
        print(f"[{i+1}/{len(PAIRS)}] {bull}/{bear}...", end=" ")

        result = backtest_pair(bull, bear, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if result:
            results.append(result)
            print(f"Return: {result['total_return']*100:+.2f}% | Sharpe: {result['sharpe']:.2f}")
        else:
            print("SKIP")

    if not results:
        print("No results!")
        return

    print()
    print("="*80)
    print("RESULTS")
    print("="*80)

    summary = pd.DataFrame([{
        'Pair': r['pair'],
        'Return': f"{r['total_return']*100:.2f}%",
        'CAGR': f"{r['cagr']*100:.2f}%",
        'Volatility': f"{r['volatility']*100:.2f}%",
        'Sharpe': f"{r['sharpe']:.2f}",
        'Max DD': f"{r['max_drawdown']*100:.2f}%",
        'Win Rate': f"{r['win_rate']*100:.1f}%",
        'Days': r['days']
    } for r in results])

    print(summary.to_string(index=False))
    print()

    # Portfolio (equal weight across all pairs)
    print("="*80)
    print("EQUAL-WEIGHT PORTFOLIO")
    print("="*80)

    eq_curves = []
    for r in results:
        ec = r['equity_curve'].copy()
        ec.columns = [r['pair']]
        eq_curves.append(ec)

    port_df = eq_curves[0]
    for ec in eq_curves[1:]:
        port_df = port_df.join(ec, how='inner')

    port_df['portfolio'] = port_df.mean(axis=1)
    port_df['returns'] = port_df['portfolio'].pct_change()

    final = port_df['portfolio'].iloc[-1]
    port_return = (final / 100000 - 1)
    years = len(port_df) / 365.25
    port_cagr = (final / 100000) ** (1/years) - 1
    port_vol = port_df['returns'].std() * np.sqrt(252)
    port_sharpe = (port_df['returns'].mean() * 252) / port_vol if port_vol > 0 else 0

    cumulative = (1 + port_df['returns']).cumprod()
    running_max = cumulative.expanding().max()
    dd = (cumulative - running_max) / running_max
    port_mdd = dd.min()
    port_wr = (port_df['returns'] > 0).sum() / len(port_df['returns'].dropna())

    print(f"Total Return: {port_return*100:.2f}%")
    print(f"CAGR: {port_cagr*100:.2f}%")
    print(f"Volatility: {port_vol*100:.2f}%")
    print(f"Sharpe: {port_sharpe:.2f}")
    print(f"Max Drawdown: {port_mdd*100:.2f}%")
    print(f"Win Rate: {port_wr*100:.2f}%")

    # Save
    results_df = pd.DataFrame([{
        'pair': r['pair'],
        'total_return': r['total_return'],
        'cagr': r['cagr'],
        'volatility': r['volatility'],
        'sharpe': r['sharpe'],
        'max_drawdown': r['max_drawdown'],
        'win_rate': r['win_rate'],
        'days': r['days']
    } for r in results])

    results_df.to_csv('ultra_simple_results.csv', index=False)
    print(f"\nResults saved to: ultra_simple_results.csv")


if __name__ == '__main__':
    main()
