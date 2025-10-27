"""
Backtest with borrow costs from iBorrowDesk

Logic:
- We have $100k
- We short $50k of each ETF, which gives us $200k total cash
- That $200k earns BIL returns
- We pay daily borrow costs on the short positions
- Daily return = 2×BIL - 0.5×Bull - 0.5×Bear - (borrow_cost_bull/252) - (borrow_cost_bear/252)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

client = FMPClient()

def fetch_borrow_data(ticker):
    """
    Fetch borrow cost data from iBorrowDesk API

    Returns:
        DataFrame with date, fee columns
    """
    url = f"https://iborrowdesk.com/api/ticker/{ticker}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        data = response.json()
        daily_data = data.get('daily', [])

        if not daily_data:
            return None

        df = pd.DataFrame(daily_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'fee']].sort_values('date').reset_index(drop=True)

        return df
    except Exception as e:
        print(f"    Warning: Could not fetch borrow data for {ticker}: {e}")
        return None


def backtest_pair(bull: str, bear: str, from_date: str, to_date: str):
    """Backtest a pair with borrow costs."""
    try:
        # Fetch price data
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
        df = df.sort_index().reset_index()

        if len(df) < 2:
            return None

        # Fetch borrow cost data
        print(f"    Fetching borrow costs...")
        borrow_bull = fetch_borrow_data(bull)
        borrow_bear = fetch_borrow_data(bear)

        # Merge borrow costs
        if borrow_bull is not None:
            df = df.merge(borrow_bull, on='date', how='left', suffixes=('', '_bull_borrow'))
            df = df.rename(columns={'fee': 'borrow_fee_bull'})
        else:
            df['borrow_fee_bull'] = np.nan

        if borrow_bear is not None:
            df = df.merge(borrow_bear, on='date', how='left', suffixes=('', '_bear_borrow'))
            df = df.rename(columns={'fee': 'borrow_fee_bear'})
        else:
            df['borrow_fee_bear'] = np.nan

        # Fill missing borrow costs with mean (or zero if no data)
        if not df['borrow_fee_bull'].isna().all():
            mean_bull = df['borrow_fee_bull'].mean()
            df['borrow_fee_bull'] = df['borrow_fee_bull'].fillna(mean_bull)
        else:
            df['borrow_fee_bull'] = 0.0
            print(f"    No borrow data for {bull}, assuming 0%")

        if not df['borrow_fee_bear'].isna().all():
            mean_bear = df['borrow_fee_bear'].mean()
            df['borrow_fee_bear'] = df['borrow_fee_bear'].fillna(mean_bear)
        else:
            df['borrow_fee_bear'] = 0.0
            print(f"    No borrow data for {bear}, assuming 0%")

        df = df.set_index('date')

        # Calculate daily returns
        df['bil_ret'] = df['p_bil'].pct_change()
        df['bull_ret'] = df['p_bull'].pct_change()
        df['bear_ret'] = df['p_bear'].pct_change()

        # Portfolio return each day:
        # = 2x BIL return (cash on $200k)
        # + short bull return (= -1 * bull return, on 50% of portfolio)
        # + short bear return (= -1 * bear return, on 50% of portfolio)
        # - borrow costs (annualized fee / 252, applied to 50% of portfolio each)
        df['cash_ret'] = 2 * df['bil_ret']
        df['short_bull_ret'] = -0.5 * df['bull_ret']
        df['short_bear_ret'] = -0.5 * df['bear_ret']

        # Daily borrow cost = (annual_fee% / 100) / 252 * position_size
        # Position size is 0.5 (50% of portfolio)
        df['borrow_cost_bull'] = -0.5 * (df['borrow_fee_bull'] / 100) / 252
        df['borrow_cost_bear'] = -0.5 * (df['borrow_fee_bear'] / 100) / 252

        df['total_return'] = (df['cash_ret'] + df['short_bull_ret'] + df['short_bear_ret'] +
                              df['borrow_cost_bull'] + df['borrow_cost_bear'])

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

        # Calculate average borrow costs
        avg_borrow_bull = df['borrow_fee_bull'].mean()
        avg_borrow_bear = df['borrow_fee_bear'].mean()

        return {
            'pair': f"{bull}/{bear}",
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'days': len(df),
            'avg_borrow_bull': avg_borrow_bull,
            'avg_borrow_bear': avg_borrow_bear,
            'total_borrow_cost': avg_borrow_bull + avg_borrow_bear,
            'equity_curve': df[['equity']]
        }

    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    # Focus on TQQQ/SQQQ for now since we have borrow data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)

    print("="*80)
    print("BACKTEST WITH BORROW COSTS")
    print("="*80)
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Strategy: Daily return = 2×BIL - 0.5×Bull - 0.5×Bear - Borrow Costs")
    print()

    bull = 'TQQQ'
    bear = 'SQQQ'

    print(f"Testing {bull}/{bear}...")
    result = backtest_pair(bull, bear, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if not result:
        print("No results!")
        return

    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()

    print(f"Pair: {result['pair']}")
    print(f"Days: {result['days']}")
    print()

    print("Borrow Costs:")
    print(f"  {bull}: {result['avg_borrow_bull']:.4f}% annually")
    print(f"  {bear}: {result['avg_borrow_bear']:.4f}% annually")
    print(f"  Total: {result['total_borrow_cost']:.4f}% annually")
    print()

    print("Performance:")
    print(f"  Total Return: {result['total_return']*100:+.2f}%")
    print(f"  CAGR: {result['cagr']*100:+.2f}%")
    print(f"  Volatility: {result['volatility']*100:.2f}%")
    print(f"  Sharpe: {result['sharpe']:.2f}")
    print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {result['win_rate']*100:.1f}%")
    print()

    # Save results
    result_df = pd.DataFrame([{
        'pair': result['pair'],
        'total_return': result['total_return'],
        'cagr': result['cagr'],
        'volatility': result['volatility'],
        'sharpe': result['sharpe'],
        'max_drawdown': result['max_drawdown'],
        'win_rate': result['win_rate'],
        'avg_borrow_bull': result['avg_borrow_bull'],
        'avg_borrow_bear': result['avg_borrow_bear'],
        'total_borrow_cost': result['total_borrow_cost'],
        'days': result['days']
    }])

    result_df.to_csv('backtest_with_borrow_costs_results.csv', index=False)

    # Save equity curve
    result['equity_curve'].to_csv('backtest_with_borrow_costs_equity.csv')

    print("Results saved to:")
    print("  - backtest_with_borrow_costs_results.csv")
    print("  - backtest_with_borrow_costs_equity.csv")


if __name__ == '__main__':
    main()
