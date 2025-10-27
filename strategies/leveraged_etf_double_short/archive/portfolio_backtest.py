"""
Portfolio backtest: Top 10 most volatile pairs with borrow costs

Strategy:
1. Select top 10 most volatile pairs from initial analysis
2. Short each pair with 10% of portfolio (5% bull + 5% bear each)
3. Portfolio has 200% cash earning BIL returns
4. Subtract actual borrow costs from iBorrowDesk for each pair
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


def fetch_borrow_data(ticker):
    """Fetch borrow cost data from iBorrowDesk API"""
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
    except Exception:
        return None


def calculate_volatility(bull, bear, from_date, to_date):
    """Calculate average volatility for a pair"""
    try:
        bull_data = client.get_historical_prices(bull, from_date=from_date, to_date=to_date)
        bear_data = client.get_historical_prices(bear, from_date=from_date, to_date=to_date)

        if not bull_data or not bear_data:
            return None, None

        df_bull = pd.DataFrame(bull_data)[['date', 'adjClose']]
        df_bear = pd.DataFrame(bear_data)[['date', 'adjClose']]

        df_bull['returns'] = df_bull['adjClose'].pct_change()
        df_bear['returns'] = df_bear['adjClose'].pct_change()

        vol_bull = df_bull['returns'].std() * np.sqrt(252)
        vol_bear = df_bear['returns'].std() * np.sqrt(252)

        avg_vol = (vol_bull + vol_bear) / 2

        return avg_vol, len(df_bull)

    except Exception:
        return None, None


def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')

    print("="*80)
    print("PORTFOLIO BACKTEST: TOP 10 VOLATILE PAIRS")
    print("="*80)
    print(f"Period: {from_date} to {to_date}")
    print()

    # Step 1: Calculate volatility for all pairs
    print("Step 1: Calculating volatility for all pairs...")
    print()

    pair_vols = []
    for i, (bull, bear) in enumerate(PAIRS):
        print(f"[{i+1}/{len(PAIRS)}] {bull}/{bear}...", end=" ")
        vol, days = calculate_volatility(bull, bear, from_date, to_date)

        if vol is not None:
            pair_vols.append({
                'bull': bull,
                'bear': bear,
                'pair': f"{bull}/{bear}",
                'volatility': vol,
                'days': days
            })
            print(f"Vol: {vol*100:.2f}%")
        else:
            print("SKIP")

    # Sort by volatility descending and take top 10
    pair_vols_df = pd.DataFrame(pair_vols).sort_values('volatility', ascending=False)
    top10 = pair_vols_df.head(10).reset_index(drop=True)

    print()
    print("="*80)
    print("TOP 10 MOST VOLATILE PAIRS")
    print("="*80)
    print()
    print(top10[['pair', 'volatility', 'days']].to_string(index=True))
    print()

    # Step 2: Fetch all price data and borrow costs
    print("="*80)
    print("Step 2: Fetching price data and borrow costs...")
    print("="*80)
    print()

    # Fetch BIL
    print("Fetching BIL...")
    bil_data = client.get_historical_prices('BIL', from_date=from_date, to_date=to_date)
    df_bil = pd.DataFrame(bil_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_bil'})
    df_bil['date'] = pd.to_datetime(df_bil['date'])
    df_bil = df_bil.set_index('date')

    # Initialize portfolio dataframe
    portfolio_df = df_bil.copy()
    portfolio_df['bil_ret'] = portfolio_df['p_bil'].pct_change()

    pair_data = []

    for idx, row in top10.iterrows():
        bull = row['bull']
        bear = row['bear']
        pair_name = row['pair']

        print(f"\n{pair_name}:")
        print(f"  Fetching prices...")

        bull_data = client.get_historical_prices(bull, from_date=from_date, to_date=to_date)
        bear_data = client.get_historical_prices(bear, from_date=from_date, to_date=to_date)

        if not bull_data or not bear_data:
            print(f"  SKIP - no price data")
            continue

        df_bull = pd.DataFrame(bull_data)[['date', 'adjClose']].rename(columns={'adjClose': f'p_{bull}'})
        df_bear = pd.DataFrame(bear_data)[['date', 'adjClose']].rename(columns={'adjClose': f'p_{bear}'})

        df_bull['date'] = pd.to_datetime(df_bull['date'])
        df_bear['date'] = pd.to_datetime(df_bear['date'])

        df_pair = df_bull.set_index('date').join(df_bear.set_index('date'))

        # Calculate returns
        df_pair[f'ret_{bull}'] = df_pair[f'p_{bull}'].pct_change()
        df_pair[f'ret_{bear}'] = df_pair[f'p_{bear}'].pct_change()

        # Fetch borrow costs
        print(f"  Fetching borrow costs...")
        borrow_bull = fetch_borrow_data(bull)
        borrow_bear = fetch_borrow_data(bear)

        # Merge borrow costs
        if borrow_bull is not None:
            borrow_bull['date'] = pd.to_datetime(borrow_bull['date'])
            borrow_bull = borrow_bull.set_index('date').rename(columns={'fee': f'borrow_{bull}'})
            df_pair = df_pair.join(borrow_bull)
        else:
            df_pair[f'borrow_{bull}'] = np.nan

        if borrow_bear is not None:
            borrow_bear['date'] = pd.to_datetime(borrow_bear['date'])
            borrow_bear = borrow_bear.set_index('date').rename(columns={'fee': f'borrow_{bear}'})
            df_pair = df_pair.join(borrow_bear)
        else:
            df_pair[f'borrow_{bear}'] = np.nan

        # Fill missing borrow costs with mean
        if not df_pair[f'borrow_{bull}'].isna().all():
            mean_bull = df_pair[f'borrow_{bull}'].mean()
            df_pair[f'borrow_{bull}'] = df_pair[f'borrow_{bull}'].fillna(mean_bull)
            print(f"    {bull}: {mean_bull:.4f}% annually")
        else:
            df_pair[f'borrow_{bull}'] = 0.0
            print(f"    {bull}: No data, assuming 0%")

        if not df_pair[f'borrow_{bear}'].isna().all():
            mean_bear = df_pair[f'borrow_{bear}'].mean()
            df_pair[f'borrow_{bear}'] = df_pair[f'borrow_{bear}'].fillna(mean_bear)
            print(f"    {bear}: {mean_bear:.4f}% annually")
        else:
            df_pair[f'borrow_{bear}'] = 0.0
            print(f"    {bear}: No data, assuming 0%")

        # Calculate pair contribution to portfolio
        # Each pair gets 10% allocation (5% bull + 5% bear)
        # Contribution = -0.05 * bull_ret - 0.05 * bear_ret - 0.05 * (borrow_bull/252) - 0.05 * (borrow_bear/252)
        df_pair[f'contribution_{pair_name}'] = (
            -0.05 * df_pair[f'ret_{bull}']
            - 0.05 * df_pair[f'ret_{bear}']
            - 0.05 * (df_pair[f'borrow_{bull}'] / 100) / 252
            - 0.05 * (df_pair[f'borrow_{bear}'] / 100) / 252
        )

        # Merge into portfolio
        portfolio_df = portfolio_df.join(df_pair[[f'contribution_{pair_name}']])

        pair_data.append({
            'pair': pair_name,
            'bull': bull,
            'bear': bear,
            'avg_borrow_bull': df_pair[f'borrow_{bull}'].mean(),
            'avg_borrow_bear': df_pair[f'borrow_{bear}'].mean(),
        })

    print()
    print("="*80)
    print("Step 3: Calculating portfolio returns...")
    print("="*80)
    print()

    # Sum all pair contributions
    contribution_cols = [col for col in portfolio_df.columns if col.startswith('contribution_')]

    # Portfolio daily return = 2×BIL + sum of all pair contributions
    portfolio_df['cash_contribution'] = 2 * portfolio_df['bil_ret']
    portfolio_df['pairs_contribution'] = portfolio_df[contribution_cols].sum(axis=1)
    portfolio_df['total_return'] = portfolio_df['cash_contribution'] + portfolio_df['pairs_contribution']

    # Build equity curve
    portfolio_df = portfolio_df.dropna(subset=['total_return'])

    equity = 100000
    equity_curve = [equity]

    for i in range(1, len(portfolio_df)):
        ret = portfolio_df.iloc[i]['total_return']
        equity = equity * (1 + ret)
        equity_curve.append(equity)

    portfolio_df['equity'] = equity_curve

    # Calculate metrics
    final_equity = portfolio_df['equity'].iloc[-1]
    total_return = (final_equity / 100000 - 1)

    years = len(portfolio_df) / 365.25
    cagr = (final_equity / 100000) ** (1/years) - 1 if years > 0 else 0

    returns = portfolio_df['equity'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    win_rate = (returns > 0).sum() / len(returns)

    # Print results
    print("="*80)
    print("PORTFOLIO RESULTS")
    print("="*80)
    print()

    print(f"Trading days: {len(portfolio_df)}")
    print(f"Number of pairs: {len(contribution_cols)}")
    print()

    print("Performance:")
    print(f"  Total Return: {total_return*100:+.2f}%")
    print(f"  CAGR: {cagr*100:+.2f}%")
    print(f"  Volatility: {volatility*100:.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd*100:.2f}%")
    print(f"  Win Rate: {win_rate*100:.1f}%")
    print()

    # Calculate average borrow costs
    avg_borrow_total = sum([p['avg_borrow_bull'] + p['avg_borrow_bear'] for p in pair_data]) / len(pair_data)
    print(f"Average borrow cost per pair: {avg_borrow_total:.4f}% annually")
    print()

    # Print borrow costs by pair
    print("Borrow Costs by Pair:")
    for p in pair_data:
        total_borrow = p['avg_borrow_bull'] + p['avg_borrow_bear']
        print(f"  {p['pair']:15} Bull: {p['avg_borrow_bull']:6.3f}%  Bear: {p['avg_borrow_bear']:6.3f}%  Total: {total_borrow:6.3f}%")

    print()

    # Save results
    portfolio_df[['equity', 'total_return']].to_csv('portfolio_backtest_equity.csv')

    results_summary = {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'num_pairs': len(contribution_cols),
        'days': len(portfolio_df),
        'avg_borrow_cost': avg_borrow_total,
    }

    pd.DataFrame([results_summary]).to_csv('portfolio_backtest_summary.csv', index=False)

    # Save pair info
    pd.DataFrame(pair_data).to_csv('portfolio_backtest_pairs.csv', index=False)

    print("Results saved:")
    print("  - portfolio_backtest_equity.csv")
    print("  - portfolio_backtest_summary.csv")
    print("  - portfolio_backtest_pairs.csv")
    print()
    print("Analysis complete!")


if __name__ == '__main__':
    main()
