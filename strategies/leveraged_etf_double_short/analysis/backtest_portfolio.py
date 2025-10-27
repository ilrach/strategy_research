"""
Portfolio backtest: Top 10 most volatile pairs with reasonable borrow costs

Strategy:
1. Select top pairs by volatility that have reasonable borrow costs (<10% total)
2. Short each pair with equal weight
3. Portfolio has 200% cash earning BIL returns
4. Subtract actual borrow costs from iBorrowDesk for each pair
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import requests
import time

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
        time.sleep(0.5)  # Rate limiting
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return None, None

        data = response.json()
        daily_data = data.get('daily', [])

        if not daily_data:
            return None, None

        df = pd.DataFrame(daily_data)
        df['date'] = pd.to_datetime(df['date'])

        # Return both dataframe and mean fee
        mean_fee = df['fee'].mean()
        df = df[['date', 'fee']].sort_values('date').reset_index(drop=True)

        return df, mean_fee
    except Exception as e:
        return None, None


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
    print("PORTFOLIO BACKTEST: TOP VOLATILE PAIRS (REASONABLE BORROW COSTS)")
    print("="*80)
    print(f"Period: {from_date} to {to_date}")
    print()

    # Step 1: Calculate volatility and borrow costs for all pairs
    print("Step 1: Calculating volatility and borrow costs for all pairs...")
    print()

    pair_info = []
    for i, (bull, bear) in enumerate(PAIRS):
        print(f"[{i+1}/{len(PAIRS)}] {bull}/{bear}")
        print(f"  Volatility...", end=" ")
        vol, days = calculate_volatility(bull, bear, from_date, to_date)

        if vol is None:
            print("SKIP - no price data")
            continue

        print(f"{vol*100:.2f}%")

        # Fetch borrow costs
        print(f"  Borrow costs...")
        _, borrow_bull = fetch_borrow_data(bull)
        _, borrow_bear = fetch_borrow_data(bear)

        borrow_bull = borrow_bull if borrow_bull is not None else 0.0
        borrow_bear = borrow_bear if borrow_bear is not None else 0.0
        total_borrow = borrow_bull + borrow_bear

        print(f"    {bull}: {borrow_bull:.3f}%  {bear}: {borrow_bear:.3f}%  Total: {total_borrow:.3f}%")

        pair_info.append({
            'bull': bull,
            'bear': bear,
            'pair': f"{bull}/{bear}",
            'volatility': vol,
            'days': days,
            'borrow_bull': borrow_bull,
            'borrow_bear': borrow_bear,
            'total_borrow': total_borrow
        })

    # Filter by reasonable borrow costs (<10% total) and sort by volatility
    MAX_BORROW = 10.0
    pair_info_df = pd.DataFrame(pair_info)
    filtered = pair_info_df[pair_info_df['total_borrow'] < MAX_BORROW].copy()
    filtered = filtered.sort_values('volatility', ascending=False).reset_index(drop=True)

    print()
    print("="*80)
    print(f"PAIRS WITH BORROW COSTS < {MAX_BORROW}% (SORTED BY VOLATILITY)")
    print("="*80)
    print()
    print(filtered[['pair', 'volatility', 'total_borrow']].to_string(index=True))
    print()

    # Take top 10
    num_pairs = min(10, len(filtered))
    top_pairs = filtered.head(num_pairs).reset_index(drop=True)

    print(f"Selecting top {num_pairs} pairs for portfolio")
    print()

    # Step 2: Fetch all price data and build portfolio
    print("="*80)
    print("Step 2: Building portfolio...")
    print("="*80)
    print()

    # Fetch BIL
    print("Fetching BIL...")
    bil_data = client.get_historical_prices('BIL', from_date=from_date, to_date=to_date)
    df_bil = pd.DataFrame(bil_data)[['date', 'adjClose']].rename(columns={'adjClose': 'p_bil'})
    df_bil['date'] = pd.to_datetime(df_bil['date'])
    df_bil = df_bil.set_index('date').sort_index(ascending=True)  # CRITICAL: sort ascending by date

    # Initialize portfolio dataframe
    portfolio_df = df_bil.copy()
    portfolio_df['bil_ret'] = portfolio_df['p_bil'].pct_change()

    # Position size per pair
    # User requirement: "assume total portfolio has 200% cash that can earn BIL"
    # This means: short $200k total from $100k capital → 200% leverage
    # With 10 pairs equally weighted: each pair = 20% of capital (10% bull + 10% bear)
    total_leverage = 2.0  # 200% cash means 2x leverage
    position_size = total_leverage / num_pairs  # e.g., 20% if 10 pairs
    position_per_etf = position_size / 2  # Split between bull and bear (e.g., 10% each)

    print(f"Position size per pair: {position_size*100:.1f}% ({position_per_etf*100:.1f}% bull + {position_per_etf*100:.1f}% bear)")
    print(f"Total shorts: {num_pairs * position_size * 100:.0f}% → {num_pairs * position_size * 100:.0f}% cash earning BIL")
    print()

    pair_data = []

    for idx, row in top_pairs.iterrows():
        bull = row['bull']
        bear = row['bear']
        pair_name = row['pair']
        borrow_bull = row['borrow_bull']  # Already fetched in Step 1
        borrow_bear = row['borrow_bear']  # Already fetched in Step 1

        print(f"\n{pair_name}:")
        print(f"  Borrow costs: {bull} {borrow_bull:.3f}% + {bear} {borrow_bear:.3f}% = {borrow_bull+borrow_bear:.3f}%")
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
        df_pair = df_pair.sort_index(ascending=True)  # Ensure ascending date order

        # Calculate returns
        df_pair[f'ret_{bull}'] = df_pair[f'p_{bull}'].pct_change()
        df_pair[f'ret_{bear}'] = df_pair[f'p_{bear}'].pct_change()

        # Calculate pair contribution to portfolio
        # Contribution = -position_per_etf * bull_ret - position_per_etf * bear_ret - borrow costs
        df_pair[f'contribution_{pair_name}'] = (
            -position_per_etf * df_pair[f'ret_{bull}']
            - position_per_etf * df_pair[f'ret_{bear}']
            - position_per_etf * (borrow_bull / 100) / 252
            - position_per_etf * (borrow_bear / 100) / 252
        )

        # Merge into portfolio
        portfolio_df = portfolio_df.join(df_pair[[f'contribution_{pair_name}']])

        pair_data.append({
            'pair': pair_name,
            'bull': bull,
            'bear': bear,
            'borrow_bull': borrow_bull,
            'borrow_bear': borrow_bear,
            'total_borrow': borrow_bull + borrow_bear,
            'position_size': position_size * 100
        })

    print()
    print("="*80)
    print("Step 3: Calculating portfolio returns...")
    print("="*80)
    print()

    # Sum all pair contributions
    contribution_cols = [col for col in portfolio_df.columns if col.startswith('contribution_')]

    # Portfolio daily return = cash_multiplier×BIL + sum of all pair contributions
    # User requirement: 200% cash earning BIL
    cash_multiplier = 2.0  # 200% cash
    portfolio_df['cash_contribution'] = cash_multiplier * portfolio_df['bil_ret']
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

    # Calculate weighted average borrow cost
    pairs_df = pd.DataFrame(pair_data)
    avg_borrow = pairs_df['total_borrow'].mean()
    print(f"Average borrow cost per pair: {avg_borrow:.3f}% annually")
    print()

    # Print borrow costs by pair
    print("Portfolio Composition and Borrow Costs:")
    for _, p in pairs_df.iterrows():
        print(f"  {p['pair']:15} Position: {p['position_size']:5.1f}%  Borrow: {p['borrow_bull']:6.3f}% + {p['borrow_bear']:6.3f}% = {p['total_borrow']:6.3f}%")

    print()

    # Save results
    portfolio_df[['equity', 'total_return']].to_csv('portfolio_backtest_v2_equity.csv')

    results_summary = {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'num_pairs': len(contribution_cols),
        'days': len(portfolio_df),
        'avg_borrow_cost': avg_borrow,
    }

    pd.DataFrame([results_summary]).to_csv('portfolio_backtest_v2_summary.csv', index=False)
    pairs_df.to_csv('portfolio_backtest_v2_pairs.csv', index=False)

    print("Results saved:")
    print("  - portfolio_backtest_v2_equity.csv")
    print("  - portfolio_backtest_v2_summary.csv")
    print("  - portfolio_backtest_v2_pairs.csv")
    print()
    print("Analysis complete!")


if __name__ == '__main__':
    main()
