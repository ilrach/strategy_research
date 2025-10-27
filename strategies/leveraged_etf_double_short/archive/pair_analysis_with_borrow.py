"""
Analyze all pairs: volatility vs borrow cost

Calculate:
1. Annualized volatility of bull ETF (long only)
2. Borrow costs from iBorrowDesk for both bull and bear
3. Edge ratio: (Volatility / 4.5) / Total_Borrow_Cost

The threshold: For each 1% borrow cost, we need 4.5% volatility for positive edge.
So Edge Ratio > 1.0 means the pair has positive edge.
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
    """Fetch borrow cost data from iBorrowDesk API with rate limiting"""
    url = f"https://iborrowdesk.com/api/ticker/{ticker}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    try:
        time.sleep(3)  # Rate limiting: 3 seconds between requests
        response = requests.get(url, headers=headers, timeout=20)

        if response.status_code == 429:
            print("Rate limited, waiting 10 seconds...")
            time.sleep(10)
            return None

        if response.status_code != 200:
            print(f"HTTP {response.status_code}")
            return None

        data = response.json()
        daily_data = data.get('daily', [])

        if not daily_data or len(daily_data) == 0:
            return None

        df = pd.DataFrame(daily_data)

        if 'fee' not in df.columns:
            return None

        mean_fee = df['fee'].mean()

        return mean_fee
    except requests.exceptions.Timeout:
        print("Timeout")
        return None
    except requests.exceptions.ConnectionError:
        print("Connection error")
        return None
    except Exception as e:
        print(f"Error: {str(e)[:50]}")
        return None


def calculate_bull_volatility(bull, from_date, to_date):
    """Calculate annualized volatility of the bull ETF (long only)"""
    try:
        bull_data = client.get_historical_prices(bull, from_date=from_date, to_date=to_date)

        if not bull_data:
            return None

        df = pd.DataFrame(bull_data)[['date', 'adjClose']]
        df['returns'] = df['adjClose'].pct_change()

        # Annualized volatility
        volatility = df['returns'].std() * np.sqrt(252)

        return volatility

    except Exception:
        return None


def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')

    print("="*80)
    print("PAIR ANALYSIS: VOLATILITY VS BORROW COST")
    print("="*80)
    print(f"Period: {from_date} to {to_date}")
    print()
    print("Edge Threshold: For each 1% borrow cost, need 4.5% volatility")
    print("Edge Ratio = (Volatility / 4.5) / Total_Borrow_Cost")
    print("  > 1.0 = Positive edge")
    print("  < 1.0 = Negative edge")
    print()

    results = []

    for i, (bull, bear) in enumerate(PAIRS):
        print(f"[{i+1}/{len(PAIRS)}] {bull}/{bear}")

        # Calculate bull volatility
        print(f"  Calculating {bull} volatility...", end=" ")
        volatility = calculate_bull_volatility(bull, from_date, to_date)

        if volatility is None:
            print("SKIP - no price data")
            continue

        print(f"{volatility*100:.2f}%")

        # Fetch borrow costs
        print(f"  Fetching borrow costs...")
        print(f"    {bull}...", end=" ")
        borrow_bull = fetch_borrow_data(bull)
        if borrow_bull is not None:
            print(f"{borrow_bull:.3f}%")
        else:
            print("No data")
            borrow_bull = 0.0

        print(f"    {bear}...", end=" ")
        borrow_bear = fetch_borrow_data(bear)
        if borrow_bear is not None:
            print(f"{borrow_bear:.3f}%")
        else:
            print("No data")
            borrow_bear = 0.0

        total_borrow = borrow_bull + borrow_bear

        # Calculate edge ratio
        # Edge ratio = (Volatility / 4.5) / Total_Borrow_Cost
        if total_borrow > 0:
            edge_ratio = (volatility * 100 / 4.5) / total_borrow
        else:
            edge_ratio = np.inf if volatility > 0 else 0

        results.append({
            'pair': f"{bull}/{bear}",
            'bull': bull,
            'bear': bear,
            'bull_volatility': volatility * 100,  # as percentage
            'borrow_bull': borrow_bull,
            'borrow_bear': borrow_bear,
            'total_borrow': total_borrow,
            'vol_threshold': volatility * 100 / 4.5,  # Volatility / 4.5
            'edge_ratio': edge_ratio,
            'has_positive_edge': edge_ratio > 1.0 if not np.isinf(edge_ratio) else True
        })

        print()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by edge ratio descending
    df = df.sort_values('edge_ratio', ascending=False).reset_index(drop=True)

    print()
    print("="*80)
    print("RESULTS (SORTED BY EDGE RATIO)")
    print("="*80)
    print()

    # Print table
    print(df[['pair', 'bull_volatility', 'total_borrow', 'vol_threshold', 'edge_ratio', 'has_positive_edge']].to_string(index=False))
    print()

    # Summary statistics
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()

    pairs_with_borrow_data = df[df['total_borrow'] > 0]
    positive_edge = df[df['has_positive_edge'] == True]

    print(f"Total pairs analyzed: {len(df)}")
    print(f"Pairs with borrow data: {len(pairs_with_borrow_data)}")
    print(f"Pairs with positive edge: {len(positive_edge)}")
    print()

    if len(pairs_with_borrow_data) > 0:
        print("Pairs with borrow data statistics:")
        print(f"  Average volatility: {pairs_with_borrow_data['bull_volatility'].mean():.2f}%")
        print(f"  Average total borrow cost: {pairs_with_borrow_data['total_borrow'].mean():.2f}%")
        print(f"  Average edge ratio: {pairs_with_borrow_data['edge_ratio'].mean():.2f}")
        print()

    print("Top 5 pairs by edge ratio:")
    top5 = df.head(5)
    for _, row in top5.iterrows():
        edge_str = "INF" if np.isinf(row['edge_ratio']) else f"{row['edge_ratio']:.2f}"
        print(f"  {row['pair']:15} Vol: {row['bull_volatility']:6.2f}%  Borrow: {row['total_borrow']:6.2f}%  Edge: {edge_str}")

    print()

    # Save to CSV
    output_file = 'pair_volatility_borrow_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print()
    print("Analysis complete!")


if __name__ == '__main__':
    main()
