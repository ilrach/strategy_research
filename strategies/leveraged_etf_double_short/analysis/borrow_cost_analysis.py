"""
Create pair analysis CSV using volatility data and manually recorded borrow costs

Using borrow costs from earlier successful iBorrowDesk API calls before IP was blocked.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

client = FMPClient()

# Borrow costs collected from earlier successful iBorrowDesk API calls
KNOWN_BORROW_COSTS = {
    'TQQQ': 0.6897,
    'SQQQ': 1.8321,
    'SOXL': 0.6010,
    'SOXS': 5.7996,
    'SPXL': 1.2860,
    'SPXS': 2.5920,
    'UPRO': 0.6960,
    'SPXU': 2.5550,
    'TNA': 1.4690,
    'TZA': 8.1780,
    'YINN': 0.9712,
    'YANG': 5.3081,
    'LABU': 2.6844,
    'LABD': 15.6993,
    'NUGT': 0.0,  # No data available
    'DUST': 7.9894,
    'BOIL': 7.2189,
    'KOLD': 6.4334,
    'FAS': 2.0430,
    'FAZ': 7.4780,
    'GUSH': 0.7580,
    'DRIP': 12.4340,
    'UCO': 1.2300,
    'SCO': 6.3440,
    'DRN': 3.7510,
    'DRV': 3.8850,
    'DDM': 1.3520,
    'DXD': 7.2420,
    'USD': 0.5961,
    'SSG': 44.4386,
    'TMF': 0.5170,
    'TMV': 21.0650,
    'UBT': 1.7080,
    'TBT': 9.3260,
    'ERX': 3.8450,
    'ERY': 11.8520,
    # Pairs with no iBorrowDesk data
    'TECL': None,
    'TECS': None,
    'WEBL': None,
    'WEBS': None,
    'QLD': None,
    'QID': None,
    'SSO': None,
    'SDS': None,
    'UWM': None,
    'TWM': None,
    'AGQ': None,
    'ZSL': None,
    'UGL': None,
    'GLL': None,
    'UDOW': None,
    'SDOW': None,
}

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
    print("Using borrow costs from earlier iBorrowDesk API calls")
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

        # Get borrow costs from known data
        borrow_bull = KNOWN_BORROW_COSTS.get(bull)
        borrow_bear = KNOWN_BORROW_COSTS.get(bear)

        if borrow_bull is None:
            borrow_bull_display = "N/A"
            borrow_bull = np.nan
        else:
            borrow_bull_display = f"{borrow_bull:.3f}%"

        if borrow_bear is None:
            borrow_bear_display = "N/A"
            borrow_bear = np.nan
        else:
            borrow_bear_display = f"{borrow_bear:.3f}%"

        print(f"  Borrow costs: {bull} {borrow_bull_display}, {bear} {borrow_bear_display}")

        # Calculate total borrow (skip if either is N/A)
        if pd.isna(borrow_bull) or pd.isna(borrow_bear):
            total_borrow = np.nan
            vol_threshold = volatility * 100 / 4.5
            edge_ratio = np.nan
            has_positive_edge = None
        else:
            total_borrow = borrow_bull + borrow_bear
            vol_threshold = volatility * 100 / 4.5

            if total_borrow > 0:
                edge_ratio = vol_threshold / total_borrow
                has_positive_edge = edge_ratio > 1.0
            else:
                edge_ratio = np.inf
                has_positive_edge = True

        results.append({
            'pair': f"{bull}/{bear}",
            'bull': bull,
            'bear': bear,
            'bull_volatility_pct': volatility * 100,
            'borrow_bull_pct': borrow_bull if not pd.isna(borrow_bull) else None,
            'borrow_bear_pct': borrow_bear if not pd.isna(borrow_bear) else None,
            'total_borrow_pct': total_borrow if not pd.isna(total_borrow) else None,
            'vol_over_4.5': vol_threshold,
            'edge_ratio': edge_ratio if not pd.isna(edge_ratio) else None,
            'has_positive_edge': has_positive_edge
        })

        print()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by edge ratio descending (NaN/None last)
    df_sorted = df.sort_values('edge_ratio', ascending=False, na_position='last').reset_index(drop=True)

    print()
    print("="*80)
    print("RESULTS (SORTED BY EDGE RATIO)")
    print("="*80)
    print()

    # Print table
    for idx, row in df_sorted.iterrows():
        edge_str = "N/A" if row['edge_ratio'] is None else (
            "INF" if np.isinf(row['edge_ratio']) else f"{row['edge_ratio']:.2f}"
        )
        borrow_str = "N/A" if row['total_borrow_pct'] is None else f"{row['total_borrow_pct']:.2f}%"
        positive_str = "N/A" if row['has_positive_edge'] is None else ("YES" if row['has_positive_edge'] else "NO")

        print(f"{row['pair']:15} Vol: {row['bull_volatility_pct']:6.2f}%  " +
              f"Borrow: {borrow_str:>8}  Edge: {edge_str:>6}  Positive: {positive_str}")

    print()

    # Summary statistics
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()

    pairs_with_borrow_data = df[df['total_borrow_pct'].notna()]
    positive_edge = df[df['has_positive_edge'] == True]

    print(f"Total pairs analyzed: {len(df)}")
    print(f"Pairs with borrow data: {len(pairs_with_borrow_data)}")
    print(f"Pairs with positive edge: {len(positive_edge)}")
    print(f"Pairs without borrow data: {len(df) - len(pairs_with_borrow_data)}")
    print()

    if len(pairs_with_borrow_data) > 0:
        print("Pairs WITH borrow data statistics:")
        print(f"  Average volatility: {pairs_with_borrow_data['bull_volatility_pct'].mean():.2f}%")
        print(f"  Average total borrow cost: {pairs_with_borrow_data['total_borrow_pct'].mean():.2f}%")

        valid_edge_ratios = pairs_with_borrow_data[~pairs_with_borrow_data['edge_ratio'].apply(lambda x: np.isinf(x) if x is not None else True)]
        if len(valid_edge_ratios) > 0:
            print(f"  Average edge ratio: {valid_edge_ratios['edge_ratio'].mean():.2f}")
        print()

    print("Top 10 pairs by edge ratio (with borrow data):")
    top_with_data = df_sorted[df_sorted['total_borrow_pct'].notna()].head(10)
    for idx, row in top_with_data.iterrows():
        edge_str = "INF" if np.isinf(row['edge_ratio']) else f"{row['edge_ratio']:.2f}"
        print(f"  {idx+1:2d}. {row['pair']:15} Vol: {row['bull_volatility_pct']:6.2f}%  " +
              f"Borrow: {row['total_borrow_pct']:6.2f}%  Edge: {edge_str:>6}")

    print()

    # Save to CSV
    output_file = 'pair_volatility_borrow_analysis.csv'
    df_sorted.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print()
    print("Analysis complete!")


if __name__ == '__main__':
    main()
