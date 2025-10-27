"""
Analysis of leveraged ETF pairs - dollar volume and volatility metrics.
This script fetches historical data for long/short leveraged ETF pairs and calculates:
- Average daily dollar volume
- Annualized volatility
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
# Add parent directory to path to import fmp_client
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient
from typing import List, Dict, Tuple
import time


# Leveraged ETF pairs (Bull/Bear)
LEVERAGED_PAIRS = [
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
    ('TYD', 'TYO'),
    ('DRN', 'DRV'),
    ('RETL', 'RETS'),
    ('ERX', 'ERY'),
    ('USD', 'SSG'),
    ('WEBL', 'WEBS'),
    ('DPST', 'WDRW'),
    ('DIG', 'DUG'),
    ('BIB', 'BIS'),
    ('ROM', 'REW'),
    ('UYM', 'SMN'),
    ('URE', 'SRS'),
    ('AGQ', 'ZSL'),
    ('UGL', 'GLL'),
]


def calculate_metrics(data: List[Dict], ticker: str) -> Dict:
    """
    Calculate dollar volume and volatility metrics from historical data.

    Args:
        data: List of historical price dictionaries
        ticker: Ticker symbol

    Returns:
        Dictionary with calculated metrics
    """
    if not data or len(data) < 2:
        return {
            'ticker': ticker,
            'avg_dollar_volume': 0,
            'volatility': 0,
            'data_points': 0,
            'error': 'Insufficient data'
        }

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Use adjClose for price (dividend-adjusted endpoint provides this)
    df['close'] = df['adjClose']

    # Calculate dollar volume (close price * volume)
    df['dollar_volume'] = df['close'] * df['volume']
    avg_dollar_volume = df['dollar_volume'].mean()

    # Calculate daily returns
    df['returns'] = df['close'].pct_change()

    # Annualized volatility (assuming 252 trading days)
    volatility = df['returns'].std() * np.sqrt(252)

    return {
        'ticker': ticker,
        'avg_dollar_volume': avg_dollar_volume,
        'volatility': volatility,
        'data_points': len(df),
        'latest_price': df.iloc[-1]['close'] if len(df) > 0 else 0,
        'latest_volume': df.iloc[-1]['volume'] if len(df) > 0 else 0,
    }


def fetch_and_analyze(lookback_days: int = 252) -> pd.DataFrame:
    """
    Fetch historical data for all leveraged ETFs and calculate metrics.

    Args:
        lookback_days: Number of days to look back (default 252 = ~1 year)

    Returns:
        DataFrame with analysis results
    """
    client = FMPClient()

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')

    print(f"Fetching data from {from_date} to {to_date}")
    print(f"Analyzing {len(LEVERAGED_PAIRS) * 2} leveraged ETFs...")
    print()

    results = []
    total_tickers = len(LEVERAGED_PAIRS) * 2

    for i, (bull, bear) in enumerate(LEVERAGED_PAIRS):
        # Fetch data for bull ETF
        try:
            print(f"[{i*2 + 1}/{total_tickers}] Fetching {bull}...")
            bull_data = client.get_historical_prices(bull, from_date=from_date, to_date=to_date)
            bull_metrics = calculate_metrics(bull_data, bull)
            results.append(bull_metrics)
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"Error fetching {bull}: {e}")
            results.append({
                'ticker': bull,
                'avg_dollar_volume': 0,
                'volatility': 0,
                'data_points': 0,
                'error': str(e)
            })

        # Fetch data for bear ETF
        try:
            print(f"[{i*2 + 2}/{total_tickers}] Fetching {bear}...")
            bear_data = client.get_historical_prices(bear, from_date=from_date, to_date=to_date)
            bear_metrics = calculate_metrics(bear_data, bear)
            results.append(bear_metrics)
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"Error fetching {bear}: {e}")
            results.append({
                'ticker': bear,
                'avg_dollar_volume': 0,
                'volatility': 0,
                'data_points': 0,
                'error': str(e)
            })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Add pair information
    pair_list = []
    for bull, bear in LEVERAGED_PAIRS:
        pair_list.extend([f"{bull}/{bear}", f"{bull}/{bear}"])
    df['pair'] = pair_list

    # Reorder columns
    columns = ['pair', 'ticker', 'avg_dollar_volume', 'volatility', 'latest_price', 'latest_volume', 'data_points']
    df = df[columns]

    return df


def format_output(df: pd.DataFrame, min_volume_m: float = 5.0):
    """
    Format and display the results in a clean table.

    Args:
        df: DataFrame with analysis results
        min_volume_m: Minimum average dollar volume in millions (default 5M)
    """
    # Format dollar volume in millions
    df['avg_dollar_volume_m'] = df['avg_dollar_volume'] / 1_000_000

    # Format volatility as percentage
    df['volatility_pct'] = df['volatility'] * 100

    # Filter for minimum dollar volume
    df_filtered = df[df['avg_dollar_volume_m'] >= min_volume_m].copy()

    # Sort by volatility descending
    df_filtered = df_filtered.sort_values('volatility_pct', ascending=False)

    # Create display DataFrame
    display_df = df_filtered.copy()
    display_df['Avg Dollar Volume'] = display_df['avg_dollar_volume_m'].apply(lambda x: f"${x:.2f}M")
    display_df['Volatility'] = display_df['volatility_pct'].apply(lambda x: f"{x:.2f}%")
    display_df['Latest Price'] = display_df['latest_price'].apply(lambda x: f"${x:.2f}")
    display_df['Latest Volume'] = display_df['latest_volume'].apply(lambda x: f"{x:,.0f}")

    # Select and rename columns for display
    display_df = display_df[['pair', 'ticker', 'Avg Dollar Volume', 'Volatility', 'Latest Price', 'Latest Volume', 'data_points']]
    display_df.columns = ['Pair', 'Ticker', 'Avg $ Volume', 'Volatility', 'Price', 'Volume', 'Days']

    print("\n" + "="*100)
    print(f"LEVERAGED ETF ANALYSIS - DOLLAR VOLUME & VOLATILITY (Min ${min_volume_m}M Volume)")
    print("="*100)
    print(display_df.to_string(index=False))
    print("="*100)

    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print(f"Total ETFs analyzed: {len(df)}")
    print(f"ETFs meeting volume criteria (>${min_volume_m}M): {len(df_filtered)}")
    print(f"Average dollar volume (filtered): ${df_filtered['avg_dollar_volume_m'].mean():.2f}M")
    print(f"Average volatility (filtered): {df_filtered['volatility_pct'].mean():.2f}%")
    print(f"\nTop 5 by dollar volume:")
    top_volume = df_filtered.nlargest(5, 'avg_dollar_volume_m')[['ticker', 'avg_dollar_volume_m', 'volatility_pct']]
    for _, row in top_volume.iterrows():
        print(f"  {row['ticker']:6s} - ${row['avg_dollar_volume_m']:>10.2f}M  (Vol: {row['volatility_pct']:.2f}%)")

    print(f"\nTop 5 by volatility:")
    top_vol = df_filtered.nlargest(5, 'volatility_pct')[['ticker', 'volatility_pct', 'avg_dollar_volume_m']]
    for _, row in top_vol.iterrows():
        print(f"  {row['ticker']:6s} - {row['volatility_pct']:>6.2f}%  ($ Vol: ${row['avg_dollar_volume_m']:.2f}M)")

    return display_df


if __name__ == '__main__':
    # Run analysis
    results_df = fetch_and_analyze(lookback_days=252)  # 1 year of data

    # Format and display
    formatted_df = format_output(results_df)

    # Save to CSV
    output_file = 'leveraged_etf_analysis.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
