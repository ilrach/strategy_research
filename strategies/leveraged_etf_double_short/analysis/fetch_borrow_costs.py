"""
Fetch historical borrow cost data from iBorrowDesk API

iBorrowDesk provides historical borrow fee data for stocks/ETFs.
API endpoint: https://iborrowdesk.com/api/ticker/{SYMBOL}
"""

import requests
import pandas as pd
import json
from datetime import datetime

def fetch_borrow_data(ticker):
    """
    Fetch borrow cost data from iBorrowDesk API

    Args:
        ticker: Stock/ETF ticker symbol (e.g., 'TQQQ')

    Returns:
        DataFrame with historical borrow fee data
    """
    url = f"https://iborrowdesk.com/api/ticker/{ticker}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    print(f"Fetching borrow data for {ticker}...")
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: HTTP {response.status_code}")

    data = response.json()

    # Extract daily historical data
    daily_data = data.get('daily', [])

    if not daily_data:
        print(f"No daily data found for {ticker}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(daily_data)

    # Convert date string to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date ascending
    df = df.sort_values('date').reset_index(drop=True)

    return df

if __name__ == '__main__':
    # Fetch TQQQ data
    ticker = 'TQQQ'
    df = fetch_borrow_data(ticker)

    if df is not None:
        print(f"\n{'='*80}")
        print(f"BORROW COST DATA: {ticker}")
        print(f"{'='*80}\n")

        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"Total records: {len(df)}")
        print()

        # Summary statistics
        print("Summary Statistics:")
        print(f"  Borrow Fee:")
        print(f"    Mean:   {df['fee'].mean():.4f}%")
        print(f"    Median: {df['fee'].median():.4f}%")
        print(f"    Min:    {df['fee'].min():.4f}%")
        print(f"    Max:    {df['fee'].max():.4f}%")
        print()

        print(f"  Available Shares:")
        print(f"    Mean:   {df['available'].mean()/1e6:.2f}M")
        print(f"    Median: {df['available'].median()/1e6:.2f}M")
        print(f"    Min:    {df['available'].min()/1e6:.2f}M")
        print(f"    Max:    {df['available'].max()/1e6:.2f}M")
        print()

        # Show first and last few records
        print("First 5 records:")
        print(df[['date', 'available', 'fee', 'rebate']].head().to_string(index=False))
        print()

        print("Last 5 records:")
        print(df[['date', 'available', 'fee', 'rebate']].tail().to_string(index=False))
        print()

        # Save to CSV
        output_file = f'{ticker}_borrow_costs.csv'
        df.to_csv(output_file, index=False)
        print(f"Data saved to: {output_file}")

        # Also try SQQQ for comparison
        print(f"\n{'='*80}")
        ticker2 = 'SQQQ'
        df2 = fetch_borrow_data(ticker2)

        if df2 is not None:
            print(f"\n{'='*80}")
            print(f"BORROW COST DATA: {ticker2}")
            print(f"{'='*80}\n")

            print(f"Date range: {df2['date'].min().date()} to {df2['date'].max().date()}")
            print(f"Total records: {len(df2)}")
            print()

            print("Summary Statistics:")
            print(f"  Borrow Fee:")
            print(f"    Mean:   {df2['fee'].mean():.4f}%")
            print(f"    Median: {df2['fee'].median():.4f}%")
            print(f"    Min:    {df2['fee'].min():.4f}%")
            print(f"    Max:    {df2['fee'].max():.4f}%")
            print()

            output_file2 = f'{ticker2}_borrow_costs.csv'
            df2.to_csv(output_file2, index=False)
            print(f"Data saved to: {output_file2}")
