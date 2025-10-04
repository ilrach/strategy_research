"""
Calculate BTAL beta to SPY during overnight periods.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmp_client import FMPClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def calculate_overnight_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate overnight returns using adjusted prices."""
    df = df.sort_values('date').copy()

    open_col = 'adjOpen' if 'adjOpen' in df.columns else 'open'
    close_col = 'adjClose' if 'adjClose' in df.columns else 'close'

    # Overnight return: (Open - Previous Close) / Previous Close
    df['overnight_return'] = (df[open_col] - df[close_col].shift(1)) / df[close_col].shift(1)

    return df


def calculate_beta(returns_x: pd.Series, returns_y: pd.Series) -> tuple:
    """
    Calculate beta and correlation.

    Args:
        returns_x: Independent variable returns (SPY)
        returns_y: Dependent variable returns (BTAL)

    Returns:
        (beta, correlation, r_squared)
    """
    # Align and drop NaN
    df = pd.DataFrame({'x': returns_x, 'y': returns_y}).dropna()

    if len(df) < 2:
        return None, None, None

    # Calculate beta using covariance method
    covariance = df['x'].cov(df['y'])
    variance_x = df['x'].var()
    beta = covariance / variance_x if variance_x != 0 else 0

    # Calculate correlation
    correlation = df['x'].corr(df['y'])

    # R-squared
    r_squared = correlation ** 2

    return beta, correlation, r_squared


def main():
    client = FMPClient()

    # Fetch data for both symbols
    from_date = (datetime.now() - timedelta(days=365 * 10 + 30)).strftime('%Y-%m-%d')

    print("Fetching SPY data...")
    spy_data = client.get_historical_prices('SPY', from_date=from_date, adjusted=True)
    spy_df = pd.DataFrame(spy_data)
    spy_df = calculate_overnight_returns(spy_df)

    print("Fetching BTAL data...")
    btal_data = client.get_historical_prices('BTAL', from_date=from_date, adjusted=True)
    btal_df = pd.DataFrame(btal_data)
    btal_df = calculate_overnight_returns(btal_df)

    # Merge on date
    merged = pd.merge(
        spy_df[['date', 'overnight_return']],
        btal_df[['date', 'overnight_return']],
        on='date',
        suffixes=('_spy', '_btal')
    )

    print(f"\nBTAL Overnight Beta to SPY")
    print("=" * 80)

    # Calculate for different periods
    periods = {
        '6 Months': 180,
        '1 Year': 365,
        '3 Years': 365 * 3,
        '5 Years': 365 * 5,
        '10 Years': 365 * 10
    }

    for period_name, days in periods.items():
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        period_data = merged[merged['date'] >= cutoff_date].copy()

        if len(period_data) < 10:
            continue

        beta, corr, r2 = calculate_beta(
            period_data['overnight_return_spy'],
            period_data['overnight_return_btal']
        )

        print(f"\n{period_name}:")
        print(f"  Beta:         {beta:>8.3f}")
        print(f"  Correlation:  {corr:>8.3f}")
        print(f"  R-squared:    {r2:>8.3f}")
        print(f"  Observations: {len(period_data):>8,}")

    print("\n" + "=" * 80)
    print("Note: Negative beta indicates inverse relationship to SPY during overnight")


if __name__ == "__main__":
    main()
