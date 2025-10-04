"""
Analyze intraday vs overnight returns for a weighted portfolio.
Portfolio: 0.8x SPY + 1.0x BTAL
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmp_client import FMPClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate intraday and overnight returns using adjusted prices."""
    df = df.sort_values('date').copy()

    open_col = 'adjOpen' if 'adjOpen' in df.columns else 'open'
    close_col = 'adjClose' if 'adjClose' in df.columns else 'close'

    # Intraday return: (Close - Open) / Open
    df['intraday_return'] = (df[close_col] - df[open_col]) / df[open_col]

    # Overnight return: (Open - Previous Close) / Previous Close
    df['overnight_return'] = (df[open_col] - df[close_col].shift(1)) / df[close_col].shift(1)

    return df


def calculate_metrics(returns: pd.Series, period_name: str) -> dict:
    """Calculate risk and return metrics for a return series."""
    returns = returns.dropna()
    annual_factor = 252

    total_return = (1 + returns).prod() - 1
    num_periods = len(returns)
    years = num_periods / annual_factor
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    volatility = returns.std() * np.sqrt(annual_factor)
    sharpe = annualized_return / volatility if volatility > 0 else 0

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0
    sortino = annualized_return / downside_std if downside_std > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    return {
        'period': period_name,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_periods': num_periods
    }


def main():
    # Portfolio weights
    spy_weight = 0.8
    btal_weight = 1.0

    print(f"Weighted Portfolio Analysis: {spy_weight}x SPY + {btal_weight}x BTAL")
    print("=" * 80)

    client = FMPClient()

    # Fetch data
    from_date = (datetime.now() - timedelta(days=365 * 10 + 30)).strftime('%Y-%m-%d')

    print("\nFetching SPY data...")
    spy_data = client.get_historical_prices('SPY', from_date=from_date, adjusted=True)
    spy_df = pd.DataFrame(spy_data)
    spy_df = calculate_returns(spy_df)

    print("Fetching BTAL data...")
    btal_data = client.get_historical_prices('BTAL', from_date=from_date, adjusted=True)
    btal_df = pd.DataFrame(btal_data)
    btal_df = calculate_returns(btal_df)

    # Merge on date
    merged = pd.merge(
        spy_df[['date', 'intraday_return', 'overnight_return']],
        btal_df[['date', 'intraday_return', 'overnight_return']],
        on='date',
        suffixes=('_spy', '_btal')
    )

    # Calculate weighted portfolio returns
    merged['portfolio_intraday'] = (
        spy_weight * merged['intraday_return_spy'] +
        btal_weight * merged['intraday_return_btal']
    )

    merged['portfolio_overnight'] = (
        spy_weight * merged['overnight_return_spy'] +
        btal_weight * merged['overnight_return_btal']
    )

    # Define lookback periods
    periods = {
        '6 Months': 180,
        '12 Months': 365,
        '3 Years': 365 * 3,
        '5 Years': 365 * 5,
        '10 Years': 365 * 10
    }

    # Analyze each period
    for period_name, days in periods.items():
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        period_df = merged[merged['date'] >= cutoff_date].copy()

        if len(period_df) < 10:
            print(f"Insufficient data for {period_name}")
            continue

        print(f"\n{period_name.upper()}")
        print("-" * 80)

        # Calculate metrics
        intraday_metrics = calculate_metrics(period_df['portfolio_intraday'], period_name)
        overnight_metrics = calculate_metrics(period_df['portfolio_overnight'], period_name)

        # Display results
        print(f"\nINTRADAY (Open to Close):")
        print(f"  Total Return:        {intraday_metrics['total_return']:>10.2%}")
        print(f"  Annualized Return:   {intraday_metrics['annualized_return']:>10.2%}")
        print(f"  Annualized Vol:      {intraday_metrics['volatility']:>10.2%}")
        print(f"  Sharpe Ratio:        {intraday_metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {intraday_metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:        {intraday_metrics['calmar_ratio']:>10.2f}")
        print(f"  Max Drawdown:        {intraday_metrics['max_drawdown']:>10.2%}")
        print(f"  Win Rate:            {intraday_metrics['win_rate']:>10.2%}")

        print(f"\nOVERNIGHT (Close to Open):")
        print(f"  Total Return:        {overnight_metrics['total_return']:>10.2%}")
        print(f"  Annualized Return:   {overnight_metrics['annualized_return']:>10.2%}")
        print(f"  Annualized Vol:      {overnight_metrics['volatility']:>10.2%}")
        print(f"  Sharpe Ratio:        {overnight_metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {overnight_metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:        {overnight_metrics['calmar_ratio']:>10.2f}")
        print(f"  Max Drawdown:        {overnight_metrics['max_drawdown']:>10.2%}")
        print(f"  Win Rate:            {overnight_metrics['win_rate']:>10.2%}")

    print("\n" + "=" * 80)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
