"""
Analyze 1.0 BTAL / 0.7 SPY portfolio:
- Calculate beta and correlation to SPY
- Plot cumulative overnight returns vs SPY overnight returns
- Calculate average bps per night by time period
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmp_client import FMPClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate overnight returns using dividend-adjusted prices."""
    df = df.sort_values('date').copy()

    open_col = 'adjOpen' if 'adjOpen' in df.columns else 'open'
    close_col = 'adjClose' if 'adjClose' in df.columns else 'close'

    # Overnight return: (Open - Previous Close) / Previous Close
    df['overnight_return'] = (df[open_col] - df[close_col].shift(1)) / df[close_col].shift(1)

    return df


def calculate_beta(portfolio_returns: pd.Series, spy_returns: pd.Series) -> float:
    """Calculate beta of portfolio to SPY."""
    # Align the data
    combined = pd.DataFrame({
        'portfolio': portfolio_returns,
        'spy': spy_returns
    }).dropna()

    # Calculate covariance and variance
    covariance = combined['portfolio'].cov(combined['spy'])
    spy_variance = combined['spy'].var()

    beta = covariance / spy_variance if spy_variance != 0 else 0
    return beta


def main():
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{timestamp}_btal_spy_analysis.txt")

    # Create a function to log and print
    def log_print(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    log_print("Analyzing 1.0 BTAL / 0.7 SPY Portfolio")
    log_print("=" * 80)

    # Initialize client
    client = FMPClient()

    # Define lookback periods
    lookback_periods = {
        '6 Months': 180,
        '12 Months': 365,
        '3 Years': 365 * 3,
        '5 Years': 365 * 5,
        '10 Years': 365 * 10
    }

    # Fetch data
    max_days = max(lookback_periods.values())
    from_date = (datetime.now() - timedelta(days=max_days + 30)).strftime('%Y-%m-%d')

    log_print("\nFetching data for SPY...")
    spy_data = client.get_historical_prices(symbol='SPY', from_date=from_date, adjusted=True)
    spy_df = pd.DataFrame(spy_data)
    spy_df = calculate_returns(spy_df)

    log_print("Fetching data for BTAL...")
    btal_data = client.get_historical_prices(symbol='BTAL', from_date=from_date, adjusted=True)
    btal_df = pd.DataFrame(btal_data)
    btal_df = calculate_returns(btal_df)

    # Calculate portfolio overnight returns (1.0 * BTAL + 0.7 * SPY)
    df = spy_df[['date']].copy()
    df = df.merge(spy_df[['date', 'overnight_return']], on='date', suffixes=('', '_spy'))
    df = df.merge(btal_df[['date', 'overnight_return']], on='date', suffixes=('', '_btal'))
    df.columns = ['date', 'spy_overnight', 'btal_overnight']
    df['portfolio_overnight'] = 0.7 * df['spy_overnight'] + 1.0 * df['btal_overnight']

    log_print(f"\nReceived {len(df)} days of data")
    log_print(f"Date range: {df['date'].min()} to {df['date'].max()}\n")

    # Analyze each lookback period
    log_print("\nBETA, CORRELATION & RETURNS ANALYSIS")
    log_print("=" * 80)

    for period_name, days in lookback_periods.items():
        cutoff_date = datetime.now() - timedelta(days=days)
        period_df = df[df['date'] >= cutoff_date.strftime('%Y-%m-%d')].copy()

        if len(period_df) < 10:
            log_print(f"\n{period_name}: Insufficient data")
            continue

        # Calculate beta
        beta = calculate_beta(period_df['portfolio_overnight'], period_df['spy_overnight'])

        # Calculate correlation
        correlation = period_df[['portfolio_overnight', 'spy_overnight']].corr().iloc[0, 1]

        # Calculate average bps per night for portfolio
        avg_return_portfolio = period_df['portfolio_overnight'].mean()
        avg_bps_portfolio = avg_return_portfolio * 10000  # Convert to bps

        # Calculate average bps per night for SPY
        avg_return_spy = period_df['spy_overnight'].mean()
        avg_bps_spy = avg_return_spy * 10000  # Convert to bps

        # Calculate Sharpe ratio (annualized)
        portfolio_std = period_df['portfolio_overnight'].std() * np.sqrt(252)
        portfolio_ann_return = avg_return_portfolio * 252
        portfolio_sharpe = portfolio_ann_return / portfolio_std if portfolio_std > 0 else 0

        spy_std = period_df['spy_overnight'].std() * np.sqrt(252)
        spy_ann_return = avg_return_spy * 252
        spy_sharpe = spy_ann_return / spy_std if spy_std > 0 else 0

        log_print(f"\n{period_name}:")
        log_print(f"  Beta to SPY (overnight):      {beta:>10.3f}")
        log_print(f"  Correlation to SPY:           {correlation:>10.3f}")
        log_print(f"  Portfolio avg bps per night:  {avg_bps_portfolio:>10.2f}")
        log_print(f"  SPY avg bps per night:        {avg_bps_spy:>10.2f}")
        log_print(f"  Portfolio Sharpe:             {portfolio_sharpe:>10.3f}")
        log_print(f"  SPY Sharpe:                   {spy_sharpe:>10.3f}")
        log_print(f"  Number of days:               {len(period_df):>10}")

    # Create cumulative PnL plots
    log_print("\n\n" + "=" * 80)
    log_print("Generating cumulative PnL plots...")

    num_periods = len(lookback_periods)
    fig, axes = plt.subplots(num_periods, 1, figsize=(14, 5 * num_periods))

    if num_periods == 1:
        axes = [axes]

    for idx, (period_name, days) in enumerate(lookback_periods.items()):
        cutoff_date = datetime.now() - timedelta(days=days)
        period_df = df[df['date'] >= cutoff_date.strftime('%Y-%m-%d')].copy()

        if len(period_df) < 10:
            continue

        # Calculate cumulative returns
        period_df['portfolio_cumulative'] = (1 + period_df['portfolio_overnight'].fillna(0)).cumprod()
        period_df['spy_cumulative'] = (1 + period_df['spy_overnight'].fillna(0)).cumprod()

        # Plot cumulative returns
        ax = axes[idx]
        ax.plot(pd.to_datetime(period_df['date']), period_df['portfolio_cumulative'],
                label='1.0 BTAL / 0.7 SPY (Overnight)', linewidth=2.5, color='blue')
        ax.plot(pd.to_datetime(period_df['date']), period_df['spy_cumulative'],
                label='SPY (Overnight)', linewidth=2.5, color='orange')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{period_name}: Cumulative Overnight Returns',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (1 = no change)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"{timestamp}_btal_spy_cumulative_overnight.png"
    filepath = os.path.join(project_dir, 'plots', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    log_print(f"Cumulative returns chart saved as plots/{filename}")
    plt.close()

    log_print("\n" + "=" * 80)
    log_print("Analysis complete!")
    log_print(f"\nLog file saved as logs/{os.path.basename(log_file)}")


if __name__ == "__main__":
    main()
