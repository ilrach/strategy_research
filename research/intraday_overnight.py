"""
Analyze intraday vs overnight returns for a given ticker.

Intraday return: (Close - Open) / Open
Overnight return: (Open - Previous Close) / Previous Close
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmp_client import FMPClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate intraday and overnight returns using dividend-adjusted prices.

    Args:
        df: DataFrame with OHLC data (adjusted prices)

    Returns:
        DataFrame with added return columns
    """
    # Sort by date ascending
    df = df.sort_values('date').copy()

    # Use adjusted prices if available (for total return including dividends)
    # Otherwise fall back to regular prices
    open_col = 'adjOpen' if 'adjOpen' in df.columns else 'open'
    close_col = 'adjClose' if 'adjClose' in df.columns else 'close'

    # Intraday return: (Close - Open) / Open
    df['intraday_return'] = (df[close_col] - df[open_col]) / df[open_col]

    # Overnight return: (Open - Previous Close) / Previous Close
    df['overnight_return'] = (df[open_col] - df[close_col].shift(1)) / df[close_col].shift(1)

    return df


def calculate_metrics(returns: pd.Series, period_name: str) -> Dict:
    """
    Calculate risk and return metrics for a return series.

    Args:
        returns: Series of returns
        period_name: Name of the period for identification

    Returns:
        Dictionary of metrics
    """
    # Remove NaN values
    returns = returns.dropna()

    # Annualization factor (assuming daily data)
    annual_factor = 252

    # Total return (compounded)
    total_return = (1 + returns).prod() - 1

    # Annualized return
    num_periods = len(returns)
    years = num_periods / annual_factor
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(annual_factor)

    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe = annualized_return / volatility if volatility > 0 else 0

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0
    sortino = annualized_return / downside_std if downside_std > 0 else 0

    # Calmar ratio (return / max drawdown)
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    # Average win/loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

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
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'num_periods': num_periods
    }


def plot_cumulative_returns(df: pd.DataFrame, symbol: str, lookback_periods: Dict[str, int], timestamp: str):
    """
    Plot cumulative returns for intraday vs overnight strategies.

    Args:
        df: DataFrame with return data
        symbol: Ticker symbol
        lookback_periods: Dictionary of period names to days
        timestamp: Timestamp for file naming
    """
    num_periods = len(lookback_periods)
    fig, axes = plt.subplots(num_periods, 1, figsize=(14, 4 * num_periods))

    if num_periods == 1:
        axes = [axes]

    for idx, (period_name, days) in enumerate(lookback_periods.items()):
        cutoff_date = datetime.now() - timedelta(days=days)
        period_df = df[df['date'] >= cutoff_date.strftime('%Y-%m-%d')].copy()

        if len(period_df) == 0:
            continue

        # Calculate cumulative returns
        period_df['intraday_cumulative'] = (1 + period_df['intraday_return'].fillna(0)).cumprod()
        period_df['overnight_cumulative'] = (1 + period_df['overnight_return'].fillna(0)).cumprod()

        # Plot
        ax = axes[idx]
        ax.plot(pd.to_datetime(period_df['date']), period_df['intraday_cumulative'],
                label='Intraday (Open to Close)', linewidth=2)
        ax.plot(pd.to_datetime(period_df['date']), period_df['overnight_cumulative'],
                label='Overnight (Close to Open)', linewidth=2)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{symbol} - {period_name} Cumulative Returns', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (1 = no change)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to plots directory with timestamp
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = f"{timestamp}_intraday_overnight_{symbol}.png"
    filepath = os.path.join(project_dir, 'plots', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nChart saved as plots/{filename}")
    plt.close()


def plot_combined_strategy(df: pd.DataFrame, symbol: str, lookback_periods: Dict[str, int], timestamp: str):
    """
    Plot combined strategy: Long overnight + Short intraday for SPY, or Short overnight + Long intraday for BTAL.

    Args:
        df: DataFrame with return data
        symbol: Ticker symbol
        lookback_periods: Dictionary of period names to days
        timestamp: Timestamp for file naming
    """
    # Determine strategy based on symbol
    if symbol.upper() == 'SPY':
        strategy_name = 'Long Overnight + Short Intraday'
    elif symbol.upper() == 'BTAL':
        strategy_name = 'Short Overnight + Long Intraday'
    else:
        strategy_name = 'Long Overnight + Short Intraday'  # Default

    num_periods = len(lookback_periods)
    fig, axes = plt.subplots(num_periods, 1, figsize=(14, 4 * num_periods))

    if num_periods == 1:
        axes = [axes]

    for idx, (period_name, days) in enumerate(lookback_periods.items()):
        cutoff_date = datetime.now() - timedelta(days=days)
        period_df = df[df['date'] >= cutoff_date.strftime('%Y-%m-%d')].copy()

        if len(period_df) == 0:
            continue

        # Calculate combined strategy returns
        if symbol.upper() == 'SPY':
            # Long overnight, short intraday
            period_df['combined_return'] = period_df['overnight_return'].fillna(0) - period_df['intraday_return'].fillna(0)
        elif symbol.upper() == 'BTAL':
            # Short overnight, long intraday
            period_df['combined_return'] = -period_df['overnight_return'].fillna(0) + period_df['intraday_return'].fillna(0)
        else:
            # Default: long overnight, short intraday
            period_df['combined_return'] = period_df['overnight_return'].fillna(0) - period_df['intraday_return'].fillna(0)

        period_df['combined_cumulative'] = (1 + period_df['combined_return']).cumprod()

        # Plot
        ax = axes[idx]
        ax.plot(pd.to_datetime(period_df['date']), period_df['combined_cumulative'],
                label=strategy_name, linewidth=2.5, color='purple')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{symbol} - {period_name} Combined Strategy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (1 = no change)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to plots directory with timestamp
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = f"{timestamp}_intraday_overnight_combined_{symbol}.png"
    filepath = os.path.join(project_dir, 'plots', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Chart saved as plots/{filename}")
    plt.close()


def save_risk_metrics(results: List[Dict], symbol: str, timestamp: str):
    """
    Save risk metrics to a text file.

    Args:
        results: List of result dictionaries
        symbol: Ticker symbol
        timestamp: Timestamp for file naming
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = f"{timestamp}_intraday_overnight_{symbol}.txt"
    filepath = os.path.join(project_dir, 'risk_metrics', filename)

    with open(filepath, 'w') as f:
        f.write(f"INTRADAY VS OVERNIGHT RETURNS ANALYSIS: {symbol}\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            period = result['period']
            intraday = result['intraday']
            overnight = result['overnight']

            f.write(f"\n{period.upper()}\n")
            f.write("-" * 80 + "\n")

            f.write(f"\nINTRADAY (Open to Close):\n")
            f.write(f"  Total Return:        {intraday['total_return']:>10.2%}\n")
            f.write(f"  Annualized Return:   {intraday['annualized_return']:>10.2%}\n")
            f.write(f"  Annualized Vol:      {intraday['volatility']:>10.2%}\n")
            f.write(f"  Sharpe Ratio:        {intraday['sharpe_ratio']:>10.2f}\n")
            f.write(f"  Sortino Ratio:       {intraday['sortino_ratio']:>10.2f}\n")
            f.write(f"  Calmar Ratio:        {intraday['calmar_ratio']:>10.2f}\n")
            f.write(f"  Max Drawdown:        {intraday['max_drawdown']:>10.2%}\n")
            f.write(f"  Win Rate:            {intraday['win_rate']:>10.2%}\n")
            f.write(f"  Avg Win:             {intraday['avg_win']:>10.2%}\n")
            f.write(f"  Avg Loss:            {intraday['avg_loss']:>10.2%}\n")

            f.write(f"\nOVERNIGHT (Close to Open):\n")
            f.write(f"  Total Return:        {overnight['total_return']:>10.2%}\n")
            f.write(f"  Annualized Return:   {overnight['annualized_return']:>10.2%}\n")
            f.write(f"  Annualized Vol:      {overnight['volatility']:>10.2%}\n")
            f.write(f"  Sharpe Ratio:        {overnight['sharpe_ratio']:>10.2f}\n")
            f.write(f"  Sortino Ratio:       {overnight['sortino_ratio']:>10.2f}\n")
            f.write(f"  Calmar Ratio:        {overnight['calmar_ratio']:>10.2f}\n")
            f.write(f"  Max Drawdown:        {overnight['max_drawdown']:>10.2%}\n")
            f.write(f"  Win Rate:            {overnight['win_rate']:>10.2%}\n")
            f.write(f"  Avg Win:             {overnight['avg_win']:>10.2%}\n")
            f.write(f"  Avg Loss:            {overnight['avg_loss']:>10.2%}\n")

    print(f"Risk metrics saved as risk_metrics/{filename}")


def main():
    # Parse command line arguments
    # Format: python intraday_overnight.py SPY
    #     or: python intraday_overnight.py "0.8*SPY+1.0*BTAL"
    if len(sys.argv) > 1:
        input_arg = sys.argv[1]

        # Check if it's a weighted portfolio (contains * or +)
        if '*' in input_arg or '+' in input_arg:
            # Parse weighted portfolio
            # Example: "0.8*SPY+1.0*BTAL"
            portfolio = {}
            parts = input_arg.replace(' ', '').split('+')
            for part in parts:
                weight, ticker = part.split('*')
                portfolio[ticker.upper()] = float(weight)
            symbol = '_'.join([f"{w:.1f}x{t}" for t, w in portfolio.items()])
            is_portfolio = True
        else:
            symbol = input_arg.upper()
            portfolio = {symbol: 1.0}
            is_portfolio = False
    else:
        symbol = "BTAL"
        portfolio = {symbol: 1.0}
        is_portfolio = False

    # Generate timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Define lookback periods (in days, approximate)
    lookback_periods = {
        '6 Months': 180,
        '12 Months': 365,
        '3 Years': 365 * 3,
        '5 Years': 365 * 5,
        '10 Years': 365 * 10
    }

    if is_portfolio:
        print(f"Analyzing intraday vs overnight returns for weighted portfolio:")
        print(f"  {' + '.join([f'{w}x {t}' for t, w in portfolio.items()])}")
    else:
        print(f"Analyzing intraday vs overnight returns for {symbol}")
    print("=" * 80)

    # Initialize client
    client = FMPClient()

    # Fetch data for all tickers in portfolio
    max_days = max(lookback_periods.values())
    from_date = (datetime.now() - timedelta(days=max_days + 30)).strftime('%Y-%m-%d')

    all_data = {}
    for ticker in portfolio.keys():
        print(f"\nFetching dividend-adjusted historical data for {ticker}...")
        data = client.get_historical_prices(symbol=ticker, from_date=from_date, adjusted=True)

        if not data:
            print(f"No data received for {ticker}")
            return

        df = pd.DataFrame(data)
        df = calculate_returns(df)
        all_data[ticker] = df

    # Merge and calculate weighted returns
    if is_portfolio:
        # Start with first ticker
        first_ticker = list(portfolio.keys())[0]
        df = all_data[first_ticker][['date']].copy()
        df['intraday_return'] = 0.0
        df['overnight_return'] = 0.0

        # Add weighted returns from each ticker
        for ticker, weight in portfolio.items():
            ticker_df = all_data[ticker][['date', 'intraday_return', 'overnight_return']]
            df = df.merge(ticker_df, on='date', suffixes=('', f'_{ticker}'))
            df['intraday_return'] = df['intraday_return'] + weight * df[f'intraday_return_{ticker}']
            df['overnight_return'] = df['overnight_return'] + weight * df[f'overnight_return_{ticker}']
            # Clean up temporary columns
            df = df.drop(columns=[f'intraday_return_{ticker}', f'overnight_return_{ticker}'])
    else:
        df = all_data[symbol]

    print(f"Received {len(df)} days of data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}\n")

    # Analyze each lookback period
    results = []

    for period_name, days in lookback_periods.items():
        cutoff_date = datetime.now() - timedelta(days=days)
        period_df = df[df['date'] >= cutoff_date.strftime('%Y-%m-%d')].copy()

        if len(period_df) < 10:
            print(f"Insufficient data for {period_name} (only {len(period_df)} days)")
            continue

        print(f"\n{period_name.upper()}")
        print("-" * 80)

        # Calculate metrics for both strategies
        intraday_metrics = calculate_metrics(period_df['intraday_return'], period_name)
        overnight_metrics = calculate_metrics(period_df['overnight_return'], period_name)

        # Display results with standardized metrics
        print(f"\nINTRADAY (Open to Close):")
        print(f"  Total Return:        {intraday_metrics['total_return']:>10.2%}")
        print(f"  Annualized Return:   {intraday_metrics['annualized_return']:>10.2%}")
        print(f"  Annualized Vol:      {intraday_metrics['volatility']:>10.2%}")
        print(f"  Sharpe Ratio:        {intraday_metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {intraday_metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:        {intraday_metrics['calmar_ratio']:>10.2f}")
        print(f"  Max Drawdown:        {intraday_metrics['max_drawdown']:>10.2%}")
        print(f"  Win Rate:            {intraday_metrics['win_rate']:>10.2%}")
        print(f"  Avg Win:             {intraday_metrics['avg_win']:>10.2%}")
        print(f"  Avg Loss:            {intraday_metrics['avg_loss']:>10.2%}")

        print(f"\nOVERNIGHT (Close to Open):")
        print(f"  Total Return:        {overnight_metrics['total_return']:>10.2%}")
        print(f"  Annualized Return:   {overnight_metrics['annualized_return']:>10.2%}")
        print(f"  Annualized Vol:      {overnight_metrics['volatility']:>10.2%}")
        print(f"  Sharpe Ratio:        {overnight_metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {overnight_metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:        {overnight_metrics['calmar_ratio']:>10.2f}")
        print(f"  Max Drawdown:        {overnight_metrics['max_drawdown']:>10.2%}")
        print(f"  Win Rate:            {overnight_metrics['win_rate']:>10.2%}")
        print(f"  Avg Win:             {overnight_metrics['avg_win']:>10.2%}")
        print(f"  Avg Loss:            {overnight_metrics['avg_loss']:>10.2%}")

        results.append({
            'period': period_name,
            'intraday': intraday_metrics,
            'overnight': overnight_metrics
        })

    # Create visualizations
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    plot_cumulative_returns(df, symbol, lookback_periods, timestamp)
    plot_combined_strategy(df, symbol, lookback_periods, timestamp)

    # Save risk metrics to file
    print("\nSaving risk metrics...")
    save_risk_metrics(results, symbol, timestamp)

    print("\n" + "=" * 80)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
