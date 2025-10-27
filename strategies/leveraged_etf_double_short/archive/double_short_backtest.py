"""
Double Short Leveraged ETF Pairs Strategy Backtest

Strategy: Equal weight short both the bull and bear ETF in each pair
Rebalancing: When allocation deviates by 10% from target weight (relative to pair allocation)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Add parent directory to path to import fmp_client
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

# Set style for plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Leveraged ETF pairs to trade (filtered for >$5M volume from analysis)
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
    ('DRN', 'DRV'),
    ('ERX', 'ERY'),
    ('USD', 'SSG'),
    ('WEBL', 'WEBS'),
    ('DPST', 'WDRW'),  # Note: WDRW may have insufficient data
    ('AGQ', 'ZSL'),
    ('UGL', 'GLL'),
]


class DoubleShortBacktest:
    """Backtest double short strategy on leveraged ETF pairs."""

    def __init__(self, initial_capital: float = 100000, rebalance_frequency: str = 'weekly', cash_rate: float = 0.045):
        """
        Initialize backtest.

        Args:
            initial_capital: Starting capital in USD
            rebalance_frequency: 'weekly', 'monthly', or 'daily'
            cash_rate: Annual interest rate on cash balance (default 4.5%)
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.cash_rate = cash_rate
        self.client = FMPClient()

    def fetch_pair_data(self, bull: str, bear: str, from_date: str, to_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch historical data for a pair."""
        try:
            bull_data = self.client.get_historical_prices(bull, from_date=from_date, to_date=to_date)
            bear_data = self.client.get_historical_prices(bear, from_date=from_date, to_date=to_date)

            if not bull_data or not bear_data:
                return None, None

            bull_df = pd.DataFrame(bull_data)
            bear_df = pd.DataFrame(bear_data)

            # Use adjClose as price
            bull_df['price'] = bull_df['adjClose']
            bear_df['price'] = bear_df['adjClose']

            bull_df['date'] = pd.to_datetime(bull_df['date'])
            bear_df['date'] = pd.to_datetime(bear_df['date'])

            bull_df = bull_df[['date', 'price']].sort_values('date').set_index('date')
            bear_df = bear_df[['date', 'price']].sort_values('date').set_index('date')

            return bull_df, bear_df
        except Exception as e:
            print(f"Error fetching {bull}/{bear}: {e}")
            return None, None

    def backtest_pair(self, bull: str, bear: str, prices_bull: pd.DataFrame, prices_bear: pd.DataFrame) -> Dict:
        """
        Backtest a single pair with double short strategy.

        Strategy: Short 50% in bull, 50% in bear. Rebalance when allocation deviates by threshold.
        """
        # Merge on dates (inner join - only common trading days)
        df = prices_bull.join(prices_bear, how='inner', lsuffix='_bull', rsuffix='_bear')
        df.columns = ['price_bull', 'price_bear']

        if len(df) < 2:
            return None

        # Initialize positions
        target_weight = 0.5  # 50% each in the pair
        starting_capital = self.initial_capital  # Save for this pair

        # Short positions: negative shares
        # Allocate 50% to short bull, 50% to short bear
        initial_price_bull = df.iloc[0]['price_bull']
        initial_price_bear = df.iloc[0]['price_bear']

        shares_bull = -(starting_capital * target_weight) / initial_price_bull
        shares_bear = -(starting_capital * target_weight) / initial_price_bear

        # Initial position values (negative)
        initial_value_bull = shares_bull * initial_price_bull
        initial_value_bear = shares_bear * initial_price_bear

        # Track equity, positions, and rebalances
        equity_curve = []
        rebalance_dates = []
        trade_count = 0
        last_rebalance_date = df.index[0]
        cumulative_interest = 0

        for i, (date, row) in enumerate(df.iterrows()):
            price_bull = row['price_bull']
            price_bear = row['price_bear']

            # Accrue interest on equity (conservative: only on equity, not on full cash balance)
            if i > 0:
                days_elapsed = (date - df.index[i-1]).days
                prev_equity = equity_curve[-1]['equity']
                interest = prev_equity * (self.cash_rate / 365) * days_elapsed
                cumulative_interest += interest

            # Calculate current position values (negative for shorts)
            current_value_bull = shares_bull * price_bull
            current_value_bear = shares_bear * price_bear

            # P&L on shorts: entry_value - current_value
            pnl_bull = initial_value_bull - current_value_bull
            pnl_bear = initial_value_bear - current_value_bear

            # Total equity = initial capital + P&L + accrued interest
            total_equity = self.initial_capital + pnl_bull + pnl_bear + cumulative_interest

            # Check if rebalance needed based on frequency
            needs_rebalance = False
            if self.rebalance_frequency == 'weekly':
                # Rebalance if it's been 7 days since last rebalance
                days_since_rebalance = (date - last_rebalance_date).days
                needs_rebalance = days_since_rebalance >= 7
            elif self.rebalance_frequency == 'monthly':
                # Rebalance if month has changed
                needs_rebalance = date.month != last_rebalance_date.month
            elif self.rebalance_frequency == 'daily':
                needs_rebalance = True

            if needs_rebalance and total_equity > 0 and i > 0:  # Don't rebalance on first day
                # Rebalance: adjust positions to 50/50 based on current equity
                shares_bull = -(total_equity * target_weight) / price_bull
                shares_bear = -(total_equity * target_weight) / price_bear

                initial_value_bull = shares_bull * price_bull
                initial_value_bear = shares_bear * price_bear

                # Reset cumulative interest after incorporating it into equity
                cumulative_interest = 0

                rebalance_dates.append(date)
                last_rebalance_date = date
                trade_count += 1

            equity_curve.append({
                'date': date,
                'equity': total_equity,
                'shares_bull': shares_bull,
                'shares_bear': shares_bear,
                'price_bull': price_bull,
                'price_bear': price_bear
            })

        # Convert to DataFrame
        equity_df = pd.DataFrame(equity_curve).set_index('date')

        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()

        # Performance metrics
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital - 1)

        # Annualized metrics
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365.25
        cagr = (equity_df['equity'].iloc[-1] / self.initial_capital) ** (1/years) - 1 if years > 0 else 0

        # Volatility (annualized)
        volatility = equity_df['returns'].std() * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = (equity_df['returns'].mean() * 252) / (equity_df['returns'].std() * np.sqrt(252)) if equity_df['returns'].std() > 0 else 0

        # Max drawdown
        cumulative = (1 + equity_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_days = (equity_df['returns'] > 0).sum()
        total_days = len(equity_df['returns'].dropna())
        win_rate = winning_days / total_days if total_days > 0 else 0

        return {
            'pair': f"{bull}/{bear}",
            'bull': bull,
            'bear': bear,
            'equity_curve': equity_df,
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': trade_count,
            'num_rebalances': len(rebalance_dates),
            'start_date': equity_df.index[0],
            'end_date': equity_df.index[-1],
            'days': days
        }

    def backtest_all_pairs(self, from_date: str, to_date: str) -> Dict:
        """Backtest all pairs and create equal-weight portfolio."""
        print(f"Backtesting from {from_date} to {to_date}")
        print(f"Initial capital: ${self.initial_capital:,.0f}")
        print(f"Rebalance frequency: {self.rebalance_frequency}")
        print(f"Cash interest rate: {self.cash_rate*100:.2f}% annually")
        print()

        pair_results = []

        for i, (bull, bear) in enumerate(LEVERAGED_PAIRS):
            print(f"[{i+1}/{len(LEVERAGED_PAIRS)}] Backtesting {bull}/{bear}...")

            # Fetch data
            bull_df, bear_df = self.fetch_pair_data(bull, bear, from_date, to_date)

            if bull_df is None or bear_df is None:
                print(f"  Skipping {bull}/{bear} - insufficient data")
                continue

            # Backtest pair
            result = self.backtest_pair(bull, bear, bull_df, bear_df)

            if result:
                pair_results.append(result)
                print(f"  Return: {result['total_return']*100:.2f}% | Sharpe: {result['sharpe']:.2f} | MDD: {result['max_drawdown']*100:.2f}%")
            else:
                print(f"  Skipping {bull}/{bear} - backtest failed")

        print(f"\nSuccessfully backtested {len(pair_results)} pairs")

        # Create equal-weight portfolio
        if pair_results:
            portfolio_equity = self.create_equal_weight_portfolio(pair_results)
            return {
                'pair_results': pair_results,
                'portfolio': portfolio_equity
            }
        else:
            return {'pair_results': [], 'portfolio': None}

    def create_equal_weight_portfolio(self, pair_results: List[Dict]) -> Dict:
        """Create equal-weight portfolio from all pair results."""
        print("\nCreating equal-weight portfolio...")

        # Get all equity curves aligned by date
        equity_curves = []
        for result in pair_results:
            ec = result['equity_curve'][['equity']].copy()
            ec.columns = [result['pair']]
            equity_curves.append(ec)

        # Merge all equity curves (inner join - only common dates)
        portfolio_df = equity_curves[0]
        for ec in equity_curves[1:]:
            portfolio_df = portfolio_df.join(ec, how='inner')

        # Calculate portfolio equity as average of all pairs
        # Each pair starts with initial_capital, so average gives equal weight
        portfolio_df['portfolio_equity'] = portfolio_df.mean(axis=1)

        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_equity'].pct_change()

        # Performance metrics
        total_return = (portfolio_df['portfolio_equity'].iloc[-1] / self.initial_capital - 1)

        days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        years = days / 365.25
        cagr = (portfolio_df['portfolio_equity'].iloc[-1] / self.initial_capital) ** (1/years) - 1 if years > 0 else 0

        volatility = portfolio_df['returns'].std() * np.sqrt(252)
        sharpe = (portfolio_df['returns'].mean() * 252) / (portfolio_df['returns'].std() * np.sqrt(252)) if portfolio_df['returns'].std() > 0 else 0

        cumulative = (1 + portfolio_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        winning_days = (portfolio_df['returns'] > 0).sum()
        total_days = len(portfolio_df['returns'].dropna())
        win_rate = winning_days / total_days if total_days > 0 else 0

        return {
            'equity_curve': portfolio_df,
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'start_date': portfolio_df.index[0],
            'end_date': portfolio_df.index[-1],
            'days': days,
            'num_pairs': len(pair_results)
        }


def print_results(results: Dict):
    """Print detailed results."""
    print("\n" + "="*100)
    print("INDIVIDUAL PAIR RESULTS")
    print("="*100)

    # Create summary table
    summary_data = []
    for result in results['pair_results']:
        summary_data.append({
            'Pair': result['pair'],
            'Total Return': f"{result['total_return']*100:.2f}%",
            'CAGR': f"{result['cagr']*100:.2f}%",
            'Volatility': f"{result['volatility']*100:.2f}%",
            'Sharpe': f"{result['sharpe']:.2f}",
            'Max DD': f"{result['max_drawdown']*100:.2f}%",
            'Win Rate': f"{result['win_rate']*100:.1f}%",
            'Rebalances': result['num_rebalances'],
            'Days': result['days']
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    if results['portfolio']:
        print("\n" + "="*100)
        print("EQUAL-WEIGHT PORTFOLIO RESULTS")
        print("="*100)

        p = results['portfolio']
        print(f"Number of pairs: {p['num_pairs']}")
        print(f"Period: {p['start_date'].date()} to {p['end_date'].date()} ({p['days']} days)")
        print(f"\nTotal Return: {p['total_return']*100:.2f}%")
        print(f"CAGR: {p['cagr']*100:.2f}%")
        print(f"Volatility (annualized): {p['volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {p['sharpe']:.2f}")
        print(f"Maximum Drawdown: {p['max_drawdown']*100:.2f}%")
        print(f"Win Rate: {p['win_rate']*100:.2f}%")
        print("="*100)


def plot_results(results: Dict, save_dir: str = 'plots'):
    """Generate plots for results."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Portfolio equity curve
    if results['portfolio']:
        plt.figure(figsize=(14, 6))
        portfolio_df = results['portfolio']['equity_curve']
        plt.plot(portfolio_df.index, portfolio_df['portfolio_equity'], linewidth=2, label='Portfolio')
        plt.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        plt.title('Equal-Weight Portfolio Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/portfolio_equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_dir}/portfolio_equity_curve.png")

    # 2. Individual pair equity curves (grid)
    n_pairs = len(results['pair_results'])
    if n_pairs > 0:
        n_cols = 4
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_pairs > 1 else [axes]

        for i, result in enumerate(results['pair_results']):
            ax = axes[i]
            equity_curve = result['equity_curve']
            ax.plot(equity_curve.index, equity_curve['equity'], linewidth=1.5)
            ax.axhline(y=100000, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f"{result['pair']}\nReturn: {result['total_return']*100:.1f}% | Sharpe: {result['sharpe']:.2f}",
                        fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('')

        # Hide empty subplots
        for i in range(n_pairs, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/individual_pairs_equity_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_dir}/individual_pairs_equity_curves.png")

    # 3. Performance comparison bar charts
    if n_pairs > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        pairs = [r['pair'] for r in results['pair_results']]
        returns = [r['total_return']*100 for r in results['pair_results']]
        sharpes = [r['sharpe'] for r in results['pair_results']]
        vols = [r['volatility']*100 for r in results['pair_results']]
        mdds = [r['max_drawdown']*100 for r in results['pair_results']]

        # Sort by return
        sorted_idx = np.argsort(returns)[::-1]

        # Total Returns
        axes[0, 0].barh([pairs[i] for i in sorted_idx], [returns[i] for i in sorted_idx])
        axes[0, 0].set_title('Total Return by Pair', fontweight='bold')
        axes[0, 0].set_xlabel('Return (%)')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # Sharpe Ratios
        sorted_idx_sharpe = np.argsort(sharpes)[::-1]
        axes[0, 1].barh([pairs[i] for i in sorted_idx_sharpe], [sharpes[i] for i in sorted_idx_sharpe])
        axes[0, 1].set_title('Sharpe Ratio by Pair', fontweight='bold')
        axes[0, 1].set_xlabel('Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3, axis='x')

        # Volatility
        sorted_idx_vol = np.argsort(vols)
        axes[1, 0].barh([pairs[i] for i in sorted_idx_vol], [vols[i] for i in sorted_idx_vol])
        axes[1, 0].set_title('Volatility by Pair', fontweight='bold')
        axes[1, 0].set_xlabel('Volatility (%)')
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        # Max Drawdown
        sorted_idx_mdd = np.argsort(mdds)
        axes[1, 1].barh([pairs[i] for i in sorted_idx_mdd], [mdds[i] for i in sorted_idx_mdd])
        axes[1, 1].set_title('Maximum Drawdown by Pair', fontweight='bold')
        axes[1, 1].set_xlabel('Max Drawdown (%)')
        axes[1, 1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_dir}/performance_comparison.png")

    # 4. Drawdown chart for portfolio
    if results['portfolio']:
        plt.figure(figsize=(14, 6))
        portfolio_df = results['portfolio']['equity_curve']
        returns = portfolio_df['returns']
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        plt.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.3, color='red')
        plt.plot(drawdown.index, drawdown * 100, color='red', linewidth=1)
        plt.title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/portfolio_drawdown.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_dir}/portfolio_drawdown.png")


if __name__ == '__main__':
    # Run backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3 years

    backtest = DoubleShortBacktest(
        initial_capital=100000,
        rebalance_frequency='weekly'  # Rebalance weekly
    )

    results = backtest.backtest_all_pairs(
        from_date=start_date.strftime('%Y-%m-%d'),
        to_date=end_date.strftime('%Y-%m-%d')
    )

    # Print results
    print_results(results)

    # Generate plots
    plot_results(results, save_dir='../../plots/leveraged_etf')

    # Save detailed results to CSV
    if results['pair_results']:
        summary_data = []
        for result in results['pair_results']:
            summary_data.append({
                'pair': result['pair'],
                'total_return': result['total_return'],
                'cagr': result['cagr'],
                'volatility': result['volatility'],
                'sharpe': result['sharpe'],
                'max_drawdown': result['max_drawdown'],
                'win_rate': result['win_rate'],
                'num_rebalances': result['num_rebalances'],
                'days': result['days']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('backtest_results_3yr_weekly.csv', index=False)
        print(f"\nDetailed results saved to: backtest_results_3yr_weekly.csv")
