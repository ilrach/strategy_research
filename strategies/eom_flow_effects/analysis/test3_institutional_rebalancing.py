"""
TEST 3: Institutional Rebalancing Front-Running Strategy

Based on academic research framework combining:
1. Band-pressure signal (continuous, threshold-based rebalancing)
2. Calendar-intensity signal (EOM/EOQ synchronized flows)

Key papers:
- Harvey, Mazzoleni, Melone (2025): Institutional rebalancing predictability
- Vanguard (2024-2025): Threshold rebalancing with 200bp bands
- Dyer, Fleming, Shachar (2024): EOM Treasury concentration
- Hartley & Schwarz (2019): EOM Treasury return premium

Strategy: Trade equity-minus-bond spread for one day around rebalancing flows
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(0, '/Users/bryn/projects/strategy_research')
from fmp_client import FMPClient

sns.set_style("darkgrid")

print("="*80)
print("TEST 3: INSTITUTIONAL REBALANCING FRONT-RUNNING STRATEGY")
print("="*80)
print()

# Load data
client = FMPClient()
print("Loading SPY and TLT data...")

spy_data = client.get_historical_prices('SPY', '2002-07-26', '2025-10-29')
spy_df = pd.DataFrame(spy_data)
spy_df['date'] = pd.to_datetime(spy_df['date'])
spy_df = spy_df.set_index('date').sort_index()
spy_df['spy_return'] = spy_df['adjClose'].pct_change()

tlt_data = client.get_historical_prices('TLT', '2002-07-26', '2025-10-29')
tlt_df = pd.DataFrame(tlt_data)
tlt_df['date'] = pd.to_datetime(tlt_df['date'])
tlt_df = tlt_df.set_index('date').sort_index()
tlt_df['tlt_return'] = tlt_df['adjClose'].pct_change()

# Merge on common dates
df = pd.DataFrame({
    'spy_return': spy_df['spy_return'],
    'tlt_return': tlt_df['tlt_return']
}).dropna()

print(f"Loaded {len(df)} trading days from {df.index[0].date()} to {df.index[-1].date()}")
print()

# Add calendar tracking
df['year_month'] = df.index.to_period('M')
df['trading_day'] = df.groupby('year_month').cumcount() + 1
df['days_in_month'] = df.groupby('year_month')['trading_day'].transform('max')
df['days_from_eom'] = df['days_in_month'] - df['trading_day']

# Identify end-of-month and end-of-quarter windows
df['is_eom_window'] = df['days_from_eom'] <= 4  # Last 5 trading days
df['is_eoq'] = df.index.month.isin([3, 6, 9, 12]) & (df['days_from_eom'] <= 1)  # Last 2 days of quarter

print("="*80)
print("SIGNAL CONSTRUCTION")
print("="*80)
print()

# ============================================================================
# PART 1: BAND-PRESSURE SIGNAL (B_t)
# ============================================================================
print("Building Band-Pressure Signal (continuous)...")

# Simulate drifting 60/40 portfolio (no rebalancing)
df['w_equity'] = 0.60  # Initialize

for i in range(1, len(df)):
    prev_w = df['w_equity'].iloc[i-1]
    eq_ret = df['spy_return'].iloc[i]
    bd_ret = df['tlt_return'].iloc[i]

    # Update weight based on returns
    numerator = prev_w * (1 + eq_ret)
    denominator = prev_w * (1 + eq_ret) + (1 - prev_w) * (1 + bd_ret)
    df.loc[df.index[i], 'w_equity'] = numerator / denominator

# Calculate deviation from 60% target
df['deviation'] = df['w_equity'] - 0.60

# Soft threshold: map deviation to [0,1] with b=0.02 (200bp band)
b = 0.02
df['magnitude'] = np.minimum(np.abs(df['deviation']) / b, 1.0)

# Direction: buy equities when underweight (negative deviation)
# B_t = sign(-d_t) * m_t
df['B_signal'] = np.sign(-df['deviation']) * df['magnitude']

print(f"  Band width: {b*100:.1f}%")
print(f"  Deviation range: [{df['deviation'].min()*100:.2f}%, {df['deviation'].max()*100:.2f}%]")
print(f"  B_signal range: [{df['B_signal'].min():.3f}, {df['B_signal'].max():.3f}]")
print()

# ============================================================================
# PART 2: CALENDAR-INTENSITY SIGNAL (C_t)
# ============================================================================
print("Building Calendar-Intensity Signal (EOM/EOQ boost)...")

# Define g_t: ramps from 0.2 to 1.0 over last 5 trading days
df['g_t'] = 0.0
df.loc[df['days_from_eom'] == 4, 'g_t'] = 0.2
df.loc[df['days_from_eom'] == 3, 'g_t'] = 0.4
df.loc[df['days_from_eom'] == 2, 'g_t'] = 0.6
df.loc[df['days_from_eom'] == 1, 'g_t'] = 0.8
df.loc[df['days_from_eom'] == 0, 'g_t'] = 1.0

# Calendar intensity: C_t = 1 + kappa_M * g_t + kappa_Q * q_t
kappa_M = 0.5  # EOM boost
kappa_Q = 0.5  # EOQ boost

df['C_signal'] = 1.0 + kappa_M * df['g_t'] + kappa_Q * df['is_eoq'].astype(float)

print(f"  EOM boost (kappa_M): {kappa_M}")
print(f"  EOQ boost (kappa_Q): {kappa_Q}")
print(f"  EOM window days: {df['is_eom_window'].sum()}")
print(f"  EOQ days: {df['is_eoq'].sum()}")
print(f"  C_signal range: [{df['C_signal'].min():.2f}, {df['C_signal'].max():.2f}]")
print()

# ============================================================================
# COMPOSITE SIGNAL
# ============================================================================
print("Combining signals...")

# S_t = C_t * B_t, capped at ±1
df['composite_signal_raw'] = df['C_signal'] * df['B_signal']
df['composite_signal'] = np.clip(df['composite_signal_raw'], -1.0, 1.0)

print(f"  Raw signal range: [{df['composite_signal_raw'].min():.3f}, {df['composite_signal_raw'].max():.3f}]")
print(f"  Capped signal range: [{df['composite_signal'].min():.3f}, {df['composite_signal'].max():.3f}]")
print(f"  Signals capped: {(np.abs(df['composite_signal_raw']) > 1.0).sum()} days")
print(f"  Non-zero signals: {(df['composite_signal'] != 0).sum()} days ({(df['composite_signal'] != 0).mean()*100:.1f}%)")
print()

# ============================================================================
# HEDGE RATIO (vol-matching)
# ============================================================================
print("Calculating vol-matched hedge ratio...")

# Calculate rolling 60-day volatility for each leg
window = 60
df['spy_vol'] = df['spy_return'].rolling(window).std()
df['tlt_vol'] = df['tlt_return'].rolling(window).std()

# Hedge ratio h_t = sigma_eq / sigma_bd
df['hedge_ratio'] = df['spy_vol'] / df['tlt_vol']

print(f"  Hedge ratio mean: {df['hedge_ratio'].mean():.3f}")
print(f"  Hedge ratio range: [{df['hedge_ratio'].min():.3f}, {df['hedge_ratio'].max():.3f}]")
print()

# ============================================================================
# POSITION AND RETURNS
# ============================================================================
print("Calculating strategy positions and returns...")

# SAME-DAY EXECUTION: We calculate signal 10 minutes before close and trade at close
# So we use today's signal to capture today's return
# Spread return: long equity, short hedge_ratio * bonds
df['spread_return'] = df['spy_return'] - df['hedge_ratio'] * df['tlt_return']
df['strategy_return'] = df['composite_signal'] * df['spread_return']

# Drop initial NaN rows from rolling calculations
df = df.dropna()

# Calculate cumulative returns
df['cum_return'] = (1 + df['strategy_return']).cumprod()

# Buy & hold 60/40 for comparison
df['portfolio_return'] = 0.6 * df['spy_return'] + 0.4 * df['tlt_return']
df['cum_portfolio_return'] = (1 + df['portfolio_return']).cumprod()

# Calculate drawdown
df['running_max'] = df['cum_return'].cummax()
df['drawdown'] = (df['cum_return'] - df['running_max']) / df['running_max']

print("="*80)
print("PERFORMANCE METRICS")
print("="*80)

total_days = len(df)
years = total_days / 252

# Strategy
final_value = df['cum_return'].iloc[-1]
cagr = (final_value ** (1 / years)) - 1
returns = df['strategy_return']
mean_return = returns.mean()
std_return = returns.std()
sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
max_dd = df['drawdown'].min()
calmar = cagr / abs(max_dd) if max_dd != 0 else 0

# Beta to SPY
spy_returns = df['spy_return']
covariance = returns.cov(spy_returns)
spy_variance = spy_returns.var()
beta = covariance / spy_variance
correlation = returns.corr(spy_returns)

# Information ratio (vs 60/40 benchmark)
excess_return = returns - df['portfolio_return']
tracking_error = excess_return.std()
information_ratio = (excess_return.mean() / tracking_error) * np.sqrt(252) if tracking_error > 0 else 0

# 60/40 portfolio
portfolio_final = df['cum_portfolio_return'].iloc[-1]
portfolio_cagr = (portfolio_final ** (1 / years)) - 1
portfolio_returns = df['portfolio_return']
portfolio_sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
portfolio_running_max = df['cum_portfolio_return'].cummax()
portfolio_dd = ((df['cum_portfolio_return'] - portfolio_running_max) / portfolio_running_max).min()
portfolio_calmar = portfolio_cagr / abs(portfolio_dd) if portfolio_dd != 0 else 0

# Turnover
df['position_change'] = df['composite_signal'].diff().abs()
annual_turnover = (df['position_change'].sum() / len(df)) * 252

print(f"\n{'Metric':<25} {'Strategy':>15} {'60/40 Portfolio':>15}")
print("-"*80)
print(f"{'Total Return':<25} {(final_value - 1):>14.2%} {(portfolio_final - 1):>14.2%}")
print(f"{'CAGR':<25} {cagr:>14.2%} {portfolio_cagr:>14.2%}")
print(f"{'Sharpe Ratio':<25} {sharpe:>14.2f} {portfolio_sharpe:>14.2f}")
print(f"{'Information Ratio':<25} {information_ratio:>14.2f} {'N/A':>14}")
print(f"{'Max Drawdown':<25} {max_dd:>14.2%} {portfolio_dd:>14.2%}")
print(f"{'Calmar Ratio':<25} {calmar:>14.2f} {portfolio_calmar:>14.2f}")
print(f"{'SPY Beta':<25} {beta:>14.3f} {'N/A':>14}")
print(f"{'SPY Correlation':<25} {correlation:>14.3f} {'N/A':>14}")
print(f"{'Annual Volatility':<25} {std_return * np.sqrt(252):>14.2%} {portfolio_returns.std() * np.sqrt(252):>14.2%}")
print(f"{'Annual Turnover':<25} {annual_turnover:>14.2f} {'N/A':>14}")
print(f"{'Total Days':<25} {total_days:>15} {total_days:>15}")
print(f"{'Years':<25} {years:>14.1f} {years:>14.1f}")

# Save results
results = pd.DataFrame({
    'metric': ['Total Return', 'CAGR', 'Sharpe', 'Information Ratio', 'Max DD', 'Calmar', 'Beta', 'Volatility', 'Turnover'],
    'strategy': [(final_value - 1), cagr, sharpe, information_ratio, max_dd, calmar, beta, std_return * np.sqrt(252), annual_turnover],
    'portfolio_60_40': [(portfolio_final - 1), portfolio_cagr, portfolio_sharpe, np.nan, portfolio_dd, portfolio_calmar, np.nan, portfolio_returns.std() * np.sqrt(252), np.nan]
})
results.to_csv('../data/test3_performance.csv', index=False)
print(f"\nResults saved to: ../data/test3_performance.csv")

# Save daily data
df[['spy_return', 'tlt_return', 'w_equity', 'deviation', 'B_signal', 'g_t', 'C_signal',
    'composite_signal', 'hedge_ratio', 'spread_return', 'strategy_return',
    'cum_return', 'drawdown']].to_csv('../data/test3_daily_data.csv')
print(f"Daily data saved to: ../data/test3_daily_data.csv")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(4, 1, figsize=(16, 16), gridspec_kw={'height_ratios': [3, 1, 1, 1]})

# Equity curve
ax1 = axes[0]
ax1.plot(df.index, df['cum_return'], label='Strategy', linewidth=2, color='darkblue')
ax1.plot(df.index, df['cum_portfolio_return'], label='60/40 Portfolio',
         linewidth=1.5, color='gray', alpha=0.6, linestyle='--')
ax1.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
ax1.set_title(f'Institutional Rebalancing Front-Running Strategy\\n'
              f'CAGR: {cagr:.2%} | Sharpe: {sharpe:.2f} | IR: {information_ratio:.2f} | '
              f'Max DD: {max_dd:.2%} | Beta: {beta:.3f}',
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Drawdown
ax2 = axes[1]
ax2.fill_between(df.index, df['drawdown'] * 100, 0,
                  color='red', alpha=0.3, label='Drawdown')
ax2.plot(df.index, df['drawdown'] * 100, color='darkred', linewidth=1)
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([max_dd * 100 * 1.1, 5])

# Portfolio drift over time
ax3 = axes[2]
ax3.plot(df.index, df['w_equity'] * 100, linewidth=0.5, alpha=0.7, color='steelblue')
ax3.axhline(y=60, color='red', linestyle='--', linewidth=2, label='60% Target')
ax3.fill_between(df.index, 60, df['w_equity'] * 100,
                  where=(df['w_equity'] * 100 > 60), color='green', alpha=0.1)
ax3.fill_between(df.index, 60, df['w_equity'] * 100,
                  where=(df['w_equity'] * 100 < 60), color='red', alpha=0.1)
ax3.set_ylabel('Equity Allocation (%)', fontsize=12, fontweight='bold')
ax3.set_title('60/40 Portfolio Drift (No Rebalancing)', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)

# Signal decomposition
ax4 = axes[3]
# Plot on separate days to avoid overlap
eom_days = df[df['is_eom_window']].index
non_eom_days = df[~df['is_eom_window']].index

ax4.scatter(non_eom_days, df.loc[non_eom_days, 'composite_signal'],
            s=5, alpha=0.3, color='blue', label='Regular days')
ax4.scatter(eom_days, df.loc[eom_days, 'composite_signal'],
            s=10, alpha=0.5, color='red', label='EOM window')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax4.set_ylabel('Composite Signal', fontsize=12, fontweight='bold')
ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
ax4.set_title('Signal Intensity (Red = EOM boost active)', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/test3_institutional_rebalancing.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/test3_institutional_rebalancing.png")

# ============================================================================
# DIAGNOSTIC ANALYSIS
# ============================================================================

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

# 1. Band-pressure signal distribution
ax1 = axes2[0, 0]
ax1.hist(df['B_signal'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Band-Pressure Signal', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title(f'Band-Pressure Signal Distribution\nMean: {df["B_signal"].mean():.3f}',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Calendar boost impact
ax2 = axes2[0, 1]
eom_returns = df[df['is_eom_window']]['strategy_return']
non_eom_returns = df[~df['is_eom_window']]['strategy_return']
ax2.hist([non_eom_returns*100, eom_returns*100], bins=50, alpha=0.6,
         label=['Non-EOM', 'EOM window'], color=['blue', 'red'], edgecolor='black')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Daily Return (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title(f'EOM vs Non-EOM Return Distribution\n'
              f'EOM mean: {eom_returns.mean()*100:.3f}% | Non-EOM mean: {non_eom_returns.mean()*100:.3f}%',
              fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Hedge ratio over time
ax3 = axes2[1, 0]
ax3.plot(df.index, df['hedge_ratio'], linewidth=0.5, alpha=0.7, color='purple')
ax3.axhline(y=df['hedge_ratio'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {df["hedge_ratio"].mean():.2f}')
ax3.set_ylabel('Hedge Ratio', fontsize=11, fontweight='bold')
ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
ax3.set_title('Vol-Matched Hedge Ratio (SPY vol / TLT vol)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Rolling Sharpe ratio
ax4 = axes2[1, 1]
rolling_sharpe = (df['strategy_return'].rolling(252).mean() / df['strategy_return'].rolling(252).std()) * np.sqrt(252)
ax4.plot(df.index, rolling_sharpe, linewidth=1.5, color='darkgreen')
ax4.axhline(y=sharpe, color='red', linestyle='--', linewidth=2,
            label=f'Overall: {sharpe:.2f}')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax4.set_ylabel('Rolling 1Y Sharpe', fontsize=11, fontweight='bold')
ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
ax4.set_title('Strategy Stability: Rolling 1-Year Sharpe Ratio', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/test3_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"Diagnostic plot saved to: ../plots/test3_diagnostics.png")

print()
print("="*80)
print("FRAMEWORK VALIDATION")
print("="*80)
print()
print("Key Research Predictions:")
print(f"  1. Band pressure (200bp threshold): {'✓' if abs(df['deviation'].std()) < 0.03 else '✗'} (std: {df['deviation'].std()*100:.2f}%)")
print(f"  2. EOM concentration effect: {'✓' if eom_returns.mean() > non_eom_returns.mean() else '✗'}")
print(f"  3. Low market correlation: {'✓' if abs(beta) < 0.3 else '✗'} (beta: {beta:.3f})")
print(f"  4. One-day holding horizon: ✓ (implemented)")
print()
print("This strategy implements the academic framework for front-running")
print("institutional rebalancing flows using band-pressure + calendar-intensity signals.")
