"""
Calculate SPY Beta for Test 2f Long-Only Full Reversal Strategy

Beta measures the systematic exposure to SPY:
- Beta = Cov(strategy_returns, spy_returns) / Var(spy_returns)
- Beta = 1.0 means same volatility and correlation as SPY
- Beta < 1.0 means lower systematic risk than SPY
- Beta > 1.0 means higher systematic risk than SPY
"""

import pandas as pd
import numpy as np

print("="*80)
print("SPY BETA CALCULATION - TEST 2f STRATEGY")
print("="*80)
print()

# Load the daily data
df = pd.read_csv('../data/test2f_daily_data.csv', index_col=0, parse_dates=True)

print(f"Loaded {len(df)} trading days")
print()

# Extract returns
spy_returns = df['spy_return'].dropna()
strategy_returns = df['strategy_return'].dropna()

# Align the data (in case of any mismatches)
aligned_df = pd.DataFrame({
    'spy_return': spy_returns,
    'strategy_return': strategy_returns
}).dropna()

print(f"Aligned data: {len(aligned_df)} observations")
print()

# Calculate beta
covariance = aligned_df['strategy_return'].cov(aligned_df['spy_return'])
spy_variance = aligned_df['spy_return'].var()
beta = covariance / spy_variance

# Calculate correlation for additional context
correlation = aligned_df['strategy_return'].corr(aligned_df['spy_return'])

# Calculate R-squared
r_squared = correlation ** 2

# Annualized statistics
strategy_vol = aligned_df['strategy_return'].std() * np.sqrt(252)
spy_vol = aligned_df['spy_return'].std() * np.sqrt(252)

print("="*80)
print("BETA ANALYSIS")
print("="*80)
print(f"\nSPY Beta:                    {beta:>10.4f}")
print(f"Correlation with SPY:        {correlation:>10.4f}")
print(f"R-squared:                   {r_squared:>10.4f}")
print()
print(f"Strategy Annualized Vol:     {strategy_vol:>10.2%}")
print(f"SPY Annualized Vol:          {spy_vol:>10.2%}")
print()

# Interpretation
print("="*80)
print("INTERPRETATION")
print("="*80)
print()
if beta < 0.3:
    print(f"The strategy has VERY LOW systematic exposure to SPY (beta = {beta:.4f})")
    print("This indicates the strategy is largely market-neutral or uncorrelated with SPY.")
elif beta < 0.7:
    print(f"The strategy has LOW to MODERATE systematic exposure to SPY (beta = {beta:.4f})")
    print("The strategy has meaningful diversification benefits relative to SPY.")
elif beta < 1.3:
    print(f"The strategy has MODERATE to HIGH systematic exposure to SPY (beta = {beta:.4f})")
    print("The strategy moves somewhat in line with SPY but with differences.")
else:
    print(f"The strategy has HIGH systematic exposure to SPY (beta = {beta:.4f})")
    print("The strategy is highly correlated with SPY movements.")

print()
print(f"R-squared of {r_squared:.2%} means {r_squared:.2%} of strategy variance is")
print(f"explained by SPY movements, and {(1-r_squared):.2%} is from other factors.")
print()

# Position analysis
print("="*80)
print("POSITION BREAKDOWN (for context)")
print("="*80)
print()
position_counts = df['position'].value_counts()
for pos, count in position_counts.items():
    pct = count / len(df) * 100
    print(f"  {pos:>10}: {count:>5} days ({pct:>5.1f}%)")
print()

# Expected beta from position exposure
spy_exposure_pct = position_counts.get('LONG_SPY', 0) / len(df)
tlt_exposure_pct = position_counts.get('LONG_TLT', 0) / len(df)

print(f"Expected beta from pure position exposure:")
print(f"  SPY exposure:    {spy_exposure_pct:>6.2%} (contributes ~{spy_exposure_pct*1.0:>5.3f} to beta)")
print(f"  TLT exposure:    {tlt_exposure_pct:>6.2%} (contributes ~{tlt_exposure_pct*0.0:>5.3f} to beta)")
print(f"  Naive estimate:  ~{spy_exposure_pct:>5.3f}")
print(f"  Actual beta:      {beta:>5.3f}")
print()

# Save results
results = pd.DataFrame({
    'metric': ['Beta', 'Correlation', 'R-squared', 'Strategy Vol', 'SPY Vol'],
    'value': [beta, correlation, r_squared, strategy_vol, spy_vol]
})
results.to_csv('../data/test2f_beta_analysis.csv', index=False)
print(f"Results saved to: ../data/test2f_beta_analysis.csv")
