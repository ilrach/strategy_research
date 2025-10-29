"""
Trendiness Analysis for Leveraged ETFs

Score each underlying (bull side) by how trending vs mean-reverting it is.
Use multiple academic measures:

1. Hurst Exponent: H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random walk
2. Autocorrelation: Positive = trending, negative = mean-reverting
3. Average Directional Index (ADX): Measures trend strength
4. Efficiency Ratio: (net move) / (total move) - higher = more trending

Theory: More trending underlyings should have MORE vol decay in their leveraged ETFs,
making them better candidates for double shorting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient

sns.set_style("darkgrid")

# Define pairs
PAIRS = [
    ('TQQQ', 'SQQQ', 3),
    ('SOXL', 'SOXS', 3),
    ('SPXL', 'SPXS', 3),
    ('UPRO', 'SPXU', 3),
    ('TNA', 'TZA', 3),
    ('TECL', 'TECS', 3),
    ('QLD', 'QID', 2),
    ('SSO', 'SDS', 2),
    ('TMF', 'TMV', 3),
    ('NUGT', 'DUST', 2),
    ('YINN', 'YANG', 3),
    ('BOIL', 'KOLD', 2),
    ('UCO', 'SCO', 2),
]

client = FMPClient()

def calculate_hurst_exponent(prices, lags=range(2, 100)):
    """
    Calculate Hurst exponent using R/S analysis.
    H > 0.5: trending (persistent)
    H < 0.5: mean-reverting (anti-persistent)
    H = 0.5: random walk
    """
    tau = []
    lagvec = []

    # Step through lags
    for lag in lags:
        # Calculate the lag difference
        pp = np.subtract(prices[lag:], prices[:-lag])

        # Calculate the variance
        lagvec.append(lag)
        tau.append(np.std(pp))

    # Linear fit to log-log plot
    try:
        m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)
        hurst = m[0]
        return hurst
    except:
        return np.nan

def calculate_autocorrelation(returns, lag=1):
    """
    Calculate autocorrelation of returns at given lag.
    Positive = trending, negative = mean-reverting
    """
    if len(returns) < lag + 10:
        return np.nan
    return returns.autocorr(lag=lag)

def calculate_efficiency_ratio(prices, period=20):
    """
    Kaufman Efficiency Ratio: (net change) / (sum of absolute changes)
    Range [0, 1]: 1 = perfectly efficient trend, 0 = random noise
    """
    net_change = abs(prices.diff(period))
    total_change = prices.diff().abs().rolling(period).sum()
    er = net_change / total_change
    return er.mean()

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index: measures trend strength
    < 20 = weak/no trend
    20-40 = moderate trend
    > 40 = strong trend
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    # Directional Movement
    up = high - high.shift()
    down = low.shift() - low

    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()

    return adx.mean()

def calculate_trend_consistency(returns, window=20):
    """
    What % of rolling windows are positive/negative consistently?
    Higher = more consistent trending
    """
    rolling_sum = returns.rolling(window).sum()
    positive_runs = (rolling_sum > 0).sum()
    negative_runs = (rolling_sum < 0).sum()
    total_runs = len(rolling_sum.dropna())

    # Consistency is how far from 50/50 we are
    consistency = abs(positive_runs - negative_runs) / total_runs if total_runs > 0 else 0
    return consistency

print("="*80)
print("TRENDINESS ANALYSIS")
print("="*80)
print("Analyzing trendiness of underlying price series (bull side)")
print()

results = []

for bull, bear, leverage in PAIRS:
    print(f"Analyzing {bull}...")

    # Load prices
    data = pd.DataFrame(client.get_historical_prices(bull, '2015-01-01', '2025-10-29'))
    if len(data) == 0:
        print(f"  No data")
        continue

    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date').sort_index()

    # Extract OHLC (check if columns exist, otherwise skip ADX)
    prices = data['adjClose']
    has_ohlc = 'high' in data.columns and 'low' in data.columns

    if has_ohlc:
        high = data['high']
        low = data['low']
        close = data['adjClose']
    else:
        high = None
        low = None
        close = prices

    # Calculate returns
    returns = prices.pct_change().dropna()

    if len(prices) < 252:
        print(f"  Insufficient data")
        continue

    # Calculate metrics
    hurst = calculate_hurst_exponent(prices.values)
    autocorr_1 = calculate_autocorrelation(returns, lag=1)
    autocorr_5 = calculate_autocorrelation(returns, lag=5)
    autocorr_20 = calculate_autocorrelation(returns, lag=20)
    efficiency = calculate_efficiency_ratio(prices, period=20)
    adx = calculate_adx(high, low, close, period=14) if has_ohlc else np.nan
    trend_consistency = calculate_trend_consistency(returns, window=20)

    # Also calculate volatility for reference
    vol_30d = returns.rolling(30).std().mean() * np.sqrt(252) * 100

    results.append({
        'ticker': bull,
        'leverage': leverage,
        'hurst': hurst,
        'autocorr_1d': autocorr_1,
        'autocorr_5d': autocorr_5,
        'autocorr_20d': autocorr_20,
        'efficiency_ratio': efficiency,
        'adx': adx,
        'trend_consistency': trend_consistency,
        'avg_vol_30d': vol_30d,
        'days': len(prices)
    })

    print(f"  Hurst: {hurst:.3f}, AutoCorr(1d): {autocorr_1:+.3f}, "
          f"Efficiency: {efficiency:.3f}, ADX: {adx:.1f}")

# Create results dataframe
df = pd.DataFrame(results)

# Create composite trendiness score (normalize each metric 0-1, then average)
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

df['hurst_norm'] = normalize(df['hurst'])
df['autocorr_norm'] = normalize(df['autocorr_1d'])
df['efficiency_norm'] = normalize(df['efficiency_ratio'])
df['consistency_norm'] = normalize(df['trend_consistency'])

# ADX may have NaNs, handle separately
if df['adx'].notna().sum() > 0:
    df['adx_norm'] = normalize(df['adx'].fillna(df['adx'].mean()))
    n_metrics = 5
else:
    df['adx_norm'] = 0
    n_metrics = 4

# Composite score (equal weighted average)
if n_metrics == 5:
    df['trendiness_score'] = (
        df['hurst_norm'] +
        df['autocorr_norm'] +
        df['efficiency_norm'] +
        df['adx_norm'] +
        df['consistency_norm']
    ) / 5
else:
    df['trendiness_score'] = (
        df['hurst_norm'] +
        df['autocorr_norm'] +
        df['efficiency_norm'] +
        df['consistency_norm']
    ) / 4

df = df.sort_values('trendiness_score', ascending=False)

print("\n" + "="*80)
print("TRENDINESS RANKINGS")
print("="*80)
print(f"\n{'Rank':<5} {'Ticker':<8} {'Score':>6} {'Hurst':>7} {'AC(1d)':>8} {'Effic':>7} {'ADX':>6} {'Consist':>8}")
print("-"*80)

for i, row in df.iterrows():
    rank = list(df.index).index(i) + 1
    print(f"{rank:<5} {row['ticker']:<8} {row['trendiness_score']:>6.3f} "
          f"{row['hurst']:>7.3f} {row['autocorr_1d']:>+8.3f} "
          f"{row['efficiency_ratio']:>7.3f} {row['adx']:>6.1f} "
          f"{row['trend_consistency']:>8.3f}")

# Summary statistics
print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("\nHurst Exponent:")
print(f"  Mean: {df['hurst'].mean():.3f}")
print(f"  > 0.5 (trending): {(df['hurst'] > 0.5).sum()}/{len(df)}")
print(f"  < 0.5 (mean-reverting): {(df['hurst'] < 0.5).sum()}/{len(df)}")

print("\n1-Day Autocorrelation:")
print(f"  Mean: {df['autocorr_1d'].mean():+.3f}")
print(f"  Positive (trending): {(df['autocorr_1d'] > 0).sum()}/{len(df)}")
print(f"  Negative (mean-reverting): {(df['autocorr_1d'] < 0).sum()}/{len(df)}")

print("\nEfficiency Ratio:")
print(f"  Mean: {df['efficiency_ratio'].mean():.3f}")
print(f"  Range: [{df['efficiency_ratio'].min():.3f}, {df['efficiency_ratio'].max():.3f}]")

print("\nADX (Trend Strength):")
print(f"  Mean: {df['adx'].mean():.1f}")
print(f"  Strong trend (>40): {(df['adx'] > 40).sum()}/{len(df)}")
print(f"  Moderate trend (20-40): {((df['adx'] >= 20) & (df['adx'] <= 40)).sum()}/{len(df)}")
print(f"  Weak trend (<20): {(df['adx'] < 20).sum()}/{len(df)}")

# Save results
df.to_csv('../data/trendiness_analysis.csv', index=False)
print(f"\nResults saved to: ../data/trendiness_analysis.csv")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Overall trendiness score
ax1 = axes[0, 0]
df.sort_values('trendiness_score').plot(x='ticker', y='trendiness_score', kind='barh',
                                         ax=ax1, legend=False, color='steelblue')
ax1.set_xlabel('Composite Trendiness Score', fontsize=12)
ax1.set_ylabel('Ticker', fontsize=12)
ax1.set_title('Overall Trendiness Ranking', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Hurst exponent
ax2 = axes[0, 1]
df.sort_values('hurst').plot(x='ticker', y='hurst', kind='barh',
                             ax=ax2, legend=False, color='darkorange')
ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Random Walk (0.5)')
ax2.set_xlabel('Hurst Exponent', fontsize=12)
ax2.set_ylabel('Ticker', fontsize=12)
ax2.set_title('Hurst Exponent (>0.5 = Trending)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Autocorrelation
ax3 = axes[0, 2]
df.sort_values('autocorr_1d').plot(x='ticker', y='autocorr_1d', kind='barh',
                                   ax=ax3, legend=False, color='green')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('1-Day Autocorrelation', fontsize=12)
ax3.set_ylabel('Ticker', fontsize=12)
ax3.set_title('Return Autocorrelation (>0 = Trending)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Efficiency ratio
ax4 = axes[1, 0]
df.sort_values('efficiency_ratio').plot(x='ticker', y='efficiency_ratio', kind='barh',
                                        ax=ax4, legend=False, color='purple')
ax4.set_xlabel('Efficiency Ratio', fontsize=12)
ax4.set_ylabel('Ticker', fontsize=12)
ax4.set_title('Kaufman Efficiency Ratio (Higher = More Trending)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# ADX
ax5 = axes[1, 1]
df.sort_values('adx').plot(x='ticker', y='adx', kind='barh',
                           ax=ax5, legend=False, color='crimson')
ax5.axvline(x=20, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax5.axvline(x=40, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax5.set_xlabel('ADX', fontsize=12)
ax5.set_ylabel('Ticker', fontsize=12)
ax5.set_title('Average Directional Index (Higher = Stronger Trend)', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Scatter: Trendiness vs Volatility
ax6 = axes[1, 2]
ax6.scatter(df['trendiness_score'], df['avg_vol_30d'], s=100, alpha=0.6, color='steelblue')
for i, row in df.iterrows():
    ax6.annotate(row['ticker'], (row['trendiness_score'], row['avg_vol_30d']),
                fontsize=9, alpha=0.8)
ax6.set_xlabel('Trendiness Score', fontsize=12)
ax6.set_ylabel('Avg 30d Vol (%)', fontsize=12)
ax6.set_title('Trendiness vs Volatility', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/trendiness_analysis.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ../plots/trendiness_analysis.png")
