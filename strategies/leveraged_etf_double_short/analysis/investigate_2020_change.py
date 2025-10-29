"""
Investigate the 2020 regime change in leveraged ETF double short strategy
"""
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, '/Users/bryn/projects/strategy_research')

from fmp_client import FMPClient

# Load TEST 4 results
results = pd.read_csv('../data/test4_edge_weighted_results.csv')
print("Summary stats from TEST 4:")
print(f"Overall CAGR: {results['cagr'].iloc[0]:.2%}")
print(f"Overall Sharpe: {results['sharpe'].iloc[0]:.2f}")
print()

# Load the detailed daily results
client = FMPClient()

# Load pairs and calculate year-by-year performance
pairs = [
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

k_values = {2: 7.07, 3: 4.71}

# Load BIL
bil_data = client.get_historical_prices('BIL', '2015-01-01', '2025-10-29')
bil_df = pd.DataFrame(bil_data)
bil_df['date'] = pd.to_datetime(bil_df['date'])
bil_df = bil_df[['date', 'adjClose']].rename(columns={'adjClose': 'p_bil'}).set_index('date').sort_index()
bil_df['bil_ret'] = bil_df['p_bil'].pct_change()
bil_df = bil_df.dropna()

print(f"Loading {len(pairs)} pairs...")
pair_data = {}

for bull, bear, leverage in pairs:
    bull_prices = pd.DataFrame(client.get_historical_prices(bull, '2015-01-01', '2025-10-29'))
    bull_prices['date'] = pd.to_datetime(bull_prices['date'])
    bull_prices = bull_prices[['date', 'adjClose']].rename(columns={'adjClose': f'p_{bull}'}).set_index('date').sort_index()

    bear_prices = pd.DataFrame(client.get_historical_prices(bear, '2015-01-01', '2025-10-29'))
    bear_prices['date'] = pd.to_datetime(bear_prices['date'])
    bear_prices = bear_prices[['date', 'adjClose']].rename(columns={'adjClose': f'p_{bear}'}).set_index('date').sort_index()

    bull_borrow_file = f'../iborrowdesk_letf_rates/{bull}.csv'
    bear_borrow_file = f'../iborrowdesk_letf_rates/{bear}.csv'

    if not os.path.exists(bull_borrow_file) or not os.path.exists(bear_borrow_file):
        continue

    bull_borrow = pd.read_csv(bull_borrow_file, parse_dates=['time'])
    bull_borrow = bull_borrow.rename(columns={'time': 'date', 'fee': f'b_{bull}'})
    bull_borrow = bull_borrow[['date', f'b_{bull}']].set_index('date')

    bear_borrow = pd.read_csv(bear_borrow_file, parse_dates=['time'])
    bear_borrow = bear_borrow.rename(columns={'time': 'date', 'fee': f'b_{bear}'})
    bear_borrow = bear_borrow[['date', f'b_{bear}']].set_index('date')

    pair_df = bull_prices.join(bear_prices, how='inner')
    pair_df = pair_df.join(bull_borrow, how='left')
    pair_df = pair_df.join(bear_borrow, how='left')

    pair_df = pair_df[~pair_df.index.duplicated(keep='first')]
    pair_df = pair_df.sort_index()

    pair_df[f'b_{bull}'] = pair_df[f'b_{bull}'].ffill()
    pair_df[f'b_{bear}'] = pair_df[f'b_{bear}'].ffill()
    pair_df = pair_df.dropna()

    pair_df[f'ret_{bull}'] = pair_df[f'p_{bull}'].pct_change()
    pair_df[f'ret_{bear}'] = pair_df[f'p_{bear}'].pct_change()
    pair_df[f'vol_{bull}'] = pair_df[f'ret_{bull}'].rolling(30).std() * np.sqrt(252)
    pair_df['avg_borrow'] = (pair_df[f'b_{bull}'] + pair_df[f'b_{bear}']) / 2

    k = k_values[leverage]
    pair_df['vol_pct'] = pair_df[f'vol_{bull}'] * 100
    pair_df['expected_edge'] = (pair_df['vol_pct'] / k) - pair_df['avg_borrow']
    pair_df['expected_edge'] = pair_df['expected_edge'].clip(lower=0)

    pair_df = pair_df.dropna()

    if len(pair_df) > 0:
        pair_data[(bull, bear, leverage)] = pair_df
        print(f"  {bull}/{bear}: {len(pair_df)} days, avg edge: {pair_df['expected_edge'].mean():.2f}%")

# Reconstruct portfolio day by day
portfolio_dates = []
portfolio_returns = []
pair_counts = []
vol_values = []
borrow_values = []

for date in bil_df.index:
    available_pairs_with_edge = []
    edges = []

    for (bull, bear, leverage), pair_df in pair_data.items():
        if date in pair_df.index:
            edge = pair_df.loc[date, 'expected_edge']
            if edge > 0:
                available_pairs_with_edge.append((bull, bear, leverage))
                edges.append(edge)

    if len(available_pairs_with_edge) == 0:
        cash_ret = 2.0 * (bil_df.loc[date, 'bil_ret'] - 0.01/252)
        portfolio_dates.append(date)
        portfolio_returns.append(cash_ret)
        pair_counts.append(0)
        vol_values.append(0)
        borrow_values.append(0)
        continue

    total_edge = sum(edges)
    weights = [edge / total_edge for edge in edges]

    cash_ret = 2.0 * (bil_df.loc[date, 'bil_ret'] - 0.01/252)
    short_ret = 0.0
    borrow_cost = 0.0
    weighted_vol = 0.0
    weighted_borrow = 0.0

    for i, (bull, bear, leverage) in enumerate(available_pairs_with_edge):
        pair_df = pair_data[(bull, bear, leverage)]
        weight = weights[i]
        weight_per_side = 0.5 * weight

        short_ret += -weight_per_side * (pair_df.loc[date, f'ret_{bull}'] + pair_df.loc[date, f'ret_{bear}'])
        pair_borrow = (pair_df.loc[date, f'b_{bull}'] + pair_df.loc[date, f'b_{bear}']) / 2
        borrow_cost += -weight_per_side * (pair_borrow/100/252) * 2  # Both sides

        # Track weighted average vol and borrow
        weighted_vol += weight * pair_df.loc[date, 'vol_pct']
        weighted_borrow += weight * pair_borrow

    total_ret = cash_ret + short_ret + borrow_cost

    portfolio_dates.append(date)
    portfolio_returns.append(total_ret)
    pair_counts.append(len(available_pairs_with_edge))
    vol_values.append(weighted_vol)
    borrow_values.append(weighted_borrow)

# Create results dataframe
results_df = pd.DataFrame({
    'date': portfolio_dates,
    'return': portfolio_returns,
    'n_pairs': pair_counts,
    'vol': vol_values,
    'borrow': borrow_values
})
results_df = results_df.set_index('date')
results_df['year'] = results_df.index.year

# Calculate year-by-year performance
print("="*80)
print("YEAR-BY-YEAR ANALYSIS")
print("="*80)
print(f"{'Year':<6} {'Return':>8} {'AvgVol':>8} {'AvgBorrow':>10} {'AvgPairs':>9} {'Days':>6}")
print("-"*80)

for year in sorted(results_df['year'].unique()):
    year_data = results_df[results_df['year'] == year]
    year_ret = (1 + year_data['return']).prod() - 1
    avg_vol = year_data[year_data['vol'] > 0]['vol'].mean()
    avg_borrow = year_data[year_data['borrow'] > 0]['borrow'].mean()
    avg_pairs = year_data['n_pairs'].mean()
    days = len(year_data)

    print(f"{year:<6} {year_ret:>7.2%} {avg_vol:>7.1f}% {avg_borrow:>9.2f}% {avg_pairs:>9.1f} {days:>6}")

# Split analysis: pre-2020 vs 2020+
print("\n" + "="*80)
print("PRE-2020 vs 2020+ COMPARISON")
print("="*80)

pre_2020 = results_df[results_df['year'] < 2020]
post_2020 = results_df[results_df['year'] >= 2020]

def calc_stats(df):
    total_ret = (1 + df['return']).prod() - 1
    years = len(df) / 252
    cagr = (1 + total_ret) ** (1/years) - 1
    vol = df['return'].std() * np.sqrt(252)
    sharpe = (df['return'].mean() * 252) / vol if vol > 0 else 0
    avg_vol = df[df['vol'] > 0]['vol'].mean()
    avg_borrow = df[df['borrow'] > 0]['borrow'].mean()
    avg_pairs = df['n_pairs'].mean()
    return cagr, sharpe, avg_vol, avg_borrow, avg_pairs, years

pre_stats = calc_stats(pre_2020)
post_stats = calc_stats(post_2020)

print(f"\nPRE-2020 (2015-2019):")
print(f"  CAGR:        {pre_stats[0]:>7.2%}")
print(f"  Sharpe:      {pre_stats[1]:>7.2f}")
print(f"  Avg Vol:     {pre_stats[2]:>7.1f}%")
print(f"  Avg Borrow:  {pre_stats[3]:>7.2f}%")
print(f"  Avg Pairs:   {pre_stats[4]:>7.1f}")
print(f"  Years:       {pre_stats[5]:>7.1f}")

print(f"\n2020+ (2020-2025):")
print(f"  CAGR:        {post_stats[0]:>7.2%}")
print(f"  Sharpe:      {post_stats[1]:>7.2f}")
print(f"  Avg Vol:     {post_stats[2]:>7.1f}%")
print(f"  Avg Borrow:  {post_stats[3]:>7.2f}%")
print(f"  Avg Pairs:   {post_stats[4]:>7.1f}")
print(f"  Years:       {post_stats[5]:>7.1f}")

print(f"\nCHANGE:")
print(f"  CAGR:        {(post_stats[0] - pre_stats[0])*100:>+6.2f}%")
print(f"  Sharpe:      {post_stats[1] - pre_stats[1]:>+7.2f}")
print(f"  Avg Vol:     {(post_stats[2] - pre_stats[2]):>+6.1f}%")
print(f"  Avg Borrow:  {(post_stats[3] - pre_stats[3]):>+6.2f}%")
print(f"  Avg Pairs:   {post_stats[4] - pre_stats[4]:>+7.1f}")
