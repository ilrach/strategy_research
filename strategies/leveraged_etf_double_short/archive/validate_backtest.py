"""
Validate backtest logic against known example
Compare with Portfolio Visualizer results for TQQQ/SQQQ
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient
from datetime import datetime, timedelta

client = FMPClient()

# Get 3 years of TQQQ and SQQQ data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)

print("Fetching TQQQ and SQQQ data...")
tqqq_data = client.get_historical_prices('TQQQ', from_date=start_date.strftime('%Y-%m-%d'), to_date=end_date.strftime('%Y-%m-%d'))
sqqq_data = client.get_historical_prices('SQQQ', from_date=start_date.strftime('%Y-%m-%d'), to_date=end_date.strftime('%Y-%m-%d'))

# Convert to DataFrames
tqqq_df = pd.DataFrame(tqqq_data)[['date', 'adjClose']]
sqqq_df = pd.DataFrame(sqqq_data)[['date', 'adjClose']]

tqqq_df['date'] = pd.to_datetime(tqqq_df['date'])
sqqq_df['date'] = pd.to_datetime(sqqq_df['date'])

tqqq_df = tqqq_df.sort_values('date').set_index('date')
sqqq_df = sqqq_df.sort_values('date').set_index('date')

# Merge
df = tqqq_df.join(sqqq_df, lsuffix='_tqqq', rsuffix='_sqqq')
df.columns = ['price_tqqq', 'price_sqqq']

print(f"\nData range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Number of days: {len(df)}")
print()

# Strategy 1: What the backtest does (SHORT both sides)
print("="*80)
print("STRATEGY 1: SHORT TQQQ + SHORT SQQQ (Current Backtest)")
print("="*80)

capital = 100000
initial_price_tqqq = df.iloc[0]['price_tqqq']
initial_price_sqqq = df.iloc[0]['price_sqqq']

# Short 50% in each
shares_tqqq = -(capital * 0.5) / initial_price_tqqq  # Negative = short
shares_sqqq = -(capital * 0.5) / initial_price_sqqq  # Negative = short

initial_value_tqqq = shares_tqqq * initial_price_tqqq
initial_value_sqqq = shares_sqqq * initial_price_sqqq

print(f"Initial TQQQ price: ${initial_price_tqqq:.2f}")
print(f"Initial SQQQ price: ${initial_price_sqqq:.2f}")
print(f"Shares TQQQ: {shares_tqqq:.2f} (short)")
print(f"Shares SQQQ: {shares_sqqq:.2f} (short)")
print()

# Weekly rebalancing
last_rebalance_date = df.index[0]
rebalance_count = 0

for i, (date, row) in enumerate(df.iterrows()):
    price_tqqq = row['price_tqqq']
    price_sqqq = row['price_sqqq']

    # Current position values
    current_value_tqqq = shares_tqqq * price_tqqq
    current_value_sqqq = shares_sqqq * price_sqqq

    # P&L for shorts
    pnl_tqqq = initial_value_tqqq - current_value_tqqq
    pnl_sqqq = initial_value_sqqq - current_value_sqqq

    equity = capital + pnl_tqqq + pnl_sqqq

    # Weekly rebalance
    days_since = (date - last_rebalance_date).days
    if days_since >= 7 and i > 0:
        shares_tqqq = -(equity * 0.5) / price_tqqq
        shares_sqqq = -(equity * 0.5) / price_sqqq
        initial_value_tqqq = shares_tqqq * price_tqqq
        initial_value_sqqq = shares_sqqq * price_sqqq
        last_rebalance_date = date
        rebalance_count += 1

final_price_tqqq = df.iloc[-1]['price_tqqq']
final_price_sqqq = df.iloc[-1]['price_sqqq']
final_value_tqqq = shares_tqqq * final_price_tqqq
final_value_sqqq = shares_sqqq * final_price_sqqq
final_pnl_tqqq = initial_value_tqqq - final_value_tqqq
final_pnl_sqqq = initial_value_sqqq - final_value_sqqq
final_equity_short = capital + final_pnl_tqqq + final_pnl_sqqq

print(f"Final TQQQ price: ${final_price_tqqq:.2f}")
print(f"Final SQQQ price: ${final_price_sqqq:.2f}")
print(f"Rebalances: {rebalance_count}")
print(f"Final equity: ${final_equity_short:,.2f}")
print(f"Total return: {(final_equity_short/capital - 1)*100:.2f}%")
print()

# Strategy 2: What Portfolio Visualizer might be doing (LONG both sides with rebalancing)
print("="*80)
print("STRATEGY 2: LONG TQQQ + LONG SQQQ (50/50 Rebalanced)")
print("="*80)

capital = 100000
shares_tqqq_long = (capital * 0.5) / initial_price_tqqq
shares_sqqq_long = (capital * 0.5) / initial_price_sqqq

print(f"Initial TQQQ shares: {shares_tqqq_long:.2f} (long)")
print(f"Initial SQQQ shares: {shares_sqqq_long:.2f} (long)")
print()

last_rebalance_date = df.index[0]
rebalance_count = 0

for i, (date, row) in enumerate(df.iterrows()):
    price_tqqq = row['price_tqqq']
    price_sqqq = row['price_sqqq']

    equity = shares_tqqq_long * price_tqqq + shares_sqqq_long * price_sqqq

    # Weekly rebalance
    days_since = (date - last_rebalance_date).days
    if days_since >= 7 and i > 0:
        shares_tqqq_long = (equity * 0.5) / price_tqqq
        shares_sqqq_long = (equity * 0.5) / price_sqqq
        last_rebalance_date = date
        rebalance_count += 1

final_equity_long = shares_tqqq_long * final_price_tqqq + shares_sqqq_long * final_price_sqqq

print(f"Rebalances: {rebalance_count}")
print(f"Final equity: ${final_equity_long:,.2f}")
print(f"Total return: {(final_equity_long/capital - 1)*100:.2f}%")
print()

# Check if Portfolio Visualizer used -50% weights (which would be shorts)
print("="*80)
print("Hypothesis: Portfolio Visualizer interprets -50% as SHORT position")
print("="*80)
print("If testfol.io shows +151% over 14 years with weekly rebalancing,")
print("that suggests LONG positions with rebalancing benefit, not shorts.")
print()
print("Let's check the individual ETF returns:")

tqqq_return = (final_price_tqqq / initial_price_tqqq - 1) * 100
sqqq_return = (final_price_sqqq / initial_price_sqqq - 1) * 100

print(f"TQQQ buy & hold return: {tqqq_return:.2f}%")
print(f"SQQQ buy & hold return: {sqqq_return:.2f}%")
print(f"Simple average: {(tqqq_return + sqqq_return)/2:.2f}%")
