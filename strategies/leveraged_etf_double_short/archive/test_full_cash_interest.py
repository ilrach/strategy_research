"""
Test interest on FULL cash balance (capital + short proceeds)

Portfolio Visualizer likely credits interest on:
- Original $100k capital
- Plus $100k proceeds from shorts
- Total: $200k earning interest

This is more generous than typical margin accounts but represents
the "synthetic" cash position when holding shorts.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient
from datetime import datetime, timedelta

client = FMPClient()

end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)

print("Fetching data...")
tqqq_data = client.get_historical_prices('TQQQ', from_date=start_date.strftime('%Y-%m-%d'), to_date=end_date.strftime('%Y-%m-%d'))
sqqq_data = client.get_historical_prices('SQQQ', from_date=start_date.strftime('%Y-%m-%d'), to_date=end_date.strftime('%Y-%m-%d'))

tqqq_df = pd.DataFrame(tqqq_data)[['date', 'adjClose']]
sqqq_df = pd.DataFrame(sqqq_data)[['date', 'adjClose']]

tqqq_df['date'] = pd.to_datetime(tqqq_df['date'])
sqqq_df['date'] = pd.to_datetime(sqqq_df['date'])

tqqq_df = tqqq_df.sort_values('date').set_index('date')
sqqq_df = sqqq_df.sort_values('date').set_index('date')

df = tqqq_df.join(sqqq_df, lsuffix='_tqqq', rsuffix='_sqqq')
df.columns = ['price_tqqq', 'price_sqqq']

print(f"Period: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)\n")

initial_capital = 100000
cash_rate_annual = 0.045

# Initial setup
p0_tqqq = df.iloc[0]['price_tqqq']
p0_sqqq = df.iloc[0]['price_sqqq']

shares_tqqq = -(initial_capital * 0.5) / p0_tqqq
shares_sqqq = -(initial_capital * 0.5) / p0_sqqq

# Initial values
entry_value_tqqq = shares_tqqq * p0_tqqq  # -$50k
entry_value_sqqq = shares_sqqq * p0_sqqq  # -$50k

# Cash balance = capital + abs(short positions)
# This represents: your money + the proceeds from short sales
cash_balance = initial_capital + abs(entry_value_tqqq) + abs(entry_value_sqqq)

print("="*80)
print("STRATEGY: Interest on FULL Cash Balance (Capital + Short Proceeds)")
print("="*80)
print(f"Initial capital: ${initial_capital:,.2f}")
print(f"Short TQQQ value: ${abs(entry_value_tqqq):,.2f}")
print(f"Short SQQQ value: ${abs(entry_value_sqqq):,.2f}")
print(f"Total cash balance: ${cash_balance:,.2f}")
print(f"Cash rate: {cash_rate_annual*100:.2f}% annually")
print()

cumulative_interest = 0
last_rebalance = df.index[0]
rebalances = 0

for i in range(len(df)):
    date = df.index[i]
    p_tqqq = df.iloc[i]['price_tqqq']
    p_sqqq = df.iloc[i]['price_sqqq']

    # Accrue interest on cash balance
    if i > 0:
        days = (date - df.index[i-1]).days
        interest = cash_balance * (cash_rate_annual / 365) * days
        cumulative_interest += interest
        cash_balance += interest

    # Current position values
    curr_value_tqqq = shares_tqqq * p_tqqq
    curr_value_sqqq = shares_sqqq * p_sqqq

    # P&L on shorts
    pnl_tqqq = entry_value_tqqq - curr_value_tqqq
    pnl_sqqq = entry_value_sqqq - curr_value_sqqq

    # Equity = cash - liability from short positions
    # Liability = abs(current short values)
    equity = cash_balance - abs(curr_value_tqqq) - abs(curr_value_sqqq)

    # Weekly rebalance
    if i > 0 and (date - last_rebalance).days >= 7:
        # Rebalance to 50/50
        shares_tqqq = -(equity * 0.5) / p_tqqq
        shares_sqqq = -(equity * 0.5) / p_sqqq

        entry_value_tqqq = shares_tqqq * p_tqqq
        entry_value_sqqq = shares_sqqq * p_sqqq

        # Reset cash balance
        cash_balance = equity + abs(entry_value_tqqq) + abs(entry_value_sqqq)
        cumulative_interest = 0

        last_rebalance = date
        rebalances += 1

# Final calculations
p_final_tqqq = df.iloc[-1]['price_tqqq']
p_final_sqqq = df.iloc[-1]['price_sqqq']

final_value_tqqq = shares_tqqq * p_final_tqqq
final_value_sqqq = shares_sqqq * p_final_sqqq

final_equity = cash_balance - abs(final_value_tqqq) - abs(final_value_sqqq)

total_return = (final_equity / initial_capital - 1) * 100
years = len(df) / 365.25
cagr = (final_equity / initial_capital) ** (1/years) - 1

print(f"Final equity: ${final_equity:,.2f}")
print(f"Total return: {total_return:.2f}%")
print(f"CAGR: {cagr*100:.2f}%")
print(f"Rebalances: {rebalances}")
print()

# For comparison
print("="*80)
print("COMPARISON")
print("="*80)
print("Portfolio Visualizer (14 years): +151.72% total, ~6.87% CAGR")
print(f"This model (3 years): +{total_return:.2f}% total, {cagr*100:.2f}% CAGR")
print()
print("If this trend continues:")
projected_14yr = (1 + cagr) ** (14/3) - 1
print(f"Projected 14-year return: {projected_14yr*100:.2f}%")
