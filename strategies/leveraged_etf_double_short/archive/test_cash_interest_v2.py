"""
Correct modeling of cash interest with short positions

When you short with $100k capital:
- You have $100k cash
- You short $50k of TQQQ (borrow shares, sell them)
- You short $50k of SQQQ (borrow shares, sell them)
- The proceeds from shorts ($100k total) are typically held as margin collateral
- You earn interest on your NET cash balance (original $100k)
- In some cases, you may earn interest on short proceeds minus borrow costs

For simplicity, let's assume:
- Original capital earns interest at cash rate
- Short proceeds are held as collateral (no additional interest)
- This is conservative vs Portfolio Visualizer which may credit interest on all cash
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
cash_rate_annual = 0.045  # 4.5% assumed cash rate

# Initial prices
p0_tqqq = df.iloc[0]['price_tqqq']
p0_sqqq = df.iloc[0]['price_sqqq']

# Short 50% in each
shares_tqqq = -(initial_capital * 0.5) / p0_tqqq
shares_sqqq = -(initial_capital * 0.5) / p0_sqqq

print("="*80)
print("STRATEGY: SHORT BOTH with Cash Interest")
print("="*80)
print(f"Initial capital: ${initial_capital:,.2f}")
print(f"Cash rate: {cash_rate_annual*100:.2f}% annually")
print(f"Initial TQQQ: ${p0_tqqq:.2f}, SQQQ: ${p0_sqqq:.2f}")
print(f"Short {shares_tqqq:.2f} TQQQ, {shares_sqqq:.2f} SQQQ")
print()

# Track using equity approach
# Equity = initial capital + P&L from shorts + accrued interest
entry_price_tqqq = p0_tqqq
entry_price_sqqq = p0_sqqq
entry_value_tqqq = shares_tqqq * entry_price_tqqq  # Negative
entry_value_sqqq = shares_sqqq * entry_price_sqqq  # Negative

cumulative_interest = 0
last_rebalance = df.index[0]
rebalances = 0

equity_curve = []

for i in range(len(df)):
    date = df.index[i]
    p_tqqq = df.iloc[i]['price_tqqq']
    p_sqqq = df.iloc[i]['price_sqqq']

    # Accrue interest on the equity balance
    if i > 0:
        days = (date - df.index[i-1]).days
        # Interest accrues on current equity (not just initial capital)
        prev_equity = equity_curve[-1]['equity']
        interest = prev_equity * (cash_rate_annual / 365) * days
        cumulative_interest += interest

    # Current position values
    curr_value_tqqq = shares_tqqq * p_tqqq
    curr_value_sqqq = shares_sqqq * p_sqqq

    # P&L on shorts (entry - current)
    pnl_tqqq = entry_value_tqqq - curr_value_tqqq
    pnl_sqqq = entry_value_sqqq - curr_value_sqqq

    # Total equity
    equity = initial_capital + pnl_tqqq + pnl_sqqq + cumulative_interest

    equity_curve.append({'date': date, 'equity': equity})

    # Weekly rebalance
    if i > 0 and (date - last_rebalance).days >= 7:
        # Close positions and reopen at current prices
        # The new "initial capital" for tracking P&L is current equity
        initial_capital = equity
        cumulative_interest = 0  # Reset interest tracking

        shares_tqqq = -(equity * 0.5) / p_tqqq
        shares_sqqq = -(equity * 0.5) / p_sqqq

        entry_price_tqqq = p_tqqq
        entry_price_sqqq = p_sqqq
        entry_value_tqqq = shares_tqqq * entry_price_tqqq
        entry_value_sqqq = shares_sqqq * entry_price_sqqq

        last_rebalance = date
        rebalances += 1

final_equity = equity_curve[-1]['equity']
total_return = (final_equity / 100000 - 1) * 100
years = len(df) / 365.25
cagr = (final_equity / 100000) ** (1/years) - 1

print(f"Final equity: ${final_equity:,.2f}")
print(f"Total return: {total_return:.2f}%")
print(f"CAGR: {cagr*100:.2f}%")
print(f"Rebalances: {rebalances}")
print()

# Calculate buy-hold returns for reference
tqqq_return = (df.iloc[-1]['price_tqqq'] / p0_tqqq - 1) * 100
sqqq_return = (df.iloc[-1]['price_sqqq'] / p0_sqqq - 1) * 100

print("Reference:")
print(f"TQQQ buy-hold: {tqqq_return:.2f}%")
print(f"SQQQ buy-hold: {sqqq_return:.2f}%")
print()

# Now test what interest rate would be needed to match Portfolio Visualizer
# If PV shows +151% over 14 years, that's CAGR of about 6.87%
# Over 3 years with same CAGR: (1.0687)^3 - 1 = 21.9%

target_3yr_return = 0.219  # 21.9%
target_final_equity = 100000 * (1 + target_3yr_return)

print("="*80)
print(f"To match Portfolio Visualizer's implied trajectory:")
print(f"3-year return needed: {target_3yr_return*100:.2f}%")
print(f"Target final equity: ${target_final_equity:,.2f}")
print(f"Actual final equity: ${final_equity:,.2f}")
print(f"Shortfall: ${target_final_equity - final_equity:,.2f}")
