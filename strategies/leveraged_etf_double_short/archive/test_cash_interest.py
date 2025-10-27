"""
Test the impact of cash interest on short positions
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient
from datetime import datetime, timedelta

client = FMPClient()

# Get 3 years of data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)

print("Fetching TQQQ and SQQQ data...")
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

print(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Days: {len(df)}")
print()

# Initial setup
initial_capital = 100000
initial_price_tqqq = df.iloc[0]['price_tqqq']
initial_price_sqqq = df.iloc[0]['price_sqqq']

# Short $50k of each
shares_tqqq = -(initial_capital * 0.5) / initial_price_tqqq
shares_sqqq = -(initial_capital * 0.5) / initial_price_sqqq

# Cash position: original capital + proceeds from shorts
# When you short, you receive cash
initial_cash = initial_capital + abs(shares_tqqq * initial_price_tqqq) + abs(shares_sqqq * initial_price_sqqq)
print(f"Initial capital: ${initial_capital:,.2f}")
print(f"Short TQQQ proceeds: ${abs(shares_tqqq * initial_price_tqqq):,.2f}")
print(f"Short SQQQ proceeds: ${abs(shares_sqqq * initial_price_sqqq):,.2f}")
print(f"Total cash position: ${initial_cash:,.2f}")
print()

# Assume cash earns a rate (approximate historical rates)
# 2022-2023: ~4-5% (Fed hiking cycle)
# 2024-2025: ~5% (elevated rates)
# Let's use 4.5% annually as average over 3 years
cash_rate_annual = 0.045

print("="*80)
print("BACKTEST WITH CASH INTEREST")
print("="*80)
print(f"Assumed cash rate: {cash_rate_annual*100:.2f}% per year")
print()

initial_value_tqqq = shares_tqqq * initial_price_tqqq
initial_value_sqqq = shares_sqqq * initial_price_sqqq

cash = initial_cash
last_rebalance_date = df.index[0]
rebalance_count = 0
cumulative_interest = 0

for i, (date, row) in enumerate(df.iterrows()):
    price_tqqq = row['price_tqqq']
    price_sqqq = row['price_sqqq']

    # Calculate days since last row for interest accrual
    if i > 0:
        days_elapsed = (date - df.index[i-1]).days
        daily_interest = cash * (cash_rate_annual / 365) * days_elapsed
        cash += daily_interest
        cumulative_interest += daily_interest

    # Current position values (negative for shorts)
    current_value_tqqq = shares_tqqq * price_tqqq
    current_value_sqqq = shares_sqqq * price_sqqq

    # P&L on shorts (initial - current for shorts)
    pnl_tqqq = initial_value_tqqq - current_value_tqqq
    pnl_sqqq = initial_value_sqqq - current_value_sqqq

    # Total equity = cash + P&L from positions
    equity = cash + pnl_tqqq + pnl_sqqq

    # Weekly rebalance
    days_since = (date - last_rebalance_date).days
    if days_since >= 7 and i > 0:
        # When we rebalance, we essentially:
        # 1. Close out old short positions (realize P&L)
        # 2. Open new short positions at current prices

        # Realize the P&L into cash
        cash = equity

        # Open new shorts: 50% of equity each
        target_short_value = equity * 0.5
        shares_tqqq = -target_short_value / price_tqqq
        shares_sqqq = -target_short_value / price_sqqq

        # The shorts have new entry prices
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
final_equity = cash + final_pnl_tqqq + final_pnl_sqqq

print(f"Final equity: ${final_equity:,.2f}")
print(f"Total return: {(final_equity/initial_capital - 1)*100:.2f}%")
print(f"Cumulative interest earned: ${cumulative_interest:,.2f}")
print(f"Interest contribution to return: {(cumulative_interest/initial_capital)*100:.2f}%")
print(f"Rebalances: {rebalance_count}")
print()

# Annualized metrics
years = len(df) / 365.25
cagr = (final_equity / initial_capital) ** (1/years) - 1
print(f"CAGR: {cagr*100:.2f}%")
print()

# Compare to strategy without cash interest
print("="*80)
print("COMPARISON: Strategy WITHOUT cash interest")
print("="*80)

shares_tqqq_nocash = -(initial_capital * 0.5) / initial_price_tqqq
shares_sqqq_nocash = -(initial_capital * 0.5) / initial_price_sqqq
initial_value_tqqq_nocash = shares_tqqq_nocash * initial_price_tqqq
initial_value_sqqq_nocash = shares_sqqq_nocash * initial_price_sqqq

last_rebalance_date = df.index[0]
for i, (date, row) in enumerate(df.iterrows()):
    price_tqqq = row['price_tqqq']
    price_sqqq = row['price_sqqq']

    current_value_tqqq = shares_tqqq_nocash * price_tqqq
    current_value_sqqq = shares_sqqq_nocash * price_sqqq

    pnl_tqqq = initial_value_tqqq_nocash - current_value_tqqq
    pnl_sqqq = initial_value_sqqq_nocash - current_value_sqqq

    equity_nocash = initial_capital + pnl_tqqq + pnl_sqqq

    days_since = (date - last_rebalance_date).days
    if days_since >= 7 and i > 0:
        shares_tqqq_nocash = -(equity_nocash * 0.5) / price_tqqq
        shares_sqqq_nocash = -(equity_nocash * 0.5) / price_sqqq
        initial_value_tqqq_nocash = shares_tqqq_nocash * price_tqqq
        initial_value_sqqq_nocash = shares_sqqq_nocash * price_sqqq
        last_rebalance_date = date

final_value_tqqq = shares_tqqq_nocash * final_price_tqqq
final_value_sqqq = shares_sqqq_nocash * final_price_sqqq
final_pnl_tqqq = initial_value_tqqq_nocash - final_value_tqqq
final_pnl_sqqq = initial_value_sqqq_nocash - final_value_sqqq
final_equity_nocash = initial_capital + final_pnl_tqqq + final_pnl_sqqq

print(f"Final equity: ${final_equity_nocash:,.2f}")
print(f"Total return: {(final_equity_nocash/initial_capital - 1)*100:.2f}%")

cagr_nocash = (final_equity_nocash / initial_capital) ** (1/years) - 1
print(f"CAGR: {cagr_nocash*100:.2f}%")
print()

print("="*80)
print("IMPACT OF CASH INTEREST")
print("="*80)
print(f"Return difference: {(final_equity - final_equity_nocash)/initial_capital*100:.2f}%")
print(f"CAGR difference: {(cagr - cagr_nocash)*100:.2f}%")
