"""Debug script to understand short position mechanics"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient
from datetime import datetime, timedelta

# Test with simple example
client = FMPClient()

end_date = datetime.now()
start_date = end_date - timedelta(days=60)

# Get TQQQ and SQQQ
bull_data = client.get_historical_prices('TQQQ', from_date=start_date.strftime('%Y-%m-%d'), to_date=end_date.strftime('%Y-%m-%d'))
bear_data = client.get_historical_prices('SQQQ', from_date=start_date.strftime('%Y-%m-%d'), to_date=end_date.strftime('%Y-%m-%d'))

bull_df = pd.DataFrame(bull_data)[['date', 'adjClose']]
bear_df = pd.DataFrame(bear_data)[['date', 'adjClose']]

bull_df['date'] = pd.to_datetime(bull_df['date'])
bear_df['date'] = pd.to_datetime(bear_df['date'])

bull_df = bull_df.sort_values('date').set_index('date')
bear_df = bear_df.sort_values('date').set_index('date')

df = bull_df.join(bear_df, lsuffix='_bull', rsuffix='_bear')
df.columns = ['price_bull', 'price_bear']

print("First 10 rows:")
print(df.head(10))
print()

# Simulate short position
initial_capital = 100000
initial_price_bull = df.iloc[0]['price_bull']
initial_price_bear = df.iloc[0]['price_bear']

print(f"Initial prices: TQQQ=${initial_price_bull:.2f}, SQQQ=${initial_price_bear:.2f}")
print()

# Short 50% in each
# When you short: you borrow shares, sell them, receive cash
# Initial: $100k
# Short $50k of TQQQ at $100/share = -500 shares (you owe 500 shares)
# Short $50k of SQQQ at $10/share = -5000 shares (you owe 5000 shares)
# Cash: $100k + $50k + $50k = $200k

shares_bull = -(initial_capital * 0.5) / initial_price_bull
shares_bear = -(initial_capital * 0.5) / initial_price_bear

print(f"Initial short positions:")
print(f"  TQQQ: {shares_bull:.2f} shares (negative = short)")
print(f"  SQQQ: {shares_bear:.2f} shares (negative = short)")
print()

# Track equity over time
print("Date          | TQQQ Price | SQQQ Price | TQQQ P&L | SQQQ P&L | Total Equity")
print("-" * 85)

for i in range(min(10, len(df))):
    row = df.iloc[i]
    price_bull = row['price_bull']
    price_bear = row['price_bear']

    # P&L on short = (entry_price - current_price) * shares (but shares are negative)
    # Alternative: position_value = shares * price (negative value for short)
    # P&L = -(position_value - initial_value)

    # Current position value (negative because we're short)
    position_value_bull = shares_bull * price_bull
    position_value_bear = shares_bear * price_bear

    # P&L: we entered at initial prices, now at current prices
    # For shorts: profit when price goes down
    initial_value_bull = shares_bull * initial_price_bull  # negative
    initial_value_bear = shares_bear * initial_price_bear  # negative

    pnl_bull = initial_value_bull - position_value_bull
    pnl_bear = initial_value_bear - position_value_bear

    # Total equity = initial capital + P&L
    total_equity = initial_capital + pnl_bull + pnl_bear

    print(f"{df.index[i].date()} | ${price_bull:8.2f} | ${price_bear:8.2f} | ${pnl_bull:8.2f} | ${pnl_bear:8.2f} | ${total_equity:10.2f}")
