"""
Test the simple logic with a concrete example.

Day 0:
- Have $100k
- Short $50k of TQQQ at $20/share = -2,500 shares
- Short $50k of SQQQ at $200/share = -250 shares
- Now have $200k cash (original $100k + $100k from short proceeds)
- Buy BIL with $200k at $91/share = 2,198 shares

Day 1 (no rebalance):
- TQQQ at $21 (up $1)
- SQQQ at $198 (down $2)
- BIL at $91.01 (up $0.01)

- TQQQ P&L: -2,500 * ($20 - $21) = -2,500 * (-$1) = +$2,500 (loss, price went up)
  Wait, that's wrong. If I'm short and price goes up, I lose money.
  P&L = shares * (entry - current) = -2,500 * ($20 - $21) = -2,500 * (-$1) = $2,500

  Actually let's think about it differently:
  - I shorted at $20, now it's $21
  - To close the short, I'd need to buy back at $21
  - I sold for $50k, buy back costs $52.5k
  - Loss = -$2,500

  Formula: For shorts, P&L = -shares_short * (current_price - entry_price)
  Or: P&L = shares_short * (entry_price - current_price) where shares_short is negative

  So: P&L_TQQQ = -2,500 * ($21 - $20) = -$2,500

- SQQQ P&L: -250 * ($198 - $200) = -250 * (-$2) = +$500 (gain, price went down)
  Check: sold at $200 for $50k, buy back at $198 costs $49.5k, profit = $500 ✓

- BIL value: 2,198 * $91.01 = $200,021.98

Equity = BIL value + short P&L
       = $200,021.98 + (-$2,500) + $500
       = $198,021.98

Starting equity was $100k, so return = ($198,021.98 - $100k) / $100k = +98.02%

Wait that can't be right. Let me recalculate...

Actually, the equity calculation is wrong. Let me think about it as a balance sheet:

Assets:
- BIL: $200,021.98

Liabilities:
- Short TQQQ: owe 2,500 shares worth 2,500 * $21 = $52,500
- Short SQQQ: owe 250 shares worth 250 * $198 = $49,500
- Total liabilities: $102,000

Equity = Assets - Liabilities = $200,021.98 - $102,000 = $98,021.98

So starting with $100k, we now have $98,021.98 = -1.98% return.

That makes sense! Short TQQQ lost us money, short SQQQ made us money, BIL made a tiny bit.
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from fmp_client import FMPClient
from datetime import datetime, timedelta

client = FMPClient()

# Get just 10 days of data for TQQQ/SQQQ/BIL
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

tqqq = client.get_historical_prices('TQQQ', from_date=start_date.strftime('%Y-%m-%d'), to_date=end_date.strftime('%Y-%m-%d'))
sqqq = client.get_historical_prices('SQQQ', from_date=start_date.strftime('%Y-%m-%d'), to_date=end_date.strftime('%Y-%m-%d'))
bil = client.get_historical_prices('BIL', from_date=start_date.strftime('%Y-%m-%d'), to_date=end_date.strftime('%Y-%m-%d'))

df_tqqq = pd.DataFrame(tqqq)[['date', 'adjClose']]
df_sqqq = pd.DataFrame(sqqq)[['date', 'adjClose']]
df_bil = pd.DataFrame(bil)[['date', 'adjClose']]

df_tqqq['date'] = pd.to_datetime(df_tqqq['date'])
df_sqqq['date'] = pd.to_datetime(df_sqqq['date'])
df_bil['date'] = pd.to_datetime(df_bil['date'])

df = df_tqqq.set_index('date').join(df_sqqq.set_index('date'), lsuffix='_tqqq', rsuffix='_sqqq')
df = df.join(df_bil.set_index('date'))
df.columns = ['p_tqqq', 'p_sqqq', 'p_bil']
df = df.sort_index()[:10]  # Just first 10 days

print("First 10 days of data:")
print(df)
print()

# Start with $100k
capital = 100000

# Day 0: Short $50k of each
p0_tqqq = df.iloc[0]['p_tqqq']
p0_sqqq = df.iloc[0]['p_sqqq']
p0_bil = df.iloc[0]['p_bil']

print(f"Day 0 prices: TQQQ=${p0_tqqq:.2f}, SQQQ=${p0_sqqq:.2f}, BIL=${p0_bil:.2f}")

shares_tqqq = -50000 / p0_tqqq  # Negative = short
shares_sqqq = -50000 / p0_sqqq
shares_bil = 200000 / p0_bil  # Long (all our cash)

print(f"Positions: {shares_tqqq:.2f} TQQQ (short), {shares_sqqq:.2f} SQQQ (short), {shares_bil:.2f} BIL (long)")
print()

# Calculate equity each day
print("Date          | TQQQ | SQQQ |  BIL  | BIL Val | TQQQ Liab | SQQQ Liab | Equity   | Return")
print("-" * 100)

for date, row in df.iterrows():
    p_tqqq = row['p_tqqq']
    p_sqqq = row['p_sqqq']
    p_bil = row['p_bil']

    # Assets
    bil_value = shares_bil * p_bil

    # Liabilities (positive numbers for what we owe)
    tqqq_liability = abs(shares_tqqq) * p_tqqq
    sqqq_liability = abs(shares_sqqq) * p_sqqq

    # Equity = Assets - Liabilities
    equity = bil_value - tqqq_liability - sqqq_liability

    ret = (equity / capital - 1) * 100

    print(f"{date.date()} | ${p_tqqq:5.2f} | ${p_sqqq:5.2f} | ${p_bil:6.2f} | ${bil_value:9.2f} | ${tqqq_liability:9.2f} | ${sqqq_liability:9.2f} | ${equity:8.2f} | {ret:+6.2f}%")

print()
print("This is the correct way to calculate equity:")
print("Equity = BIL value - abs(short_tqqq_shares) * price_tqqq - abs(short_sqqq_shares) * price_sqqq")
