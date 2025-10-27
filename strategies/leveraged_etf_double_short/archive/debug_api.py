"""Debug script to check FMP API response format"""

from fmp_client import FMPClient
from datetime import datetime, timedelta
import json

client = FMPClient()

# Test with one ticker
ticker = 'TQQQ'
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
from_date = start_date.strftime('%Y-%m-%d')
to_date = end_date.strftime('%Y-%m-%d')

print(f"Testing {ticker} from {from_date} to {to_date}")
print()

try:
    data = client.get_historical_prices(ticker, from_date=from_date, to_date=to_date)
    print(f"Data type: {type(data)}")
    print(f"Data length: {len(data) if isinstance(data, list) else 'N/A'}")
    print()

    if data:
        print("First record:")
        print(json.dumps(data[0], indent=2))
        print()
        print("Available keys:", list(data[0].keys()) if isinstance(data, list) else "N/A")
    else:
        print("No data returned")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
