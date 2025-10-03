"""
Test script to fetch daily OHLC stock data using the FMP client.
"""

from fmp_client import FMPClient
from datetime import datetime, timedelta
import json


def main():
    # Initialize the FMP client
    print("Initializing FMP client...")
    client = FMPClient()

    # Fetch historical data for Apple (AAPL)
    symbol = "AAPL"
    print(f"\nFetching historical data for {symbol}...")

    # Get last 30 days of data
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    try:
        data = client.get_historical_prices(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date
        )

        print(f"\nReceived {len(data)} days of data")
        print(f"Date range: {from_date} to {to_date}\n")

        # Display the most recent 5 days
        print("Most recent 5 trading days:")
        print("-" * 100)
        print(f"{'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>15} {'Change %':>10}")
        print("-" * 100)

        for record in data[:5]:
            print(
                f"{record['date']:<12} "
                f"{record['open']:>10.2f} "
                f"{record['high']:>10.2f} "
                f"{record['low']:>10.2f} "
                f"{record['close']:>10.2f} "
                f"{record['volume']:>15,} "
                f"{record.get('changePercent', 0):>9.2f}%"
            )

        # Get current quote
        print(f"\n\nCurrent quote for {symbol}:")
        print("-" * 100)
        quote = client.get_quote(symbol)
        print(json.dumps(quote, indent=2))

    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure you have set your FMP_API_KEY in the .env file")


if __name__ == "__main__":
    main()
