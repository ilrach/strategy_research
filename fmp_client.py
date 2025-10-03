"""
Financial Modeling Prep API Client
Provides methods to fetch stock data from the FMP API.
"""

import os
import requests
from typing import Optional, Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv


class FMPClient:
    """Client for interacting with the Financial Modeling Prep API."""

    BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FMP client.

        Args:
            api_key: FMP API key. If not provided, will load from .env file.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv('FMP_API_KEY')

        if not self.api_key or self.api_key == 'your_api_key_here':
            raise ValueError(
                "FMP API key not found. Please set FMP_API_KEY in .env file or pass it to the constructor."
            )

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Make a request to the FMP API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        url = f"{self.BASE_URL}/{endpoint}"

        if params is None:
            params = {}

        params['apikey'] = self.api_key

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def get_historical_prices(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        adjusted: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get historical daily OHLC prices for a stock.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            from_date: Start date in YYYY-MM-DD format (optional)
            to_date: End date in YYYY-MM-DD format (optional)
            adjusted: Use dividend-adjusted prices (default True for total return analysis)

        Returns:
            List of dictionaries containing OHLC data with keys:
            - date: Trading date
            - open: Opening price (adjusted if adjusted=True)
            - high: Highest price (adjusted if adjusted=True)
            - low: Lowest price (adjusted if adjusted=True)
            - close: Closing price (adjusted if adjusted=True)
            - volume: Trading volume
            - adjClose: Adjusted close price (if available)
            - unadjustedVolume: Unadjusted volume
            - change: Price change
            - changePercent: Percentage change
            - vwap: Volume weighted average price
            - label: Date label
            - changeOverTime: Change over time
        """
        params = {'symbol': symbol.upper()}

        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date

        # Use dividend-adjusted endpoint for total return analysis
        endpoint = 'historical-price-eod/dividend-adjusted' if adjusted else 'historical-price-eod/full'
        data = self._make_request(endpoint, params)

        # The API returns data in 'historical' key
        if isinstance(data, dict) and 'historical' in data:
            return data['historical']
        elif isinstance(data, list):
            return data
        else:
            return []

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a stock.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            Dictionary containing current quote data
        """
        data = self._make_request(f'quote/{symbol.upper()}')
        return data[0] if isinstance(data, list) and len(data) > 0 else data
