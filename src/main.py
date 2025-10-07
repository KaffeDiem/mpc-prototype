"""Simple example usage of the PricesService to fetch Danish electricity prices."""

from datetime import date, timedelta
import logging
from prices_service import PricesService

# Set up logging to see any warnings about missing dates
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Initialize service for DK2 region (Copenhagen/East of Great Belt)
    # Use "DK1" for Aarhus/West of Great Belt
    service = PricesService(region="DK2")
    
    # Get today's and tomorrow's prices
    today = date.today()
    tomorrow = today + timedelta(days=1)
    
    print(f"Fetching electricity prices for {today} to {tomorrow}...")
    prices = service.get_prices(today, tomorrow)
    
    print(f"\nRetrieved {len(prices)} hourly prices:\n")
    
    # Display all prices
    for price in prices:
        print(f"{price.date.strftime('%Y-%m-%d %H:%M')}: {price.price:.4f} DKK/kWh")
    
    # Calculate average price
    if prices:
        avg_price = sum(p.price for p in prices) / len(prices)
        print(f"\nAverage price: {avg_price:.4f} DKK/kWh")

if __name__ == "__main__":
    main()
