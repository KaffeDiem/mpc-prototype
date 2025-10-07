from datetime import date, datetime, timedelta
from typing import List, Literal
from dataclasses import dataclass
import urllib.request
import urllib.error
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class Price:
    date: date
    price: float

Region = Literal["DK1", "DK2"]

class PricesService(): 
    BASE_URL = "https://www.elprisenligenu.dk/api/v1/prices"
    
    def __init__(self, region: Region = "DK2") -> None:
        """
        Initialize the PricesService.
        
        Args:
            region: Danish price region. DK1 = West of Great Belt (Aarhus), 
                    DK2 = East of Great Belt (Copenhagen). Default is DK2.
        """
        self.region = region

    def get_prices(self, from_date: date, to_date: date) -> List[Price]:
        """
        Fetch electricity prices for a date range.
        
        Args:
            from_date: Start date (inclusive)
            to_date: End date (inclusive). Maximum is tomorrow's date.
            
        Returns:
            List of Price objects with date and price (DKK per kWh)
            
        Raises:
            ValueError: If to_date is beyond tomorrow
        """
        # Validate that to_date is not beyond tomorrow
        tomorrow = date.today() + timedelta(days=1)
        if to_date > tomorrow:
            raise ValueError(f"to_date cannot be beyond tomorrow ({tomorrow}). Got: {to_date}")
        
        if from_date > to_date:
            raise ValueError(f"from_date ({from_date}) cannot be after to_date ({to_date})")
        
        all_prices = []
        current_date = from_date
        missing_dates = []
        
        # Fetch prices for each date in the range
        while current_date <= to_date:
            try:
                daily_prices = self._fetch_prices_for_date(current_date)
                all_prices.extend(daily_prices)
            except Exception as e:
                logger.warning(f"Missing or unavailable prices for date {current_date}: {e}")
                missing_dates.append(current_date)
            
            current_date += timedelta(days=1)
        
        # Log summary of missing dates
        if missing_dates:
            logger.warning(
                f"Missing prices for {len(missing_dates)} date(s): "
                f"{', '.join(str(d) for d in missing_dates)}"
            )
        
        return all_prices
    
    def _fetch_prices_for_date(self, target_date: date) -> List[Price]:
        """
        Fetch prices for a specific date from the API.
        
        Args:
            target_date: The date to fetch prices for
            
        Returns:
            List of Price objects for each hour of the day
            
        Raises:
            Exception: If the API request fails or returns invalid data
        """
        # Format: YYYY/MM-DD_REGION.json
        url = (
            f"{self.BASE_URL}/"
            f"{target_date.year}/"
            f"{target_date.month:02d}-{target_date.day:02d}_"
            f"{self.region}.json"
        )
        
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            prices = []
            for entry in data:
                # Parse the start time to get the date and hour
                time_start = datetime.fromisoformat(entry['time_start'])
                # Use DKK_per_kWh as the price
                price = entry['DKK_per_kWh']
                
                prices.append(Price(
                    date=time_start.replace(tzinfo=None),
                    price=price
                ))
            
            return prices
            
        except urllib.error.HTTPError as e:
            raise Exception(f"HTTP error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise Exception(f"URL error: {e.reason}")
        except (KeyError, json.JSONDecodeError) as e:
            raise Exception(f"Invalid response format: {e}")