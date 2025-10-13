from datetime import datetime, timedelta
from typing import List
from dataclasses import dataclass
import urllib.request
import urllib.error
import urllib.parse
import json
import logging
import time
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env from the project root (parent of src directory)
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    timestamp: datetime
    temperature: float  # Celsius

class WeatherService:
    BASE_URL = "https://dmigw.govcloud.dk/v2/metObs/collections/observation/items"

    def __init__(self, station_id: str = "06123", api_key: str = None) -> None:
        """
        Initialize the WeatherService.

        Args:
            station_id: DMI station ID. Default is "06123" (Assens/Torø)
            api_key: DMI API key. If not provided, will read from DMI_API_KEY environment variable.

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.station_id = station_id
        self.api_key = api_key or os.getenv("DMI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "DMI API key is required. Provide it as api_key parameter or set DMI_API_KEY environment variable."
            )

    def get_current_temperature(self) -> float:
        """
        Fetch the current ambient temperature (most recent hourly mean).

        Returns:
            Current temperature in Celsius

        Raises:
            Exception: If the API request fails or returns invalid data
        """
        # Try to fetch recent data, going back in time if needed
        # This handles cases where the system clock might be ahead of available data
        to_datetime = datetime.now()

        # Try different time windows going back in time
        for days_back in [0, 1, 7]:
            adjusted_to = to_datetime - timedelta(days=days_back)
            from_datetime = adjusted_to - timedelta(hours=3)

            try:
                weather_data = self._fetch_temperature_data(from_datetime, adjusted_to)

                if weather_data:
                    logger.debug(f"Found temperature data from {days_back} days back")
                    return weather_data[-1].temperature
            except Exception as e:
                logger.debug(f"Failed to fetch data from {days_back} days back: {e}")
                continue

        raise Exception("No temperature data available for any recent time period")

    def _fetch_temperature_data(
        self,
        from_datetime: datetime,
        to_datetime: datetime,
        max_retries: int = 2
    ) -> List[WeatherData]:
        """
        Fetch temperature data from DMI API with pagination support.

        Args:
            from_datetime: Start datetime
            to_datetime: End datetime
            max_retries: Maximum number of retry attempts

        Returns:
            List of WeatherData objects

        Raises:
            Exception: If the API request fails after max retries
        """
        all_weather_data = []
        parameter_id = "temp_mean_past1h"
        limit = 100  # Small limit since we only need recent data
        offset = 0

        # Format datetime as ISO strings with timezone
        from_str = from_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_str = to_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "api-key": self.api_key,
            "parameterId": parameter_id,
            "stationId": self.station_id,
            "datetime": f"{from_str}/{to_str}",
            "limit": str(limit),
            "offset": str(offset)
        }

        # Build URL with query parameters
        query_string = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        url = f"{self.BASE_URL}?{query_string}"

        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url)
                req.add_header("Accept", "application/geo+json")

                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode('utf-8'))

                features = data.get('features', [])

                if not features:
                    logger.debug(f"No data available")
                    return []

                logger.debug(f"Fetched {len(features)} records")

                # Parse features into WeatherData objects
                for feature in features:
                    props = feature.get('properties', {})
                    timestamp_str = props.get('observed')
                    value = props.get('value')

                    if timestamp_str and value is not None:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        all_weather_data.append(WeatherData(
                            timestamp=timestamp.replace(tzinfo=None),
                            temperature=float(value)
                        ))

                return all_weather_data

            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Short delay before retry
                else:
                    raise Exception(f"Failed to fetch data after {max_retries} attempts: {e}")
            except (KeyError, json.JSONDecodeError, ValueError) as e:
                raise Exception(f"Invalid response format: {e}")

        return all_weather_data
