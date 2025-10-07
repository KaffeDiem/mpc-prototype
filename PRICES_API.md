# Electricity Prices API Implementation

## Overview
Implementation of the Danish electricity prices API from [elprisenligenu.dk](https://www.elprisenligenu.dk/elpris-api).

## Features

### PricesService Class
- **Region Support**: DK1 (West of Great Belt/Aarhus) and DK2 (East of Great Belt/Copenhagen)
- **Date Range Queries**: Fetch prices for any date range up to tomorrow
- **Max Date Validation**: Automatically enforces that queries cannot exceed tomorrow's date
- **Missing Date Logging**: Logs warnings when prices are unavailable for specific dates
- **Hourly Price Data**: Returns 24 hourly prices per day in DKK per kWh

## Usage

```python
from datetime import date
from prices_service import PricesService
import logging

# Set up logging to see warnings about missing dates
logging.basicConfig(level=logging.WARNING)

# Initialize for DK2 region (Copenhagen)
service = PricesService(region="DK2")

# Get today's prices
today = date.today()
prices = service.get_prices(today, today)

# Each price contains:
# - date: datetime with hour
# - price: float (DKK per kWh, excluding VAT and taxes)
for price in prices:
    print(f"{price.date}: {price.price} DKK/kWh")
```

## API Details

### Endpoint Format
```
GET https://www.elprisenligenu.dk/api/v1/prices/[YEAR]/[MONTH]-[DAY]_[REGION].json
```

### Response Format
Each day returns 24 hourly entries:
```json
{
  "DKK_per_kWh": 1.3929,
  "EUR_per_kWh": 0.18729,
  "EXR": 7.437118,
  "time_start": "2022-11-25T00:00:00+01:00",
  "time_end": "2022-11-25T01:00:00+01:00"
}
```

## Important Notes

1. **Historical Data**: Only available from November 1, 2022 onwards
2. **Tomorrow's Prices**: Available from 13:00 (1 PM) the day before
3. **Prices Exclude**: VAT and additional electricity taxes
4. **Currency**: Converted from EUR to DKK using daily exchange rates
5. **Max Date**: Cannot query beyond tomorrow (enforced by the service)

## Error Handling

- **ValueError**: Raised if `to_date` exceeds tomorrow or if `from_date` > `to_date`
- **Missing Dates**: Logged as warnings but don't stop the entire request
- **API Errors**: HTTP and connection errors are caught and logged

## Example Output

```
=== Fetching prices for today (2025-10-07) ===
Retrieved 24 hourly prices
First price: 2025-10-07 00:00:00 - 0.7486 DKK/kWh
Last price: 2025-10-07 23:00:00 - 0.7236 DKK/kWh
```

## Running the Example

```bash
# Run the simple example (fetches today and tomorrow's prices)
uv run src/main.py

# Run the full test suite
uv run tests/test_prices_service.py
```

The test suite includes:
- ✅ Fetching today's prices
- ✅ Fetching today and tomorrow's prices
- ✅ Validating max date (tomorrow) enforcement
- ✅ Handling date ranges with potential missing data
- ✅ Testing both DK1 and DK2 regions
- ✅ Validating error handling for invalid date ranges

## Attribution
When using this data publicly, please credit: [Elprisen lige nu.dk](https://www.elprisenligenu.dk/)
