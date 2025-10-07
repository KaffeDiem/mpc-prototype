"""Tests for the PricesService to fetch Danish electricity prices."""

from datetime import date, timedelta
import logging
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prices_service import PricesService

# Set up logging to see any warnings about missing dates
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_get_today_prices():
    """Test fetching today's prices."""
    service = PricesService(region="DK2")
    today = date.today()
    print(f"\n=== Test: Fetching prices for today ({today}) ===")
    try:
        prices = service.get_prices(today, today)
        print(f"✓ Retrieved {len(prices)} hourly prices")
        assert len(prices) == 24, f"Expected 24 prices, got {len(prices)}"
        if prices:
            print(f"  First price: {prices[0].date} - {prices[0].price:.4f} DKK/kWh")
            print(f"  Last price: {prices[-1].date} - {prices[-1].price:.4f} DKK/kWh")
            assert prices[0].price > 0, "Price should be positive"
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

def test_get_today_and_tomorrow_prices():
    """Test fetching prices for today and tomorrow."""
    service = PricesService(region="DK2")
    today = date.today()
    tomorrow = today + timedelta(days=1)
    print(f"\n=== Test: Fetching prices from {today} to {tomorrow} ===")
    try:
        prices = service.get_prices(today, tomorrow)
        print(f"✓ Retrieved {len(prices)} hourly prices across {(tomorrow - today).days + 1} days")
        # Should have 24-48 prices depending on whether tomorrow's data is available
        assert len(prices) >= 24, f"Expected at least 24 prices, got {len(prices)}"
        assert len(prices) <= 48, f"Expected at most 48 prices, got {len(prices)}"
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

def test_reject_future_dates():
    """Test that fetching beyond tomorrow is rejected."""
    service = PricesService(region="DK2")
    today = date.today()
    day_after_tomorrow = today + timedelta(days=2)
    print(f"\n=== Test: Attempting to fetch beyond tomorrow ({day_after_tomorrow}) ===")
    try:
        prices = service.get_prices(today, day_after_tomorrow)
        print(f"✗ Should have raised ValueError but got {len(prices)} prices")
        raise AssertionError("Expected ValueError for date beyond tomorrow")
    except ValueError as e:
        print(f"✓ Expected error caught: {e}")
        assert "cannot be beyond tomorrow" in str(e)

def test_date_range_with_potential_missing_dates():
    """Test fetching a date range that may have some missing dates."""
    service = PricesService(region="DK2")
    today = date.today()
    week_ago = today - timedelta(days=7)
    print(f"\n=== Test: Fetching prices from {week_ago} to {today} ===")
    try:
        prices = service.get_prices(week_ago, today)
        print(f"✓ Retrieved {len(prices)} hourly prices")
        # Should have prices for most days (at least some data)
        assert len(prices) >= 24, f"Expected at least 24 prices, got {len(prices)}"
        # At most 8 days * 24 hours = 192 prices
        assert len(prices) <= 192, f"Expected at most 192 prices, got {len(prices)}"
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

def test_dk1_region():
    """Test fetching prices for DK1 region."""
    service = PricesService(region="DK1")
    today = date.today()
    print(f"\n=== Test: Fetching DK1 region prices for today ({today}) ===")
    try:
        prices = service.get_prices(today, today)
        print(f"✓ Retrieved {len(prices)} hourly prices for DK1")
        assert len(prices) == 24, f"Expected 24 prices, got {len(prices)}"
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

def test_invalid_date_range():
    """Test that invalid date ranges are rejected."""
    service = PricesService(region="DK2")
    today = date.today()
    yesterday = today - timedelta(days=1)
    print(f"\n=== Test: Attempting invalid date range (from > to) ===")
    try:
        prices = service.get_prices(today, yesterday)
        print(f"✗ Should have raised ValueError but got {len(prices)} prices")
        raise AssertionError("Expected ValueError for invalid date range")
    except ValueError as e:
        print(f"✓ Expected error caught: {e}")
        assert "cannot be after" in str(e)

def run_all_tests():
    """Run all tests."""
    tests = [
        test_get_today_prices,
        test_get_today_and_tomorrow_prices,
        test_reject_future_dates,
        test_date_range_with_potential_missing_dates,
        test_dk1_region,
        test_invalid_date_range,
    ]
    
    print("=" * 70)
    print("Running PricesService Tests")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
