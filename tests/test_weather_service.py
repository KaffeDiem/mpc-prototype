"""Tests for the WeatherService to fetch Danish weather data from DMI."""

import logging
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weather_service import WeatherService

# Set up logging to see any warnings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_get_current_temperature():
    """Test fetching current temperature."""
    service = WeatherService(station_id="06123")
    print(f"\n=== Test: Fetching current temperature ===")
    try:
        temperature = service.get_current_temperature()
        print(f"✓ Retrieved current temperature: {temperature:.1f}°C")
        # Sanity check: temperature should be reasonable for Denmark
        assert -30 < temperature < 40, f"Temperature {temperature}°C outside reasonable range"
        assert isinstance(temperature, float), "Temperature should be a float"
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

def test_get_current_temperature_multiple_times():
    """Test fetching current temperature multiple times."""
    service = WeatherService(station_id="06123")
    print(f"\n=== Test: Fetching current temperature multiple times ===")
    try:
        temp1 = service.get_current_temperature()
        temp2 = service.get_current_temperature()
        print(f"✓ First call: {temp1:.1f}°C")
        print(f"✓ Second call: {temp2:.1f}°C")
        # Both should be reasonable temperatures
        assert -30 < temp1 < 40, f"Temperature {temp1}°C outside reasonable range"
        assert -30 < temp2 < 40, f"Temperature {temp2}°C outside reasonable range"
        # They might be the same or very close
        temp_diff = abs(temp1 - temp2)
        print(f"  Temperature difference: {temp_diff:.1f}°C")
        assert temp_diff < 5, "Temperature difference too large between consecutive calls"
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

def test_different_station():
    """Test fetching temperature from a different station."""
    # Try Copenhagen Airport station (06180)
    service = WeatherService(station_id="06180")
    print(f"\n=== Test: Fetching from Copenhagen Airport station (06180) ===")
    try:
        temperature = service.get_current_temperature()
        print(f"✓ Retrieved temperature: {temperature:.1f}°C")
        assert -30 < temperature < 40, f"Temperature {temperature}°C outside reasonable range"
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

def run_all_tests():
    """Run all tests."""
    tests = [
        test_get_current_temperature,
        test_get_current_temperature_multiple_times,
        test_different_station,
    ]

    print("=" * 70)
    print("Running WeatherService Tests")
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
