from datetime import date, timedelta
import logging
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from controller_service import *

def test_controller_service_heats_when_min_temp_reached():
    """Test the ControllerService's get_next_action method."""

    config = ControllerConfiguration(
        enable_fcr_d=False
    )

    params = ControllerParameters(
        target_temp=60, # °C
        min_temp=50, # °C
        max_temp=70, # °C
        ua=0.0011, # kW/K
        cp=4.18, # Specific heat (kJ/kg·K)
        eta=0.95, # Efficiency
        steps_per_hour=4, # Every 15 minutes
        hours_ahead=24, # Look one day ahead
    )

    controller = ControllerService(config=config, params=params)

    next_action = controller.get_next_action(
        current_temp=50, # °C
        ambient_temp=20, # °C
        future_prices=[0.5]*96, # Flat price forecast (96 steps for 24 hours at 15 min intervals)
    )

    if next_action != Action.ON:
        print("✗ Controller did not turn ON the heater when it should have.")
        raise AssertionError("Expected Action.ON but got Action.OFF")

