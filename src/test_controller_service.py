import sys
from pathlib import Path

# Add src directory to path if not already there
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from controller_service import *
import unittest
import numpy as np


class TestControllerService(unittest.TestCase):
    def test_controller_minimizes_cost(self):
        initial_measurements = ThermalSystemParams(
            heating_rate_k_per_step=2.0,
            cooling_coefficient=0.04,
            ambient_temp_k=20.0,
        )
        config = ControllerServiceConfig(
            temp_min=20.0,
            temp_max=22.0,
            steps_per_hour=1,
        )
        controller = ControllerService(initial_measurements=initial_measurements, config=config)
        result = controller.get_next_action(
            current_temp=21.0,
            future_prices=[0.1, 1.0],
            ambient_temp=20.0,
            watts_on=1000.0,
        )
        self.assertEqual(result.action, Action.OFF)
    
    def test_controller_heats_when_below_temp_min(self):
        initial_measurements = ThermalSystemParams(
            heating_rate_k_per_step=2.0,
            cooling_coefficient=0.04,
            ambient_temp_k=20.0,
        )
        config = ControllerServiceConfig(
            temp_min=20.0,
            temp_max=22.0,
            steps_per_hour=1,
        )
        controller = ControllerService(initial_measurements=initial_measurements, config=config)
        result = controller.get_next_action(
            current_temp=19.0,
            future_prices=[0.1, 1.0],
            ambient_temp=20.0,
            watts_on=1000.0,
        )
        self.assertEqual(result.action, Action.ON, f"Expected ON, got {result.action}")


if __name__ == "__main__":
    unittest.main()
