from controller_service import *
import unittest


class TestControllerService(unittest.TestCase):
    def test_controller_service_price_for_sequence(self):
        thermal_system = ThermalSystemParams.water_heater()
        initial_measurements = ControllerServiceInitialMeasurements(thermal_system=thermal_system)
        config = ControllerServiceConfig(steps_per_hour=4)
        controller_service = ControllerService(initial_measurements, config)

        future_prices = [10.0]
        sequence = [Action.ON, Action.OFF, Action.ON, Action.OFF]

        cost = controller_service._price_for_sequence(future_prices, sequence, watts_on=1_000)
        print(f"Calculated cost: {cost}")
        assert cost == 5.0, f"Cost should be 5.0, got {cost}"

    def test_minimize_cost_turn_off(self):
        thermal_system = ThermalSystemParams.water_heater()
        initial_measurements = ControllerServiceInitialMeasurements(thermal_system=thermal_system)
        config = ControllerServiceConfig(steps_per_hour=4)
        controller_service = ControllerService(initial_measurements, config)

        prices = [100.0, 50]
        pred_result = controller_service._minimize_cost(prices, ambient_temp=celsius_to_kelvin(20), watts_on=1_000, current_temp=celsius_to_kelvin(50))
        assert (
            pred_result.action == Action.OFF
        ), f"Action should be OFF, got {pred_result.action}"

    def test_temp_below_min(self):
        thermal_system = ThermalSystemParams.water_heater()
        initial_measurements = ControllerServiceInitialMeasurements(thermal_system=thermal_system)
        config = ControllerServiceConfig(
            steps_per_hour=4,
            temp_min=celsius_to_kelvin(30),
            temp_max=celsius_to_kelvin(70),
        )
        controller_service = ControllerService(initial_measurements, config)

        prices = [10.0, 50]
        pred_result = controller_service.get_next_action(
            celsius_to_kelvin(25), future_prices=prices, ambient_temp=20, watts_on=1_000
        )
        assert (
            pred_result.action == Action.ON
        ), f"Action should be ON, got {pred_result.action}"

    def test_thermal_model_heating(self):
        """Test that temperature increases when heater is ON"""
        thermal_system = ThermalSystemParams.water_heater(
            heating_rate_k_per_step=1.0,
            cooling_coefficient=0.01,
            ambient_temp_celsius=20.0
        )
        initial_measurements = ControllerServiceInitialMeasurements(thermal_system=thermal_system)
        config = ControllerServiceConfig(steps_per_hour=12)
        controller_service = ControllerService(initial_measurements, config)

        current_temp = celsius_to_kelvin(50)  # 323.15K
        ambient_temp = celsius_to_kelvin(20)
        next_temp = controller_service._predict_future_temperature(Action.ON, current_temp, ambient_temp)

        # Should heat up (heating_rate - cooling due to temp difference)
        assert next_temp > current_temp, "Temperature should increase when heater is ON"

    def test_thermal_model_cooling(self):
        """Test that temperature decreases when heater is OFF"""
        thermal_system = ThermalSystemParams.water_heater(
            heating_rate_k_per_step=1.0,
            cooling_coefficient=0.01,
            ambient_temp_celsius=20.0
        )
        initial_measurements = ControllerServiceInitialMeasurements(thermal_system=thermal_system)
        config = ControllerServiceConfig(steps_per_hour=12)
        controller_service = ControllerService(initial_measurements, config)

        current_temp = celsius_to_kelvin(50)  # Above ambient
        ambient_temp = celsius_to_kelvin(20)
        next_temp = controller_service._predict_future_temperature(Action.OFF, current_temp, ambient_temp)

        # Should cool down towards ambient
        assert next_temp < current_temp, "Temperature should decrease when heater is OFF and above ambient"

    def test_radiator_vs_water_heater(self):
        """Test that radiator heats faster than water heater"""
        radiator_system = ThermalSystemParams.electric_radiator()
        water_system = ThermalSystemParams.water_heater()

        config = ControllerServiceConfig(steps_per_hour=12)

        radiator_controller_service = ControllerService(
            ControllerServiceInitialMeasurements(thermal_system=radiator_system), config
        )
        water_controller_service = ControllerService(
            ControllerServiceInitialMeasurements(thermal_system=water_system), config
        )

        start_temp = celsius_to_kelvin(25)
        ambient_temp = celsius_to_kelvin(20)

        radiator_temp = radiator_controller_service._predict_future_temperature(Action.ON, start_temp, ambient_temp)
        water_temp = water_controller_service._predict_future_temperature(Action.ON, start_temp, ambient_temp)

        radiator_delta = radiator_temp - start_temp
        water_delta = water_temp - start_temp

        # Radiator should heat faster (higher heating_rate_k_per_step)
        assert radiator_delta > water_delta, "Radiator should heat faster than water heater"


if __name__ == "__main__":
    unittest.main()
