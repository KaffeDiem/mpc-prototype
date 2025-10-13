from predictor import *
import unittest


class TestPredictor(unittest.TestCase):
    def test_predictor_price_for_sequence(self):
        thermal_system = ThermalSystemParams.water_heater()
        initial_measurements = PredictorInitialMeasurements(thermal_system=thermal_system)
        config = PredictorConfig(steps_per_hour=4)
        predictor = Predictor(initial_measurements, config)

        future_prices = [10.0]
        sequence = [Action.ON, Action.OFF, Action.ON, Action.OFF]

        cost = predictor._price_for_sequence(future_prices, sequence, watts_on=1_000)
        print(f"Calculated cost: {cost}")
        assert cost == 5.0, f"Cost should be 5.0, got {cost}"

    def test_minimize_cost_turn_off(self):
        thermal_system = ThermalSystemParams.water_heater()
        initial_measurements = PredictorInitialMeasurements(thermal_system=thermal_system)
        config = PredictorConfig(steps_per_hour=4)
        predictor = Predictor(initial_measurements, config)

        prices = [100.0, 50]
        pred_result = predictor._minimize_cost(prices, ambient_temp=20, watts_on=1_000, current_temp=50)
        assert (
            pred_result.action == Action.OFF
        ), f"Action should be OFF, got {pred_result.action}"

    def test_temp_below_min(self):
        thermal_system = ThermalSystemParams.water_heater()
        initial_measurements = PredictorInitialMeasurements(thermal_system=thermal_system)
        config = PredictorConfig(
            steps_per_hour=4,
            temp_min=celsius_to_kelvin(30),
            temp_max=celsius_to_kelvin(70),
        )
        predictor = Predictor(initial_measurements, config)

        prices = [10.0, 50]
        pred_result = predictor.get_next_action(
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
        initial_measurements = PredictorInitialMeasurements(thermal_system=thermal_system)
        config = PredictorConfig(steps_per_hour=12)
        predictor = Predictor(initial_measurements, config)

        current_temp = celsius_to_kelvin(50)  # 323.15K
        next_temp = predictor._predict_future_temperature(Action.ON, current_temp)

        # Should heat up (heating_rate - cooling due to temp difference)
        assert next_temp > current_temp, "Temperature should increase when heater is ON"

    def test_thermal_model_cooling(self):
        """Test that temperature decreases when heater is OFF"""
        thermal_system = ThermalSystemParams.water_heater(
            heating_rate_k_per_step=1.0,
            cooling_coefficient=0.01,
            ambient_temp_celsius=20.0
        )
        initial_measurements = PredictorInitialMeasurements(thermal_system=thermal_system)
        config = PredictorConfig(steps_per_hour=12)
        predictor = Predictor(initial_measurements, config)

        current_temp = celsius_to_kelvin(50)  # Above ambient
        next_temp = predictor._predict_future_temperature(Action.OFF, current_temp)

        # Should cool down towards ambient
        assert next_temp < current_temp, "Temperature should decrease when heater is OFF and above ambient"

    def test_radiator_vs_water_heater(self):
        """Test that radiator heats faster than water heater"""
        radiator_system = ThermalSystemParams.electric_radiator()
        water_system = ThermalSystemParams.water_heater()

        config = PredictorConfig(steps_per_hour=12)

        radiator_predictor = Predictor(
            PredictorInitialMeasurements(thermal_system=radiator_system), config
        )
        water_predictor = Predictor(
            PredictorInitialMeasurements(thermal_system=water_system), config
        )

        start_temp = celsius_to_kelvin(25)

        radiator_temp = radiator_predictor._predict_future_temperature(Action.ON, start_temp)
        water_temp = water_predictor._predict_future_temperature(Action.ON, start_temp)

        radiator_delta = radiator_temp - start_temp
        water_delta = water_temp - start_temp

        # Radiator should heat faster (higher heating_rate_k_per_step)
        assert radiator_delta > water_delta, "Radiator should heat faster than water heater"


if __name__ == "__main__":
    unittest.main()
