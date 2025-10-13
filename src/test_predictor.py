from predictor import *
import unittest

class TestPredictor(unittest.TestCase):
    def test_predictor_price_for_sequence(self):
        initial_measurements = PredictorInitialMeasurements(temperature=300, power=1_000)
        config = PredictorConfig(steps_per_hour=4)
        predictor = Predictor(initial_measurements, config)

        future_prices = [10.0] 
        sequence = [Action.ON, Action.OFF, Action.ON, Action.OFF]

        cost = predictor._price_for_sequence(future_prices, sequence)
        print(f"Calculated cost: {cost}")
        assert cost == 5.0, f"Cost should be 5.0, got {cost}"

    def test_minimize_cost_turn_off(self):
        initial_measurements = PredictorInitialMeasurements(temperature=300, power=1_000)
        config = PredictorConfig(steps_per_hour=4)
        predictor = Predictor(initial_measurements, config)
        
        prices = [100.0, 50]
        pred_result = predictor._minimize_cost(prices)
        assert pred_result.action == Action.OFF, f"Action should be OFF, got {pred_result.action}"

    def test_minimize_cost_turn_on(self):
        initial_measurements = PredictorInitialMeasurements(temperature=300, power=1_000)
        config = PredictorConfig(steps_per_hour=4)
        predictor = Predictor(initial_measurements, config)
        
        prices = [10.0, 50]
        pred_result = predictor._minimize_cost(prices)
        assert pred_result.action == Action.ON, f"Action should be ON, got {pred_result.action}"


if __name__ == "__main__":
    unittest.main()