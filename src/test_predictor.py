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

if __name__ == "__main__":
    unittest.main()