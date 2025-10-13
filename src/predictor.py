from typing import Any
from dataclasses import dataclass
from enum import Enum
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PredictorInitialMeasurements:
    temperature: float  # in Kelvin
    power: float  # in Watts
    ua: float = 3.0  # Overall heat transfer coefficient in W/Km2
    ambient_temp: float = 293.15  # in Kelvin (20°C)


@dataclass
class PredictorConfig:
    temp_min: float = 323.15  # in Kelvin (50°C)
    temp_max: float = 343.15  # in Kelvin (70°C)
    steps_per_hour: int = 30


class Action(Enum):
    OFF = 0
    ON = 1


@dataclass
class PredictorResult:
    action: Action
    predicted_temperature: float
    predicted_power: float
    trajectory: list[Action]


class Predictor:
    def __init__(
        self,
        initial_measurements: PredictorInitialMeasurements,
        config: PredictorConfig,
    ):
        self.temperature = initial_measurements.temperature
        self.power = initial_measurements.power
        self.ua = initial_measurements.ua
        self.ambient_temp = initial_measurements.ambient_temp
        self.config = config

    def get_next_action(
        self, current_temp: float, future_prices: list[float]
    ) -> PredictorResult:
        """
        Determine the next action (ON/OFF) for the system
        """
        if current_temp < self.config.temp_min:
            return PredictorResult(
                Action.ON, current_temp + 5, 100, [Action.ON]
            )  # Example values
        elif current_temp > self.config.temp_max:
            return PredictorResult(
                Action.OFF, current_temp - 5, 0, [Action.OFF]
            )  # Example values
        else:
            result = self._minimize_cost(future_prices)
            return result

    def _minimize_cost(self, future_prices: list[float]) -> PredictorResult:
        """
        Optimize the action sequence to minimize cost over the prediction horizon.
        """

        def objective(actions: np.ndarray) -> float:
            sequence_price = self._price_for_sequence(
                future_prices, [Action.ON if a >= 0.5 else Action.OFF for a in actions]
            )

            outside_comfort_penalty = 0.0
            if (
                self.temperature > self.config.temp_max
                or self.temperature < self.config.temp_min
            ):
                outside_comfort_penalty = (
                    100.0  # Penalty for being outside comfort zone
                )

            total = sequence_price + outside_comfort_penalty
            print(f"-- Minimization step --")
            print(
                f"Sequence price: {sequence_price}, Outside comfort penalty: {outside_comfort_penalty}, Total: {total}"
            )
            print("Minimizing with actions:", actions)
            return total

        result = minimize(
            fun=objective,
            x0=[0.0] * len(future_prices),
            bounds=[(0, 1)] * len(future_prices),
        )

        actions = [Action.ON if a >= 0.5 else Action.OFF for a in result.x]

        return PredictorResult(
            actions[0],
            self._predict_future_temperature(
                actions[0], ambient_temp=celsius_to_kelvin(15), power_on=1500
            ),
            self.power,
            actions,
        )

    def _predict_future_temperature(
        self, action: Action, ambient_temp: float, power_on: float
    ) -> float:
        """
        Predict future temperature based on chosen action (ON/OFF).
        This is done using a simple thermal model with the UA value, ambient temperature, and current power.
        """

        power = power_on if action == Action.ON else 0

        # Simple thermal model: T_next = T_current + (Power - UA * (T_current - T_ambient)) * dt / C
        dt = 1  # time step in hours
        C = 4_184  # thermal capacity in J/K (water)
        delta_temp = (
            (power - self.ua * (self.temperature - ambient_temp)) * dt * 3600 / C
        )
        return self.temperature + delta_temp

    def plot(self, save_path: str):
        # Implement plotting logic for visualizing predictions
        pass

    def _price_for_sequence(
        self,
        future_prices: list[float],
        sequence: list[Action],
        watts: float | None = None,
    ) -> float:
        """
        Get the price for the next hour from the future prices list.
        """
        if watts is None:
            watts = self.power

        # Sequence may include fractional hours (e.g. 30 steps per hour meaning 2 minutes per step)
        total_cost = 0.0
        steps_per_hour = self.config.steps_per_hour
        index = 0
        max_index = max(len(future_prices) * steps_per_hour, len(sequence))
        while index < max_index:
            hour = index // steps_per_hour
            if hour < len(future_prices):
                price = future_prices[hour]
                action = sequence[index] if index < len(sequence) else Action.OFF
                if action == Action.ON:
                    total_cost += (watts / 1000) * price / steps_per_hour
            index += 1

        return total_cost


def celsius_to_kelvin(celsius: float) -> float:
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    return kelvin - 273.15
