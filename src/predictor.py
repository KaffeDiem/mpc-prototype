from typing import Any
from dataclasses import dataclass
from enum import Enum
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np


def celsius_to_kelvin(celsius: float) -> float:
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    return kelvin - 273.15


@dataclass
class ThermalSystemParams:
    """
    Simplified thermal system parameters that work for both water heaters and radiators.

    All temperatures in Kelvin, making calculations consistent and future-proof.
    """
    heating_rate_k_per_step: float  # Temperature increase per step when heater is ON (K/step)
    cooling_coefficient: float  # Cooling rate coefficient (fraction of temp difference lost per step)
    ambient_temp_k: float  # Ambient temperature (Kelvin)

    @classmethod
    def water_heater(
        cls,
        heating_rate_k_per_step: float = 0.5,  # Slow heating (high thermal mass)
        cooling_coefficient: float = 0.02,  # Slow cooling (well insulated)
        ambient_temp_celsius: float = 20.0,
    ) -> "ThermalSystemParams":
        """Factory method for water heater systems (high thermal mass, slow response)"""
        return cls(
            heating_rate_k_per_step=heating_rate_k_per_step,
            cooling_coefficient=cooling_coefficient,
            ambient_temp_k=celsius_to_kelvin(ambient_temp_celsius),
        )

    @classmethod
    def electric_radiator(
        cls,
        heating_rate_k_per_step: float = 2.0,  # Fast heating (low thermal mass)
        cooling_coefficient: float = 0.1,  # Fast cooling (designed to dissipate heat)
        ambient_temp_celsius: float = 20.0,
    ) -> "ThermalSystemParams":
        """Factory method for electric radiator systems (low thermal mass, fast response)"""
        return cls(
            heating_rate_k_per_step=heating_rate_k_per_step,
            cooling_coefficient=cooling_coefficient,
            ambient_temp_k=celsius_to_kelvin(ambient_temp_celsius),
        )


@dataclass
class PredictorInitialMeasurements:
    thermal_system: ThermalSystemParams


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
        self.thermal_system = initial_measurements.thermal_system
        self.config = config

    def get_next_action(
        self,
        current_temp: float,
        future_prices: list[float],
        ambient_temp: float,
        watts_on: float,
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
            result = self._minimize_cost(
                future_prices, ambient_temp, watts_on, current_temp
            )
            return result

    def _minimize_cost(
        self,
        future_prices: list[float],
        ambient_temp: float,
        watts_on: float,
        current_temp: float,
    ) -> PredictorResult:
        """
        Optimize the action sequence to minimize cost over the prediction horizon.
        """

        def objective(actions: np.ndarray) -> float:
            sequence_price = self._price_for_sequence(
                future_prices=future_prices,
                sequence=[Action.ON if a >= 0.5 else Action.OFF for a in actions],
                watts_on=watts_on,
            )

            outside_comfort_penalty = 0.0
            if (
                current_temp > self.config.temp_max
                or current_temp < self.config.temp_min
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
            self._predict_future_temperature(actions[0], current_temp),
            watts_on,
            actions,
        )

    def _predict_future_temperature(
        self, action: Action, current_temp: float
    ) -> float:
        """
        Predict future temperature based on chosen action (ON/OFF).

        Simple thermal model:
        - When ON: ΔT = heating_rate - cooling_coefficient × (T - T_ambient)
        - When OFF: ΔT = -cooling_coefficient × (T - T_ambient)

        All calculations in Kelvin.
        """
        # Temperature difference from ambient
        temp_diff = current_temp - self.thermal_system.ambient_temp_k

        # Heat loss (always occurs)
        cooling_delta = -self.thermal_system.cooling_coefficient * temp_diff

        # Heating (only when ON)
        heating_delta = self.thermal_system.heating_rate_k_per_step if action == Action.ON else 0.0

        # Total temperature change
        delta_temp = heating_delta + cooling_delta

        return current_temp + delta_temp

    def plot(self, save_path: str):
        # Implement plotting logic for visualizing predictions
        pass

    def _price_for_sequence(
        self,
        future_prices: list[float],
        sequence: list[Action],
        watts_on: float,
    ) -> float:
        """
        Get the price for the next hour from the future prices list.
        """

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
                    total_cost += (watts_on / 1000) * price / steps_per_hour
            index += 1

        return total_cost
