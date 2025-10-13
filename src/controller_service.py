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
class ControllerServiceInitialMeasurements:
    thermal_system: ThermalSystemParams


@dataclass
class ControllerServiceConfig:
    temp_min: float = 323.15  # in Kelvin (50°C)
    temp_max: float = 343.15  # in Kelvin (70°C)
    steps_per_hour: int = 30


class Action(Enum):
    OFF = 0
    ON = 1


@dataclass
class ControllerServiceResult:
    action: Action
    predicted_temperature: float
    predicted_power: float
    trajectory: list[Action]


class ControllerService:
    def __init__(
        self,
        initial_measurements: ControllerServiceInitialMeasurements,
        config: ControllerServiceConfig,
    ):
        self.thermal_system = initial_measurements.thermal_system
        self.config = config

        # RLS adaptive learning state
        # Parameter vector: θ = [heating_rate_k_per_step, cooling_coefficient]
        self.theta = np.array([
            self.thermal_system.heating_rate_k_per_step,
            self.thermal_system.cooling_coefficient
        ])

        # Covariance matrix (initialized with high values for initial uncertainty)
        self.P = np.eye(2) * 100.0

        # Forgetting factor (0.95-0.99): higher = trust old data more
        self.lambda_rls = 0.98

        # History tracking
        self.observation_history: list[tuple[float, Action, float]] = []  # (temp_before, action, temp_after)
        self.prediction_errors: list[float] = []

        # Parameter bounds for physical validity
        self.heating_rate_bounds = (0.5, 3.0)  # K/step - tighter bounds for stability
        self.cooling_coeff_bounds = (0.010, 0.030)  # Must be positive - tighter bounds

    def get_next_action(
        self,
        current_temp: float,
        future_prices: list[float],
        ambient_temp: float,
        watts_on: float,
    ) -> ControllerServiceResult:
        """
        Determine the next action (ON/OFF) for the system
        """
        if current_temp < self.config.temp_min:
            return ControllerServiceResult(
                Action.ON, current_temp + 5, 100, [Action.ON]
            )  # Example values
        elif current_temp > self.config.temp_max:
            return ControllerServiceResult(
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
    ) -> ControllerServiceResult:
        """
        Optimize the action sequence to minimize cost over the prediction horizon.
        Now properly simulates temperature trajectory and enforces constraints.
        """

        def objective(actions: np.ndarray) -> float:
            # Convert to action sequence
            action_sequence = [Action.ON if a >= 0.5 else Action.OFF for a in actions]

            # Calculate electricity cost
            sequence_price = self._price_for_sequence(
                future_prices=future_prices,
                sequence=action_sequence,
                watts_on=watts_on,
            )

            # Simulate temperature trajectory and penalize constraint violations
            temp = current_temp
            comfort_penalty = 0.0

            for action in action_sequence:
                # Predict next temperature
                temp = self._predict_future_temperature(action, temp)

                # Penalize violations of temperature bounds (hard constraints)
                if temp < self.config.temp_min:
                    comfort_penalty += 1000.0 * (self.config.temp_min - temp) ** 2
                elif temp > self.config.temp_max:
                    comfort_penalty += 1000.0 * (temp - self.config.temp_max) ** 2

            total = sequence_price + comfort_penalty
            return total

        # Limit optimization horizon (max 24 hours look-ahead) 
        optimization_horizon_hours = min(24, len(future_prices))
        optimization_steps = optimization_horizon_hours * self.config.steps_per_hour

        # Create initial guess based on price patterns
        # Prefer heating during cheaper periods
        avg_price = sum(future_prices[:optimization_horizon_hours]) / optimization_horizon_hours if optimization_horizon_hours > 0 else 1.0
        x0 = []
        for hour_idx in range(optimization_horizon_hours):
            for _ in range(self.config.steps_per_hour):
                if hour_idx < len(future_prices):
                    # Turn on more likely if price is below average
                    initial_action = 1.0 if future_prices[hour_idx] < avg_price else 0.0
                    x0.append(initial_action)

        # Ensure x0 has correct length
        assert(len(x0) == optimization_steps)

        # Optimize action sequence
        result = minimize(
            fun=objective,
            x0=x0,
            bounds=[(0, 1)] * len(x0),
            options={'disp': False, 'maxiter': 100}  # Limit iterations for speed
        )

        actions = [Action.ON if a >= 0.5 else Action.OFF for a in result.x]

        return ControllerServiceResult(
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

    def update_model(self, temp_before: float, action_taken: Action, temp_after: float):
        """
        Update model parameters using Recursive Least Squares (RLS).

        Args:
            temp_before: Temperature before action (Kelvin)
            action_taken: Action that was taken (ON/OFF)
            temp_after: Observed temperature after action (Kelvin)
        """
        # Store observation
        self.observation_history.append((temp_before, action_taken, temp_after))

        # Need at least one observation to update
        if len(self.observation_history) < 2:
            return

        # Observed temperature change
        delta_temp_observed = temp_after - temp_before

        # Predicted temperature change using current parameters
        delta_temp_predicted = self._predict_delta_temp(action_taken, temp_before)

        # Prediction error
        prediction_error = delta_temp_observed - delta_temp_predicted
        self.prediction_errors.append(prediction_error)

        # Build regressor vector φ
        # Model: ΔT = heating_rate × action - cooling_coefficient × (T - T_ambient)
        temp_diff = temp_before - self.thermal_system.ambient_temp_k
        action_value = 1.0 if action_taken == Action.ON else 0.0

        phi = np.array([
            action_value,           # Coefficient for heating_rate
            -temp_diff              # Coefficient for cooling_coefficient
        ])

        # RLS update
        # K = P·φ / (λ + φᵀ·P·φ)
        P_phi = self.P @ phi
        denominator = self.lambda_rls + phi @ P_phi

        if abs(denominator) > 1e-10:  # Avoid division by zero
            K = P_phi / denominator

            # Update parameters: θ = θ + K·e
            self.theta = self.theta + K * prediction_error

            # Update covariance: P = (P - K·φᵀ·P) / λ
            self.P = (self.P - np.outer(K, P_phi)) / self.lambda_rls

            # Apply parameter bounds for physical validity
            self.theta[0] = np.clip(self.theta[0], *self.heating_rate_bounds)
            self.theta[1] = np.clip(self.theta[1], *self.cooling_coeff_bounds)

            # Update thermal system parameters with learned values
            self.thermal_system.heating_rate_k_per_step = self.theta[0]
            self.thermal_system.cooling_coefficient = self.theta[1]

    def _predict_delta_temp(self, action: Action, current_temp: float) -> float:
        """Helper to predict temperature change (not absolute temperature)"""
        temp_diff = current_temp - self.thermal_system.ambient_temp_k
        cooling_delta = -self.thermal_system.cooling_coefficient * temp_diff
        heating_delta = self.thermal_system.heating_rate_k_per_step if action == Action.ON else 0.0
        return heating_delta + cooling_delta

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
