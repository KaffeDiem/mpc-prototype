from typing import Any
from dataclasses import dataclass
from enum import Enum
from scipy.optimize import differential_evolution
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


@dataclass
class ControllerServiceConfig:
    temp_min: float  
    temp_max: float 
    steps_per_hour: int 


class Action(Enum):
    OFF = 0
    ON = 1


@dataclass
class TrajectoryStep:
    action: Action
    predicted_temperature: float  # Kelvin
    predicted_cost: float  # DKK for this step
    price: float  # DKK/kWh for this step


@dataclass
class ControllerServiceResult:
    predicted_power: float
    trajectory: list[TrajectoryStep]


class ControllerService:
    def __init__(
        self,
        initial_measurements: ThermalSystemParams,
        config: ControllerServiceConfig,
    ):
        self.thermal_system = initial_measurements
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
        Determine the next action (ON/OFF) for the system using MPC optimization with
        soft temperature constraints enforced via large penalties in the objective.
        """

        def objective(actions: np.ndarray) -> float:
            TEMP_VIOLATION_PENALTY = 1_000

            temperature = current_temp
            cost = 0.0
            for i, action in enumerate(actions):
                action = Action.ON if action > 0.5 else Action.OFF
                temperature = self._predict_future_temperature(action, temperature, ambient_temp)
                price_index = i // self.config.steps_per_hour
                cost += future_prices[price_index] * watts_on / 1000.0

                if temperature < self.config.temp_min:
                    penalty = TEMP_VIOLATION_PENALTY * (self.config.temp_min - temperature) ** 3
                    cost += penalty
                if temperature > self.config.temp_max:
                    penalty = TEMP_VIOLATION_PENALTY * (temperature - self.config.temp_max) ** 3
                    cost += penalty

            return cost

        # Limit optimization horizon (max 24 hours look-ahead) 
        optimization_horizon_hours = min(24, len(future_prices))
        optimization_steps = optimization_horizon_hours * self.config.steps_per_hour

        x0 = np.full(optimization_steps, 0.5)

        bounds = [(0, 1)] * optimization_steps
        result = differential_evolution(objective, bounds, maxiter=5000, polish=False)

        actions = [Action.ON if action > 0.5 else Action.OFF for action in result.x]

        # Build detailed trajectory with temperature and cost predictions
        trajectory = self._build_trajectory_with_details(
            actions=actions,
            initial_temp=current_temp,
            ambient_temp=ambient_temp,
            future_prices=future_prices,
            watts_on=watts_on
        )

        return ControllerServiceResult(
            predicted_power=watts_on if actions[0] == Action.ON else 0.0,
            trajectory=trajectory,
        )
    

    def _predict_future_temperature(
        self, action: Action, current_temp: float, ambient_temp: float
    ) -> float:
        """
        Predict future temperature based on chosen action (ON/OFF).

        Simple thermal model:
        - When ON: ΔT = heating_rate - cooling_coefficient × (T - T_ambient)
        - When OFF: ΔT = -cooling_coefficient × (T - T_ambient)

        All calculations in Kelvin.
        """
        # Temperature difference from ambient
        temp_diff = current_temp - ambient_temp

        # Heat loss (always occurs)
        cooling_delta = -self.thermal_system.cooling_coefficient * temp_diff

        # Heating (only when ON)
        heating_delta = self.thermal_system.heating_rate_k_per_step if action == Action.ON else 0.0

        # Total temperature change
        delta_temp = heating_delta + cooling_delta

        return current_temp + delta_temp

    def _simulate_temperature_trajectory(
        self, action_sequence: list[Action], initial_temp: float, ambient_temp: float
    ) -> list[float]:
        """
        Simulate the temperature trajectory over the entire action sequence.

        Args:
            action_sequence: Sequence of actions (ON/OFF) to simulate
            initial_temp: Starting temperature (Kelvin)
            ambient_temp: Ambient temperature (Kelvin)

        Returns:
            List of temperatures at each step (Kelvin)
        """
        trajectory = []
        current_temp = initial_temp

        for action in action_sequence:
            # Predict next temperature based on action
            current_temp = self._predict_future_temperature(action, current_temp, ambient_temp)
            trajectory.append(current_temp)

        return trajectory

    def update_model(self, temp_before: float, action_taken: Action, temp_after: float, ambient_temp: float):
        """
        Update model parameters using Recursive Least Squares (RLS).

        Args:
            temp_before: Temperature before action (Kelvin)
            action_taken: Action that was taken (ON/OFF)
            temp_after: Observed temperature after action (Kelvin)
            ambient_temp: Current ambient temperature (Kelvin)
        """
        # Store observation
        self.observation_history.append((temp_before, action_taken, temp_after))

        # Need at least one observation to update
        if len(self.observation_history) < 2:
            return

        # Observed temperature change
        delta_temp_observed = temp_after - temp_before

        # Predicted temperature change using current parameters
        delta_temp_predicted = self._predict_delta_temp(action_taken, temp_before, ambient_temp)

        # Prediction error
        prediction_error = delta_temp_observed - delta_temp_predicted
        self.prediction_errors.append(prediction_error)

        # Build regressor vector φ
        # Model: ΔT = heating_rate × action - cooling_coefficient × (T - T_ambient)
        temp_diff = temp_before - ambient_temp
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

    def _predict_delta_temp(self, action: Action, current_temp: float, ambient_temp: float) -> float:
        """Helper to predict temperature change (not absolute temperature)"""
        temp_diff = current_temp - ambient_temp
        cooling_delta = -self.thermal_system.cooling_coefficient * temp_diff
        heating_delta = self.thermal_system.heating_rate_k_per_step if action == Action.ON else 0.0
        return heating_delta + cooling_delta

    def plot(self, save_path: str):
        # Implement plotting logic for visualizing predictions
        pass

    def _build_trajectory_with_details(
        self,
        actions: list[Action],
        initial_temp: float,
        ambient_temp: float,
        future_prices: list[float],
        watts_on: float
    ) -> list[TrajectoryStep]:
        """
        Build a detailed trajectory with temperature, cost, and price for each step.

        Args:
            actions: Sequence of actions to simulate
            initial_temp: Starting temperature (Kelvin)
            ambient_temp: Ambient temperature (Kelvin)
            future_prices: List of hourly prices (DKK/kWh)
            watts_on: Power consumption when ON (watts)

        Returns:
            List of TrajectoryStep objects with detailed predictions
        """
        trajectory = []
        current_temp = initial_temp

        for step_idx, action in enumerate(actions):
            # Get price for this step (hourly prices)
            hour_idx = step_idx // self.config.steps_per_hour
            if hour_idx < len(future_prices):
                price = future_prices[hour_idx]
            else:
                price = future_prices[-1] if future_prices else 0.0

            # Predict next temperature
            next_temp = self._predict_future_temperature(action, current_temp, ambient_temp)

            # Calculate cost for this step
            if action == Action.ON:
                energy_kwh = (watts_on / 1000.0) * (1.0 / self.config.steps_per_hour)
                cost = energy_kwh * price
            else:
                cost = 0.0

            # Store trajectory step
            trajectory.append(TrajectoryStep(
                action=action,
                predicted_temperature=next_temp,
                predicted_cost=cost,
                price=price
            ))

            # Update for next iteration
            current_temp = next_temp

        return trajectory
