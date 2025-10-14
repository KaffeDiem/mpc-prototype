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
        Determine the next action (ON/OFF) for the system using MPC optimization.
        The optimization handles temperature constraints internally.
        """
        # Emergency override: if currently violating constraints, force corrective action
        if current_temp < self.config.temp_min:
            # Force heater ON if below minimum temperature
            predicted_temp = self._predict_future_temperature(Action.ON, current_temp, ambient_temp)
            return ControllerServiceResult(
                action=Action.ON,
                predicted_temperature=predicted_temp,
                predicted_power=watts_on,
                trajectory=[Action.ON] * 96  # Placeholder trajectory
            )
        elif current_temp > self.config.temp_max:
            # Force heater OFF if above maximum temperature
            predicted_temp = self._predict_future_temperature(Action.OFF, current_temp, ambient_temp)
            return ControllerServiceResult(
                action=Action.OFF,
                predicted_temperature=predicted_temp,
                predicted_power=0.0,
                trajectory=[Action.OFF] * 96  # Placeholder trajectory
            )
        
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
        Uses hard constraints to strictly enforce temperature bounds.
        """

        def objective(actions: np.ndarray) -> float:
            # Convert to action sequence
            action_sequence = [Action.ON if a >= 0.5 else Action.OFF for a in actions]

            # Calculate electricity cost (no penalty terms needed with hard constraints)
            sequence_price = self._price_for_sequence(
                future_prices=future_prices,
                sequence=action_sequence,
                watts_on=watts_on,
            )

            # Calculate the future temperature trajectory. If they are outside the bounds, return a large penalty
            # The penalty should be extremely large and increasing as the temperature moves further outside the bounds.
            # This makes sure that the optimizer will never choose an action that will move the temperature further outside the bounds.
            temp_trajectory = self._simulate_temperature_trajectory(
                action_sequence, current_temp, ambient_temp
            )
            
            # Calculate penalty for constraint violations
            penalty = 0.0
            for temp in temp_trajectory:
                if temp < self.config.temp_min:
                    # Exponential penalty for being below minimum
                    violation = self.config.temp_min - temp
                    penalty += 1e6 * (violation ** 2)  # Quadratic scaling for smooth gradient
                elif temp > self.config.temp_max:
                    # Exponential penalty for being above maximum
                    violation = temp - self.config.temp_max
                    penalty += 1e6 * (violation ** 2)  # Quadratic scaling for smooth gradient

            return sequence_price + penalty

        # Limit optimization horizon (max 24 hours look-ahead) 
        optimization_horizon_hours = min(24, len(future_prices))
        optimization_steps = optimization_horizon_hours * self.config.steps_per_hour

        # Create temperature-aware initial guess
        x0 = self._create_initial_guess(
            current_temp=current_temp,
            ambient_temp=ambient_temp,
            future_prices=future_prices,
            optimization_horizon_hours=optimization_horizon_hours
        )

        # Ensure x0 has correct length
        assert(len(x0) == optimization_steps)

        # Optimize action sequence with hard constraints
        # Note: pyright has strict typing for scipy.optimize.minimize constraints parameter
        result = minimize(  # type: ignore[call-overload]
            fun=objective,
            x0=x0,
            bounds=[(0, 1)] * len(x0),
            method='SLSQP',  # Required for constraints
            options={'disp': False, 'maxiter': 200}  # Increased iterations for constrained optimization
        )

        actions = [Action.ON if a >= 0.5 else Action.OFF for a in result.x]

        return ControllerServiceResult(
            actions[0],
            self._predict_future_temperature(actions[0], current_temp, ambient_temp),
            watts_on if actions[0] == Action.ON else 0.0,
            actions,
        )
    
    def _create_initial_guess(
        self,
        current_temp: float,
        ambient_temp: float,
        future_prices: list[float],
        optimization_horizon_hours: int
    ) -> list[float]:
        """
        Create a temperature-aware initial guess for the optimizer.
        If temperature is out of bounds, prioritize corrective action.
        Otherwise, use price-based heuristic.
        """
        x0 = []
        
        # Calculate how many steps needed to recover if out of bounds
        temp_margin_min = current_temp - self.config.temp_min
        temp_margin_max = self.config.temp_max - current_temp
        
        # If below minimum, initialize with ON actions until recovery expected
        if temp_margin_min < 0:
            # Estimate steps needed to recover
            net_heating_rate = self.thermal_system.heating_rate_k_per_step - \
                              self.thermal_system.cooling_coefficient * (current_temp - ambient_temp)
            steps_to_recover = max(1, int(np.ceil(abs(temp_margin_min) / max(net_heating_rate, 0.1))))
            
            for hour_idx in range(optimization_horizon_hours):
                for step_in_hour in range(self.config.steps_per_hour):
                    total_step = hour_idx * self.config.steps_per_hour + step_in_hour
                    if total_step < steps_to_recover:
                        x0.append(1.0)  # ON to recover
                    else:
                        # After recovery, use price-based logic
                        if hour_idx < len(future_prices):
                            avg_price = sum(future_prices[:optimization_horizon_hours]) / optimization_horizon_hours
                            x0.append(1.0 if future_prices[hour_idx] < avg_price else 0.0)
                        else:
                            x0.append(0.0)
        
        # If above maximum, initialize with OFF actions until recovery expected
        elif temp_margin_max < 0:
            # Estimate steps needed to cool down
            net_cooling_rate = abs(self.thermal_system.cooling_coefficient * (current_temp - ambient_temp))
            steps_to_recover = max(1, int(np.ceil(abs(temp_margin_max) / max(net_cooling_rate, 0.1))))
            
            for hour_idx in range(optimization_horizon_hours):
                for step_in_hour in range(self.config.steps_per_hour):
                    total_step = hour_idx * self.config.steps_per_hour + step_in_hour
                    if total_step < steps_to_recover:
                        x0.append(0.0)  # OFF to cool down
                    else:
                        # After recovery, use price-based logic
                        if hour_idx < len(future_prices):
                            avg_price = sum(future_prices[:optimization_horizon_hours]) / optimization_horizon_hours
                            x0.append(1.0 if future_prices[hour_idx] < avg_price else 0.0)
                        else:
                            x0.append(0.0)
        
        # If within bounds, use price-based initialization
        else:
            avg_price = sum(future_prices[:optimization_horizon_hours]) / optimization_horizon_hours if optimization_horizon_hours > 0 else 1.0
            for hour_idx in range(optimization_horizon_hours):
                for _ in range(self.config.steps_per_hour):
                    if hour_idx < len(future_prices):
                        # Turn on more likely if price is below average
                        initial_action = 1.0 if future_prices[hour_idx] < avg_price else 0.0
                        x0.append(initial_action)
        
        return x0

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
