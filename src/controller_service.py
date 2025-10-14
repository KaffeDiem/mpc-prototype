from typing import Any
from dataclasses import dataclass
from enum import Enum
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np


# Large soft-constraint penalty scale (applied per-step, grows with violation distance)
PENALTY_SCALE = 1e6
# Small regularizer to encourage binary actions (0 or 1) during continuous optimization
# The term a*(1-a) is 0 at a∈{0,1} and maximal at a=0.5
BINARY_REG_SCALE = 1e-1

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
            """Continuous, differentiable objective for SciPy optimizers.

            - Actions are continuous in [0, 1] and represent duty cycle this step.
            - Dynamics use fractional heating proportional to action level.
            - Costs scale linearly with action level.
            - Temperature bounds enforced via large soft penalties.
            - Binary regularizer encourages actions near 0 or 1.
            """
            total_cost = 0.0

            # Energy/cost term (per-step, using hourly prices expanded internally)
            steps_per_hour = self.config.steps_per_hour
            for idx, a in enumerate(actions):
                hour = idx // steps_per_hour
                price = future_prices[hour] if hour < len(future_prices) else future_prices[-1] if future_prices else 0.0
                # energy_kwh for this step at action level a
                energy_kwh = (watts_on / 1000.0) * (1.0 / steps_per_hour) * float(a)
                total_cost += energy_kwh * price

            # Temperature penalty by simulating with fractional actions
            temp = current_temp
            for a in actions:
                temp = self._predict_future_temperature_continuous(float(a), temp, ambient_temp)
                below_violation = max(0.0, self.config.temp_min - temp)
                above_violation = max(0.0, temp - self.config.temp_max)
                if below_violation > 0.0:
                    total_cost += PENALTY_SCALE * (below_violation ** 3)
                if above_violation > 0.0:
                    total_cost += PENALTY_SCALE * (above_violation ** 3)

            # Encourage binary actions (0 or 1)
            # Sum a*(1-a) over the horizon; scale kept modest relative to monetary costs
            binary_reg = float(np.sum(actions * (1.0 - actions)))
            total_cost += BINARY_REG_SCALE * binary_reg

            return float(total_cost)

        # Limit optimization horizon (max 24 hours look-ahead) 
        optimization_horizon_hours = min(24, len(future_prices))
        optimization_steps = optimization_horizon_hours * self.config.steps_per_hour

        x0 = np.full(optimization_steps, 0.5)

        # Optimize with SciPy (bounded continuous variables)
        # Build nonlinear inequality constraints to keep temps within bounds at each step
        # For SLSQP: c(x) >= 0
        def make_step_violation_low(k: int):
            def fun(actions: np.ndarray) -> float:
                temp = current_temp
                for i in range(k + 1):
                    a = float(actions[i])
                    temp = self._predict_future_temperature_continuous(a, temp, ambient_temp)
                return temp - self.config.temp_min
            return fun

        def make_step_violation_high(k: int):
            def fun(actions: np.ndarray) -> float:
                temp = current_temp
                for i in range(k + 1):
                    a = float(actions[i])
                    temp = self._predict_future_temperature_continuous(a, temp, ambient_temp)
                return self.config.temp_max - temp
            return fun

        constraints: list[dict[str, Any]] = []
        for k in range(optimization_steps):
            constraints.append({"type": "ineq", "fun": make_step_violation_low(k)})
            constraints.append({"type": "ineq", "fun": make_step_violation_high(k)})

        result = minimize(
            fun=objective,
            x0=x0,
            bounds=[(0.0, 1.0)] * len(x0),
            method="SLSQP",  # differentiable, respects bounds
            constraints=constraints,  # type: ignore
            options={"maxiter": 300},  # type: ignore
        )

        optimized_actions = np.clip(result.x, 0.0, 1.0)

        # Discretize actions for execution/reporting with stepwise constraint enforcement
        actions: list[Action] = []
        sim_temp = current_temp
        for idx, a in enumerate(optimized_actions):
            proposed = Action.ON if a >= 0.5 else Action.OFF
            next_temp = self._predict_future_temperature(proposed, sim_temp, ambient_temp)
            if next_temp < self.config.temp_min:
                proposed = Action.ON
                next_temp = self._predict_future_temperature(proposed, sim_temp, ambient_temp)
            elif next_temp > self.config.temp_max:
                proposed = Action.OFF
                next_temp = self._predict_future_temperature(proposed, sim_temp, ambient_temp)
            actions.append(proposed)
            sim_temp = next_temp

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

    def _predict_future_temperature_continuous(
        self, action_level: float, current_temp: float, ambient_temp: float
    ) -> float:
        """Like _predict_future_temperature but with fractional heating.

        action_level ∈ [0, 1] scales the heating contribution linearly.
        """
        temp_diff = current_temp - ambient_temp
        cooling_delta = -self.thermal_system.cooling_coefficient * temp_diff
        heating_delta = self.thermal_system.heating_rate_k_per_step * max(0.0, min(1.0, action_level))
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

    def _price_for_sequence(
        self,
        future_prices: list[float],
        sequence: list[Action],
        watts_on: float,
    ) -> float:
        """Cost for a discrete ON/OFF sequence over a horizon with hourly prices."""

        total_cost = 0.0
        steps_per_hour = self.config.steps_per_hour
        for idx, action in enumerate(sequence):
            hour = idx // steps_per_hour
            if hour < len(future_prices):
                price = future_prices[hour]
            else:
                price = future_prices[-1] if future_prices else 0.0
            if action == Action.ON:
                total_cost += (watts_on / 1000.0) * price / steps_per_hour

        return total_cost
