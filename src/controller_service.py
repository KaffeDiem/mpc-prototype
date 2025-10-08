import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt


@dataclass
class HeaterParams:
    """Physical parameters of the water heater"""

    volume_liters: float = 200.0  # Water volume in liters
    power_kw: float = 3.0  # Heating element power in kW
    heat_loss_coeff: float = 0.02  # Heat loss coefficient (kW/°C)
    ambient_temp: float = 20.0  # Ambient temperature (°C)
    temp_min: float = 45.0  # Minimum acceptable temperature (°C)
    temp_max: float = 75.0  # Maximum temperature (°C)
    temp_target: float = 60.0  # Target temperature (°C)

    # FCR-D parameters
    fcrd_min_mw: float = 0.001  # Minimum capacity (MW) - assuming part of VPP
    fcrd_response_time: float = 2.5  # seconds
    fcrd_delivery_min: int = 4  # Minimum 20 min = 4 timesteps of 5 min

    # Cost parameters
    temp_comfort_weight: float = 100.0  # Penalty for temperature deviation


class EconomicMPC:
    """
    Economic Model Predictive Controller for electric water heater optimization.
    Optimizes between electricity spot market costs and FCR-D capacity revenue.
    """

    def __init__(self, params: HeaterParams, horizon_steps: int = 24):
        """
        Initialize the MPC controller.

        Args:
            params: Physical parameters of the water heater
            horizon_steps: Prediction horizon in timesteps (5 min each)
        """
        self.params = params
        self.horizon = horizon_steps
        self.timestep_hours = 5 / 60  # 5 minutes in hours

        # Thermal capacity (kWh/°C)
        self.thermal_capacity = (params.volume_liters * 4.186) / 3600

        # State history for adaptive learning
        self.temp_history: list[float] = []
        self.power_history: list[float] = []
        self.actual_heat_loss_coeff = params.heat_loss_coeff

    def predict_temperature(
        self, T_current: float, heater_on: bool, timesteps: int = 1
    ) -> float:
        """
        Predict future temperature using simple thermal model.

        dT/dt = (P_heating - P_loss) / C
        where P_loss = k * (T - T_ambient)
        """
        T = T_current

        for _ in range(timesteps):
            # Heat input
            P_in = self.params.power_kw if heater_on else 0.0

            # Heat loss
            P_loss = self.actual_heat_loss_coeff * (T - self.params.ambient_temp)

            # Temperature change
            dT = (P_in - P_loss) * self.timestep_hours / self.thermal_capacity
            T += dT

        return T

    def objective_function(
        self,
        u: np.ndarray,
        T_current: float,
        spot_prices: np.ndarray,
        fcrd_prices: np.ndarray,
    ) -> float:
        """
        Objective function for optimization.

        Args:
            u: Control vector [heater_on_0, fcrd_down_0, heater_on_1, fcrd_down_1, ...]
               heater_on: 0 or 1 (binary)
               fcrd_down: MW capacity offered for down-regulation (continuous)
            T_current: Current water temperature
            spot_prices: Spot electricity prices (DKK/kWh) for horizon
            fcrd_prices: FCR-D down prices (DKK/MW/h) for horizon

        Returns:
            Total cost (negative revenue is profit)
        """
        n = self.horizon
        total_cost = 0.0
        T = T_current

        for i in range(n):
            # Extract controls
            heater_on = u[2 * i] > 0.5  # Binary decision
            fcrd_capacity_mw = u[2 * i + 1]  # Continuous

            # Predict temperature
            T_next = self.predict_temperature(T, heater_on, timesteps=1)

            # Electricity cost
            energy_kwh = (
                self.params.power_kw * self.timestep_hours if heater_on else 0.0
            )
            electricity_cost = energy_kwh * spot_prices[i]

            # FCR-D revenue (capacity payment)
            fcrd_revenue = fcrd_capacity_mw * fcrd_prices[i] * self.timestep_hours

            # Temperature comfort penalty (soft constraint)
            temp_deviation = 0.0
            if T_next < self.params.temp_min:
                temp_deviation = (self.params.temp_min - T_next) ** 2
            elif T_next > self.params.temp_max:
                temp_deviation = (T_next - self.params.temp_max) ** 2

            temp_penalty = self.params.temp_comfort_weight * temp_deviation

            # Total cost for this timestep
            total_cost += electricity_cost - fcrd_revenue + temp_penalty

            T = T_next

        return total_cost

    def constraints_fcrd(self, u: np.ndarray, T_current: float) -> np.ndarray:
        """
        Constraints for FCR-D participation.

        Returns array of constraint values (should be >= 0 for feasibility)
        Fixed size: 2 constraints per timestep
        """
        n = self.horizon
        constraints = np.zeros(2 * n)  # Fixed size: 2 per timestep
        T = T_current

        for i in range(n):
            heater_on_val = u[2 * i]  # Continuous value 0-1
            fcrd_capacity_mw = u[2 * i + 1]

            idx = 2 * i

            # Constraint 1: FCR-D capacity <= max_capacity * heater_on
            # This ensures FCR-D can only be offered when heater is on
            max_fcrd = self.params.power_kw / 1000  # Convert to MW
            constraints[idx] = max_fcrd * heater_on_val - fcrd_capacity_mw

            # Constraint 2: If offering FCR-D capacity > threshold and enough time left,
            # ensure we can deliver for minimum duration without violating temp_min
            if i + self.params.fcrd_delivery_min <= n and fcrd_capacity_mw > 0.0001:
                # Predict temperature if we turn off for delivery period
                T_future = self.predict_temperature(
                    T, heater_on=False, timesteps=self.params.fcrd_delivery_min
                )
                # Temperature must stay above minimum
                # Add small buffer and scale by capacity offered
                temp_margin = T_future - (self.params.temp_min - 1.0)
                constraints[idx + 1] = temp_margin
            else:
                # Not offering capacity or not enough time - constraint satisfied
                constraints[idx + 1] = 10.0  # Large positive number

            # Update temperature for next iteration
            heater_on_bool = heater_on_val > 0.5
            T = self.predict_temperature(T, heater_on_bool, timesteps=1)

        return constraints

    def optimize(
        self,
        T_current: float,
        spot_prices: np.ndarray,
        fcrd_prices: np.ndarray,
        current_hour: int,
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve the optimization problem.

        Args:
            T_current: Current water temperature (°C)
            spot_prices: Electricity spot prices for horizon (DKK/kWh)
            fcrd_prices: FCR-D capacity prices for horizon (DKK/MW/h)
            current_hour: Current hour (0-23) for checking price availability

        Returns:
            Optimal control sequence and optimization info
        """
        # Adjust horizon based on available data
        horizon = min(self.horizon, len(spot_prices), len(fcrd_prices))

        # Ensure we have at least a few timesteps
        if horizon < 1:
            # Fallback: return simple on/off decision
            heater_on = 1.0 if T_current < self.params.temp_target else 0.0
            return np.array([heater_on, 0.0]), {
                "success": False,
                "cost": 0,
                "iterations": 0,
                "message": "Insufficient horizon",
            }

        # Update horizon for this optimization
        original_horizon = self.horizon
        self.horizon = horizon

        # Initial guess: simple heuristic
        u0 = np.zeros(2 * horizon)
        for i in range(horizon):
            # Start with heater on if below target
            u0[2 * i] = 1.0 if T_current < self.params.temp_target else 0.0
            # Small FCR-D capacity as initial guess
            u0[2 * i + 1] = 0.0005 if u0[2 * i] > 0.5 else 0.0

        # Bounds
        bounds = []
        for i in range(horizon):
            bounds.append((0, 1))  # heater_on: binary
            bounds.append((0, self.params.power_kw / 1000))  # fcrd: 0 to max MW

        # Constraints
        def constraint_wrapper(u):
            return self.constraints_fcrd(u, T_current)

        constraints = {"type": "ineq", "fun": constraint_wrapper}

        # Optimize
        result = minimize(
            fun=lambda u: self.objective_function(
                u, T_current, spot_prices[:horizon], fcrd_prices[:horizon]
            ),
            x0=u0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-6},
        )

        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")

        info = {
            "success": result.success,
            "cost": result.fun,
            "iterations": result.nit,
            "message": result.message,
        }

        # Restore original horizon
        self.horizon = original_horizon

        return result.x, info

    def get_control_action(self, optimal_u: np.ndarray) -> Tuple[bool, float]:
        """
        Extract the first control action from optimal sequence.

        Returns:
            heater_on: Boolean, whether to turn heater on
            fcrd_capacity: Float, FCR-D capacity to offer in MW
        """
        heater_on = optimal_u[0] > 0.5
        fcrd_capacity = optimal_u[1]

        return heater_on, fcrd_capacity

    def update_model(self, T_measured: float, heater_was_on: bool):
        """
        Adaptive learning: Update model parameters based on measurements.
        """
        self.temp_history.append(T_measured)
        self.power_history.append(heater_was_on)

        # Only update if we have enough history
        if len(self.temp_history) < 5:
            return

        # Simple adaptive estimation of heat loss coefficient
        # Compare predicted vs actual temperature change
        T_prev = self.temp_history[-2]
        T_actual = self.temp_history[-1]
        heater_on = self.power_history[-1]

        T_predicted = self.predict_temperature(T_prev, heater_on, timesteps=1)

        error = T_actual - T_predicted

        # Adjust heat loss coefficient (simple gradient update)
        learning_rate = 0.01
        self.actual_heat_loss_coeff += learning_rate * error * 0.01

        # Keep within reasonable bounds
        self.actual_heat_loss_coeff = np.clip(self.actual_heat_loss_coeff, 0.01, 0.05)
