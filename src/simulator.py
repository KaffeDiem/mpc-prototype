"""Simulator class for running thermal system simulations with adaptive MPC controller."""

from typing import Dict, List
import numpy as np
from controller_service import (
    ControllerService,
    ControllerServiceInitialMeasurements,
    ThermalSystemParams,
    ControllerServiceConfig,
    Action
)


class Simulator:
    """Simulates a thermal system controlled by an adaptive MPC controller.

    This class handles:
    - True physics simulation with configurable parameters
    - Controller service initialization and interaction
    - Measurement noise
    - Adaptive learning through model updates
    - Results tracking
    """

    def __init__(
        self,
        thermal_system: ThermalSystemParams,
        config: ControllerServiceConfig,
        watts_on: float = 3000.0,
        initial_temp: float = 323.15,  # 50°C in Kelvin
        true_heating_rate: float = 1.5,
        true_cooling_coeff: float = 0.02,
        measurement_noise_std: float = 0.1
    ):
        """Initialize the simulator.

        Args:
            thermal_system: Initial thermal system parameters for the controller
            config: Controller service configuration
            watts_on: Heater power in watts
            initial_temp: Initial temperature in Kelvin
            true_heating_rate: True heating rate for physics simulation
            true_cooling_coeff: True cooling coefficient for physics simulation
            measurement_noise_std: Standard deviation of measurement noise
        """
        self.thermal_system = thermal_system
        self.config = config
        self.watts_on = watts_on
        self.initial_temp = initial_temp
        self.true_heating_rate = true_heating_rate
        self.true_cooling_coeff = true_cooling_coeff
        self.measurement_noise_std = measurement_noise_std

        # Initialize controller service
        self.controller_service = ControllerService(
            ControllerServiceInitialMeasurements(thermal_system=thermal_system),
            config=config
        )

        # Current state
        self.T_current = initial_temp
        self.cum_cost = 0.0

        # Results storage
        self.results: Dict[str, List] = {
            "time": [],
            "temperature": [],
            "heater_on": [],
            "spot_price": [],
            "electricity_cost": [],
            "heating_rate": [],
            "cooling_coeff": [],
        }

    def step(self, t: int, spot_prices: np.ndarray) -> None:
        """Simulate one timestep.

        Args:
            t: Current timestep index
            spot_prices: Array of spot prices for all timesteps
        """
        # Get remaining prices for optimization horizon
        remaining_prices = spot_prices[t:].tolist()

        # Get next action from controller service
        pred_result = self.controller_service.get_next_action(
            current_temp=self.T_current,
            future_prices=remaining_prices,
            ambient_temp=self.thermal_system.ambient_temp_k,
            watts_on=self.watts_on
        )

        heater_on = pred_result.action == Action.ON

        # Simulate true physics (with potentially different parameters than controller's model)
        temp_diff = self.T_current - self.thermal_system.ambient_temp_k
        heating_delta = self.true_heating_rate if heater_on else 0.0
        cooling_delta = -self.true_cooling_coeff * temp_diff
        T_next = self.T_current + heating_delta + cooling_delta

        # Add measurement noise
        T_measured = T_next + np.random.normal(0, self.measurement_noise_std)

        # Update controller service's model (adaptive learning with RLS)
        self.controller_service.update_model(self.T_current, pred_result.action, T_measured)

        # Calculate costs
        timestep_hours = 1.0 / self.config.steps_per_hour
        energy_kwh = (self.watts_on / 1000) * timestep_hours if heater_on else 0.0
        cost = energy_kwh * spot_prices[t]

        self.cum_cost += cost

        # Store results
        self.results["time"].append(t * timestep_hours)
        self.results["temperature"].append(T_measured)
        self.results["heater_on"].append(heater_on)
        self.results["spot_price"].append(spot_prices[t])
        self.results["electricity_cost"].append(cost)
        self.results["heating_rate"].append(self.controller_service.theta[0])
        self.results["cooling_coeff"].append(self.controller_service.theta[1])

        # Update current temperature
        self.T_current = T_measured

    def run(
        self,
        spot_prices: np.ndarray,
        total_timesteps: int,
        print_progress: bool = True,
        progress_interval_hours: int = 2
    ) -> Dict[str, List]:
        """Run the simulation for the specified number of timesteps.

        Args:
            spot_prices: Array of spot prices for all timesteps
            total_timesteps: Total number of timesteps to simulate
            print_progress: Whether to print progress updates
            progress_interval_hours: How often to print progress (in hours)

        Returns:
            Dictionary containing simulation results
        """
        timesteps_per_hour = self.config.steps_per_hour

        for t in range(total_timesteps):
            self.step(t, spot_prices)

            # Progress update
            if print_progress and t % (timesteps_per_hour * progress_interval_hours) == 0:
                current_hour = (t // timesteps_per_hour) % 24
                T_celsius = self.results["temperature"][-1] - 273.15
                heater_on = self.results["heater_on"][-1]
                print(
                    f"Hour {current_hour:2d}: T={T_celsius:.1f}°C, "
                    f"Heater={'ON' if heater_on else 'OFF'}, "
                    f"Cost={self.cum_cost:.2f} DKK, "
                    f"Learned: h={self.controller_service.theta[0]:.3f}, c={self.controller_service.theta[1]:.4f}"
                )

        return self.results

    def print_summary(self) -> None:
        """Print a summary of simulation results."""
        temps_celsius = np.array(self.results["temperature"]) - 273.15
        temp_min_c = self.config.temp_min - 273.15
        temp_max_c = self.config.temp_max - 273.15

        print(f"\n{'='*60}")
        print(f"SIMULATION RESULTS - Adaptive Controller Service with RLS")
        print(f"{'='*60}")
        print(f"Total electricity cost: {self.cum_cost:.2f} DKK")
        print(f"\nTemperature Statistics:")
        print(f"  Range: {temps_celsius.min():.1f}°C - {temps_celsius.max():.1f}°C")
        print(f"  Mean: {temps_celsius.mean():.1f}°C")
        print(f"  Allowed range: {temp_min_c:.1f}°C - {temp_max_c:.1f}°C")
        print(f"  Range utilization: {temps_celsius.max() - temps_celsius.min():.1f}°C of {temp_max_c - temp_min_c:.1f}°C available")
        print(f"\nHeater Usage:")
        heater_on_pct = np.mean(self.results["heater_on"]) * 100
        print(f"  On: {heater_on_pct:.1f}% of time")
        print(f"  Total energy: {self.cum_cost / np.mean(self.results['spot_price']):.2f} kWh")
        print(f"\nAdaptive Learning:")
        print(f"  Initial heating rate: {self.thermal_system.heating_rate_k_per_step:.1f} K/step")
        print(f"  Final heating rate: {self.controller_service.theta[0]:.3f} K/step (true: {self.true_heating_rate:.1f})")
        print(f"  Initial cooling coeff: {self.thermal_system.cooling_coefficient:.3f}")
        print(f"  Final cooling coeff: {self.controller_service.theta[1]:.4f} (true: {self.true_cooling_coeff:.2f})")
        print(f"  Mean prediction error: {np.mean(np.abs(self.controller_service.prediction_errors)):.3f} K")
        print(f"{'='*60}\n")
