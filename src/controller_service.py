from dataclasses import dataclass
from enum import Enum
from scipy.optimize import minimize
import numpy as np


class Action(Enum):
    """
    Recommended next action to perform.
    """
    OFF = 0
    ON = 1


@dataclass
class ActionRecommendation:
    action: Action


@dataclass
class ControllerConfiguration:
    enable_fcr_d: bool


@dataclass 
class ControllerParameters:
    # Target temperature to maintain (e.g. for water heater)
    # A value between min_temp and max_temp that allows for good comfort
    # and flexibility for demand response.
    target_temp: float  # °C

    # Minimum allowed temperature (safety limit)
    # E.g. 45°C for water heaters
    min_temp: float  # °C

    # Maximum allowed temperature (safety limit)
    # E.g. 75°C for water heaters
    max_temp: float  # °C

    # Heat loss coefficient (UA), W/K
    # U = Overall heat transfer coefficient, W/m²K
    # A = Surface area, m²
    # Example: 
    # Cylinder with height 1.2m and diameter 0.5m
    # A = π * d * h + 2 * π * (d/2)² = 2.2 m²
    # If insulation is 0.5 W/m²K, then UA = 1.1 W/K
    # Converted to kW/K: 0.0011 kW/K
    # So a temperature difference of 10K would give a heat loss of 0.011 kW
    # E_loss = U * A * ΔT = 1.1 W/K * 10K = 11W = 0.011 kW
    ua: float

    # Specific heat capacity of the medium (e.g. water), kJ/kg·K
    # Water has a specific heat capacity of 4.18
    # If using another medium, adjust accordingly
    # Amount of kJ needed to raise 1 kg of the medium by 1 K
    cp: float  

    # Efficiency of the heater (0 to 1)
    # E.g. electric heater is close to 1, gas heater might be around 0.9
    # How much of the input energy is converted to useful heating
    eta: float

    # Amount of steps per hour to control the device
    # E.g. 4 means every 15 minutes
    steps_per_hour: int 

    # How many hours ahead to consider in the MPC optimization
    # E.g. 24 means looking one day ahead and calculating the optimal trajectory
    # for the next 24 hours
    hours_ahead: int  


class ControllerService:
    """
    Control a device (e.g. an electric water heater).
    Uses MPC to determine the best action.
    """

    def __init__(self, config: ControllerConfiguration, params: ControllerParameters):
        self.config = config
        self.params = params

        self.heat_loss_per_hour = 0.1  # °C per hour, placeholder value

    def get_next_action(self, current_temp: float, ambient_temp: float, future_prices: list[float]) -> Action:
        """
        Cost function to minimize.
        Balances energy cost and comfort (deviation from target temp).
        """

        total_steps = self.params.steps_per_hour * self.params.hours_ahead

        def cost_function(u_sequence: list[float]) -> float:
            print("Evaluating cost function with control sequence:", u_sequence)

            total_cost = 0

            for i in range(len(future_prices)):
                # Make sure we don't exceed the planned horizon.
                if i >= total_steps:
                    break

                u = u_sequence[i]  # Control action (0=OFF, 1=ON)

                # Cost in this time step (e.g. 15 minutes if steps_per_hour=4)
                dt = 1 / self.params.steps_per_hour  # hours
                price_forecast = future_prices[i]

                # Heater power when ON (kW)
                P_heater = 3.0 # in kW, TODO: Use actual heater power                

                # Current temperature

            

            return total_cost

        # Initial guess: all off indicated by 0 values
        initial_guess = [0.0] * (self.params.steps_per_hour * self.params.hours_ahead)

        # Optimize
        result = minimize(cost_function, initial_guess, method='SLSQP', 
                        bounds=[(0, 1)] * total_steps,
                        options={'maxiter': 100})
        
        print("Optimal control sequence:", result)

        optimal_u = result.x
        return Action.ON if optimal_u[0] > 0.5 else Action.OFF
