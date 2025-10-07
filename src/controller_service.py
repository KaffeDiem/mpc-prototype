from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """
    Recommended next action to perform.
    """
    OFF = 0
    ON = 1


@dataclass
class ActionRecommendation:
    action: Action
    fcr_d_up_capacity: float  # kW
    fcr_d_down_capacity: float  # kW


@dataclass
class ControllerConfiguration:
    enable_fcr_d: bool


@dataclass 
class ControllerParameters:
    target_temp: float  # °C
    min_temp: float  # °C
    max_temp: float  # °C


class ControllerService:
    """
    Control a device (e.g. an electric water heater).
    Uses MPC to determine the best action.
    """

    def __init__(self, config: ControllerConfiguration, params: ControllerParameters):
        self.config = config
        self.params = params

        self.heat_loss_per_hour = 0.1  # °C per hour, placeholder value

    def get_next_action(self) -> Action:
        return Action.OFF
    
    def calculate_affr_revenue(self) -> float:
        """
        Calculate expected revenue from participating in the FCR-D market.
        Based on capacity until next action change.
        """
        return 0.0
    
