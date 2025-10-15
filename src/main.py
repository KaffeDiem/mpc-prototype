"""Simple example usage of the PricesService to fetch Danish electricity prices."""

from datetime import date, datetime, timedelta
import time
import logging
import csv
import signal
import sys
from typing import Tuple, Optional, TextIO, Any
import numpy as np

from prices_service import PricesService, Price, Region
from weather_service import WeatherService
from smart_plug_service import SmartPlugService
from thermometer_service import ThermometerService
from controller_service import (
    Action,
    celsius_to_kelvin,
    kelvin_to_celsius,
    ThermalSystemParams,
    ControllerServiceConfig,
    ControllerService,
)
from fcr_service import FCRService

# Set up logging to see any warnings about missing dates
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# ============================================================================
# Service Initialization Functions
# ============================================================================

def initialize_services(region: Region) -> Tuple[PricesService, WeatherService, SmartPlugService, ThermometerService, FCRService]:
    """Initialize all required services."""
    price_service = PricesService(region=region)
    weather_service = WeatherService()
    smart_plug_service = SmartPlugService()
    thermometer_service = ThermometerService()
    fcr_service = FCRService()
    return price_service, weather_service, smart_plug_service, thermometer_service, fcr_service


def initialize_controller(
    weather_service: WeatherService,
    steps_per_hour: int,
    temp_min_celsius: float,
    temp_max_celsius: float
) -> ControllerService:
    """Initialize the controller with thermal system parameters and config."""
    current_ambient_temp_k = celsius_to_kelvin(weather_service.get_current_temperature())
    thermal_system = ThermalSystemParams(
        heating_rate_k_per_step=1.5,
        cooling_coefficient=0.04,
        ambient_temp_k=current_ambient_temp_k,
    )
    config = ControllerServiceConfig(
        temp_min=celsius_to_kelvin(temp_min_celsius),
        temp_max=celsius_to_kelvin(temp_max_celsius),
        steps_per_hour=steps_per_hour
    )
    return ControllerService(initial_measurements=thermal_system, config=config)


def perform_initial_measurements(smart_plug_service: SmartPlugService) -> float:
    """
    Perform initial measurements to determine watts when on.
    Returns the measured watts value or a default fallback.
    """
    logging.info("Performing initial measurements...")
    turn_on_success = smart_plug_service.turn_on()
    if not turn_on_success:
        logging.warning("Failed to turn on plug during initialization. Using default watts value.")
        return 500.0  # Default fallback value
    
    time.sleep(1)
    watts_on = smart_plug_service.get_status().power_watts
    if watts_on == 0.0:
        logging.warning("Got 0W reading during initialization. Using default watts value.")
        return 500.0  # Default fallback value
    
    logging.info(f"Initial measurement: {watts_on}W when ON")
    return watts_on


# ============================================================================
# Price Management Functions
# ============================================================================

def should_refetch_prices(
    last_fetch_time: float,
    prices: list[Price],
    min_future_hours: int
) -> bool:
    """Determine if prices should be refetched."""
    current_time = time.time()
    now = datetime.now()
    
    # Filter to get only future prices
    future_prices = [p for p in prices if p.date >= now.replace(minute=0, second=0, microsecond=0)]
    
    # Refetch if it's been an hour, or if we have fewer than min_future_hours of future prices
    return (current_time - last_fetch_time >= 3600) or (len(future_prices) < min_future_hours)


def fetch_prices(
    price_service: PricesService,
    today: date,
    tomorrow: date
) -> list[Price]:
    """
    Fetch prices with error handling.
    Returns a list of Price objects, or default prices on failure.
    """
    try:
        prices = price_service.get_prices(today, tomorrow)
        print(f"Fetched {len(prices)} hourly prices")
        return prices
    except Exception as e:
        logging.error(f"Failed to fetch prices: {e}")
        # Create default Price objects for the next 48 hours
        now = datetime.now()
        default_prices = [Price(date=now + timedelta(hours=i), price=1.0) for i in range(48)]
        print("Using default prices due to fetch failure")
        return default_prices


def get_current_spot_price(prices: list[Price]) -> float:
    """
    Get the current spot price by filtering out all prices before the current hour.
    Returns the first remaining price, or 1.0 as a default.
    """
    now = datetime.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    
    # Filter out prices before current hour
    remaining_prices = [p for p in prices if p.date >= current_hour]
    
    if remaining_prices:
        return remaining_prices[0].price
    else:
        logging.warning("No prices available for current or future hours, using default")
        return 1.0


def prepare_future_prices(prices: list[Price]) -> list[float]:
    """
    Filter future prices and return hourly values.
    Controller will expand these per-step internally.
    """
    now = datetime.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    
    # Filter to get only future prices
    future_prices = [p for p in prices if p.date >= current_hour]
    
    # Extract just the hourly price values (no repetition)
    future_price_values = [p.price for p in future_prices]
    
    return future_price_values


# ============================================================================
# CSV Logging Functions
# ============================================================================

def setup_csv_file(timestamp: str) -> Tuple[TextIO, Any, str]:
    """
    Setup CSV file for logging statistics.
    Returns tuple of (file_handle, csv_writer, filename).
    """
    csv_filename = f"stats_{timestamp}.csv"
    csv_file = open(csv_filename, 'w', newline='', buffering=1)
    csv_writer = csv.writer(csv_file)
    
    # Write CSV headers
    csv_writer.writerow([
        'timestamp', 'step', 'current_temp_c', 'ambient_temp_c', 
        'action', 'watts_on', 'spot_price', 'heating_rate', 
        'cooling_coeff', 'predicted_temp_c', 'predicted_power',
        'cost_eur_per_step', 'cumulative_cost_eur'
    ])
    
    return csv_file, csv_writer, csv_filename


def log_step_to_csv(
    csv_writer: Any,
    step_counter: int,
    current_temperature_k: float,
    ambient_temp_c: float,
    action: Action,
    watts_on: float,
    current_spot_price: float,
    heating_rate: float,
    cooling_coeff: float,
    predicted_temperature_k: float,
    predicted_power: float,
    cost_per_step: float,
    cumulative_cost_eur: float,
    fcr_d_down_price: float,
    fcr_d_up_price: float
) -> None:
    """Log a single step's data to CSV."""
    csv_writer.writerow([
        datetime.now().isoformat(),
        step_counter,
        kelvin_to_celsius(current_temperature_k),
        ambient_temp_c,
        action.name,
        watts_on,
        current_spot_price,
        heating_rate,
        cooling_coeff,
        kelvin_to_celsius(predicted_temperature_k),
        predicted_power,
        cost_per_step,
        cumulative_cost_eur,
        fcr_d_down_price,
        fcr_d_up_price
    ])


# ============================================================================
# Control Logic Functions
# ============================================================================

def calculate_step_cost(
    action: Action,
    watts_on: float,
    spot_price: float,
    seconds_per_step: float
) -> float:
    """
    Calculate the cost in EUR for a single step.
    Energy consumed = (watts / 1000) * (seconds_per_step / 3600) in kWh
    """
    if action == Action.ON:
        energy_kwh = (watts_on / 1000.0) * (seconds_per_step / 3600.0)
        return energy_kwh * spot_price
    return 0.0


def execute_action_and_update_watts(
    action: Action,
    smart_plug_service: SmartPlugService
) -> float:
    """
    Execute the action on the smart plug and return the updated watts reading.
    """
    if action == Action.ON:
        smart_plug_service.turn_on()
        return smart_plug_service.get_status().power_watts
    else:
        smart_plug_service.turn_off()
        return 0.0


def print_trajectory_details(trajectory: list) -> None:
    """Print formatted trajectory showing action, temperature, price, and cost for each step."""
    print("\nPredicted Trajectory:")
    print("Step | Action | Predicted Temp (°C) | Price (DKK/kWh) | Cost (DKK)")
    print("-" * 75)
    
    for step_idx, step in enumerate(trajectory):
        temp_celsius = kelvin_to_celsius(step.predicted_temperature)
        print(f"{step_idx:4d} | {step.action.name:6s} | {temp_celsius:18.2f} | {step.price:15.2f} | {step.predicted_cost:11.4f}")


def print_step_info(
    step_counter: int,
    action: Action,
    current_temp_celsius: float,
    ambient_temp_celsius: float,
    cost_per_step: float,
    cumulative_cost_eur: float,
    fcr_d_down_price: float,
    fcr_d_up_price: float
) -> None:
    """Print formatted information about the current step."""
    print("--------------------------------")
    print(f"Step {step_counter}:")
    print(f"Next action: {action}")
    print(f"Current temperature: {current_temp_celsius}")
    print(f"Ambient temperature: {ambient_temp_celsius}")
    print(f"Cost this step: {cost_per_step:.4f} EUR")
    print(f"Cumulative cost: {cumulative_cost_eur:.2f} EUR")
    print(f"FCR-D down price: {fcr_d_down_price:.4f} EUR")
    print(f"FCR-D up price: {fcr_d_up_price:.4f} EUR")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main control loop for temperature management system."""
    # Configuration
    steps_per_hour = 4  # E.g. 4 == 15-minute intervals
    seconds_per_step = 3600 / steps_per_hour
    MIN_FUTURE_HOURS = 12  # Refetch if we have less than this many hours of future prices
    
    # Setup CSV logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_file, csv_writer, csv_filename = setup_csv_file(timestamp)
    
    # Setup signal handler for graceful shutdown using closure
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\n\nShutting down gracefully...")
        csv_file.close()
        print(f"Stats saved to {csv_filename}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Logging stats to {csv_filename}")
    print("Press Ctrl+C to stop...\n")

    # Initialize services for DK2 region (Copenhagen/East of Great Belt)
    # Use "DK1" for Aarhus/West of Great Belt
    price_service, weather_service, smart_plug_service, thermometer_service, fcr_service = initialize_services("DK2")
    
    # Initialize controller
    controller = initialize_controller(
        weather_service=weather_service,
        steps_per_hour=steps_per_hour,
        temp_min_celsius=20.0,
        temp_max_celsius=22.0
    )
    
    # Perform initial measurements
    expected_watts_on = perform_initial_measurements(smart_plug_service)

    # Initialize control loop state
    step_counter = 0
    cumulative_cost_eur = 0.0
    prices = []
    last_price_fetch_time = 0  # Force initial fetch
    fcr_d_down_price, fcr_d_up_price = fcr_service.get_fcr_prices()

    # Main control loop
    while True:
        # ===== FETCH PRICES =====
        # Fetch prices if needed
        if should_refetch_prices(last_price_fetch_time, prices, MIN_FUTURE_HOURS):
            print("Fetching updated electricity prices...")
            today = date.today()
            tomorrow = today + timedelta(days=1)
            prices = fetch_prices(price_service, today, tomorrow)
            last_price_fetch_time = time.time()
            fcr_d_down_price, fcr_d_up_price = fcr_service.get_fcr_prices()
        
        if not prices:  # Safety check
            logging.warning("No prices available, skipping this iteration")
            time.sleep(seconds_per_step)
            continue

        # ===== MEASURE CURRENT STATE =====
        current_temperature_k = celsius_to_kelvin(thermometer_service.get_current_temperature())
        ambient_temp_c = weather_service.get_current_temperature()
        ambient_temp_k = celsius_to_kelvin(ambient_temp_c)
        future_price_list = prepare_future_prices(prices)
        current_spot_price = get_current_spot_price(prices)

        # ===== PREDICT NEXT ACTION =====
        # Use expected_watts_on for prediction to ensure predicted_power matches the action
        prediction = controller.get_next_action(
            current_temp=current_temperature_k,
            future_prices=future_price_list,
            ambient_temp=ambient_temp_k,
            watts_on=expected_watts_on,
            fcr_d_down_price=fcr_d_down_price,
            fcr_d_up_price=fcr_d_up_price
        )

        # ===== EXECUTE ACTION =====
        actual_watts = execute_action_and_update_watts(prediction.trajectory[0].action, smart_plug_service)
        
        # Update expected watts if actual measurement differs significantly (>10%)
        if abs(actual_watts - expected_watts_on) / max(expected_watts_on, 1.0) > 0.10:
            logging.info(f"Updating expected watts: {expected_watts_on:.1f}W -> {actual_watts:.1f}W")
            expected_watts_on = actual_watts
        
        # ===== CALCULATE COSTS =====
        cost_per_step = calculate_step_cost(
            prediction.trajectory[0].action,
            expected_watts_on,  # Use expected watts for cost calculation
            current_spot_price,
            seconds_per_step
        )
        cumulative_cost_eur += cost_per_step
        
        # ===== LOG DATA =====
        log_step_to_csv(
            csv_writer=csv_writer,
            step_counter=step_counter,
            current_temperature_k=current_temperature_k,
            ambient_temp_c=ambient_temp_c,
            action=prediction.trajectory[0].action,
            watts_on=actual_watts,  # Log actual measured watts
            current_spot_price=current_spot_price,
            heating_rate=controller.theta[0],
            cooling_coeff=controller.theta[1],
            predicted_temperature_k=prediction.trajectory[0].predicted_temperature,
            predicted_power=prediction.predicted_power,
            cost_per_step=cost_per_step,
            cumulative_cost_eur=cumulative_cost_eur,
            fcr_d_down_price=fcr_d_down_price,
            fcr_d_up_price=fcr_d_up_price
        )

        print_step_info(
            step_counter=step_counter,
            action=prediction.trajectory[0].action,
            current_temp_celsius=kelvin_to_celsius(current_temperature_k),
            ambient_temp_celsius=ambient_temp_c,
            cost_per_step=cost_per_step,
            cumulative_cost_eur=cumulative_cost_eur,
            fcr_d_down_price=fcr_d_down_price,
            fcr_d_up_price=fcr_d_up_price
        )
        
        # ===== PRINT TRAJECTORY DETAILS =====
        print_trajectory_details(prediction.trajectory)

        # ===== WAIT FOR NEXT STEP =====
        time.sleep(seconds_per_step)
        
        # ===== UPDATE ADAPTIVE MODEL =====
        prev_temperature_k = current_temperature_k
        next_temperature_k = celsius_to_kelvin(thermometer_service.get_current_temperature())
        next_ambient_temp_k = celsius_to_kelvin(weather_service.get_current_temperature())
        controller.update_model(prev_temperature_k, prediction.trajectory[0].action, next_temperature_k, next_ambient_temp_k)
        
        step_counter += 1


if __name__ == "__main__":
    main()
