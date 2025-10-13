"""Simple example usage of the PricesService to fetch Danish electricity prices."""

from datetime import date, datetime, timedelta
import time
import logging
import csv
import signal
import sys
from prices_service import PricesService
from weather_service import WeatherService
from smart_plug_service import SmartPlugService
from thermometer_service import ThermometerService
from controller_service import *
from simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt

# Set up logging to see any warnings about missing dates
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Global variables for signal handler
csv_file = None
csv_filename = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\n\nShutting down gracefully...")
    if csv_file:
        csv_file.close()
        print(f"Stats saved to {csv_filename}")
    sys.exit(0)


def main():
    global csv_file, csv_filename
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    steps_per_hour = 12  # 5-minute intervals
    seconds_per_step = 3600 / steps_per_hour
    start_time = time.time()
    
    # Create CSV file with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"stats_{timestamp}.csv"
    csv_file = open(csv_filename, 'w', newline='', buffering=1)
    csv_writer = csv.writer(csv_file)
    
    # Write CSV headers
    csv_writer.writerow([
        'timestamp', 'step', 'current_temp_c', 'ambient_temp_c', 
        'action', 'watts_on', 'spot_price', 'heating_rate', 
        'cooling_coeff', 'predicted_temp_c', 'predicted_power'
    ])
    
    print(f"Logging stats to {csv_filename}")
    print("Press Ctrl+C to stop...\n")

    # Initialize service for DK2 region (Copenhagen/East of Great Belt)
    # Use "DK1" for Aarhus/West of Great Belt
    price_service = PricesService(region="DK2")
    weather_service = WeatherService()
    smart_plug_service = SmartPlugService()
    thermometer_service = ThermometerService()

    current_ambient_temp_k = celsius_to_kelvin(weather_service.get_current_temperature())
    thermal_system = ThermalSystemParams(
        heating_rate_k_per_step=1.5,
        cooling_coefficient=0.04,
        ambient_temp_k=current_ambient_temp_k,
    )
    initial_measurements = ControllerServiceInitialMeasurements(
        thermal_system=thermal_system
    )
    config = ControllerServiceConfig(
        temp_min=celsius_to_kelvin(20.0),
        temp_max=celsius_to_kelvin(22.0),
        steps_per_hour=steps_per_hour
    )
    controller = ControllerService(initial_measurements=initial_measurements, config=config)

    # Get today's and tomorrow's prices
    today = date.today()
    tomorrow = today + timedelta(days=1)

    prices = price_service.get_prices(today, tomorrow)
    prices = [p.price for p in prices]  # Extract just the price values

    # Perform initial measurements. Flip on for a few seconds to measure the initial watts when on.
    smart_plug_service.turn_on()
    time.sleep(3)
    watts_on = smart_plug_service.get_status().power_watts

    step_counter = 0

    while True:
        spot_prices = np.repeat(prices, steps_per_hour)
        current_temperature_k = celsius_to_kelvin(smart_plug_service.get_status().temperature_c)
        ambient_temp_c = weather_service.get_current_temperature()

        prediction = controller.get_next_action(
            current_temp=current_temperature_k,
            future_prices=spot_prices.tolist(),
            ambient_temp=celsius_to_kelvin(ambient_temp_c),
            watts_on=watts_on
        )

        # Determine current spot price based on step counter
        current_hour_index = step_counter // steps_per_hour
        current_spot_price = prices[current_hour_index % len(prices)]
        
        # Log stats to CSV
        csv_writer.writerow([
            datetime.now().isoformat(),
            step_counter,
            kelvin_to_celsius(current_temperature_k),
            ambient_temp_c,
            prediction.action.name,
            watts_on,
            current_spot_price,
            controller.theta[0],
            controller.theta[1],
            kelvin_to_celsius(prediction.predicted_temperature),
            prediction.predicted_power
        ])

        if prediction.action == Action.ON:
            smart_plug_service.turn_on()
            watts_on = smart_plug_service.get_status().power_watts
        else:
            smart_plug_service.turn_off()
        
        print("--------------------------------")
        print(f"Step {step_counter}:")
        print(f"Action taken: {prediction.action}")
        print(f"Current temperature: {current_temperature_k}")
        print(f"Ambient temperature: {ambient_temp_c}")

        time.sleep(seconds_per_step)  # Wait for next step
        
        # Update model with observed temperature change (for adaptive learning)
        prev_temperature_k = current_temperature_k
        current_temperature_k = celsius_to_kelvin(thermometer_service.get_current_temperature())
        controller.update_model(prev_temperature_k, prediction.action, current_temperature_k)
        
        step_counter += 1


def plot_results(results, config):
    """Create visualization of simulation results"""
    fig, axes = plt.subplots(5, 1, figsize=(12, 12))

    time = np.array(results["time"])

    # Temperature - Convert from Kelvin to Celsius for display
    temp_celsius = np.array(results["temperature"]) - 273.15
    min_celsius = config.temp_min - 273.15
    max_celsius = config.temp_max - 273.15

    axes[0].plot(time, temp_celsius, "b-", linewidth=2)
    axes[0].axhline(min_celsius, color="r", linestyle="--", label="Min")
    axes[0].axhline(max_celsius, color="r", linestyle="--", label="Max")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Adaptive Controller Service with RLS - 24h Simulation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Heater status
    heater_on = np.array(results["heater_on"])
    axes[1].fill_between(
        time, 0, heater_on, alpha=0.3, color="orange", label="Heater ON"
    )
    axes[1].set_ylabel("Heater Status")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # Spot prices
    axes[2].plot(time, results["spot_price"], "b-", linewidth=2, label="Spot Price")
    axes[2].set_ylabel("Spot Price (DKK/kWh)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Learned parameters over time
    axes[3].plot(time, results["heating_rate"], "r-", linewidth=2, label="Heating Rate")
    axes[3].axhline(1.5, color="r", linestyle="--", alpha=0.5, label="True value (1.5)")
    axes[3].set_ylabel("Heating Rate (K/step)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(
        time, results["cooling_coeff"], "b-", linewidth=2, label="Cooling Coefficient"
    )
    axes[4].axhline(
        0.02, color="b", linestyle="--", alpha=0.5, label="True value (0.02)"
    )
    axes[4].set_ylabel("Cooling Coefficient")
    axes[4].set_xlabel("Time (hours)")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("adaptive_controller_service_results.png")
    print("Plot saved to adaptive_controller_service_results.png")


if __name__ == "__main__":
    main()
