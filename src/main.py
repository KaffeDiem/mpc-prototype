"""Simple example usage of the PricesService to fetch Danish electricity prices."""

from datetime import date, timedelta
import logging
from prices_service import PricesService
from controller_service import ThermalSystemParams, ControllerServiceConfig
from simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt

# Set up logging to see any warnings about missing dates
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Initialize service for DK2 region (Copenhagen/East of Great Belt)
    # Use "DK1" for Aarhus/West of Great Belt
    service = PricesService(region="DK2")

    # Get today's and tomorrow's prices
    today = date.today()
    tomorrow = today + timedelta(days=1)

    prices = service.get_prices(today, tomorrow)
    prices = [p.price for p in prices] # Extract just the price values
    print(f"\nRetrieved {len(prices)} hourly prices:")
    print(f"Price range: {min(prices):.3f} - {max(prices):.3f} DKK/kWh")
    print(f"Mean price: {sum(prices)/len(prices):.3f} DKK/kWh\n")

    horizon_hours = 24 # Limit to 24 hours for performance
    timesteps_per_hour = 2  # 30-minute intervals (balance between resolution and speed)
    total_timesteps = horizon_hours * timesteps_per_hour

    spot_prices = np.repeat(prices, timesteps_per_hour)

    # Initialize controller with initial parameter guesstimates (intentionally slightly wrong)
    thermal_system = ThermalSystemParams.water_heater(
        heating_rate_k_per_step=1.5,  # Initial guess (true is 1.5)
        cooling_coefficient=0.015,     # Initial guess (true might be 0.02)
        ambient_temp_celsius=20.0
    )
    config = ControllerServiceConfig(
        temp_min=318.15,  # 45°C
        temp_max=333.15,  # 60°C - more realistic max for water heater
        steps_per_hour=timesteps_per_hour
    )

    # Create and run simulator
    simulator = Simulator(
        thermal_system=thermal_system,
        config=config,
        watts_on=3000.0,  # 3kW heater
        initial_temp=323.15,  # Start at 50°C = 323.15K
        true_heating_rate=1.5,  # True value
        true_cooling_coeff=0.02,  # True value
        measurement_noise_std=0.1
    )

    results = simulator.run(
        spot_prices=spot_prices,
        total_timesteps=total_timesteps,
        print_progress=True,
        progress_interval_hours=2
    )

    simulator.print_summary()
    print("Simulation complete. Plotting results...")

    plot_results(results, config)


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

    axes[4].plot(time, results["cooling_coeff"], "b-", linewidth=2, label="Cooling Coefficient")
    axes[4].axhline(0.02, color="b", linestyle="--", alpha=0.5, label="True value (0.02)")
    axes[4].set_ylabel("Cooling Coefficient")
    axes[4].set_xlabel("Time (hours)")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("adaptive_controller_service_results.png")
    print("Plot saved to adaptive_controller_service_results.png")


if __name__ == "__main__":
    main()