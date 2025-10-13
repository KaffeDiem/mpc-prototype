"""Simple example usage of the PricesService to fetch Danish electricity prices."""

from datetime import date, timedelta
import logging
from prices_service import PricesService
from controller_service import (
    ControllerService,
    ControllerServiceInitialMeasurements,
    ThermalSystemParams,
    ControllerServiceConfig,
    Action
)
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

    results = {
        "time": [],
        "temperature": [],
        "heater_on": [],
        "spot_price": [],
        "electricity_cost": [],
        "heating_rate": [],  # Track learned heating rate
        "cooling_coeff": [],  # Track learned cooling coefficient
    }

    cum_cost = 0.0

    # Initialize controller_service with initial parameter guesstimates (intentionally slightly wrong)
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
    controller_service = ControllerService(
        ControllerServiceInitialMeasurements(thermal_system=thermal_system),
        config=config
    )

    T_current = 323.15  # Start at 50°C = 323.15K (slightly below target)
    watts_on = 3000.0  # 3kW heater

    for t in range(total_timesteps):
        current_hour = (t // timesteps_per_hour) % 24

        # Get remaining prices for optimization horizon
        remaining_prices = spot_prices[t:].tolist()

        # Get next action from controller_service
        pred_result = controller_service.get_next_action(
            current_temp=T_current,
            future_prices=remaining_prices,
            ambient_temp=thermal_system.ambient_temp_k,
            watts_on=watts_on
        )

        heater_on = pred_result.action == Action.ON

        # Simulate true physics (with slightly different parameters than initial guess)
        # This simulates reality where true parameters are unknown
        true_heating_rate = 1.5  # True value (predictor starts with 0.3)
        true_cooling_coeff = 0.02  # True value (predictor starts with 0.015)
        temp_diff = T_current - thermal_system.ambient_temp_k
        heating_delta = true_heating_rate if heater_on else 0.0
        cooling_delta = -true_cooling_coeff * temp_diff
        T_next = T_current + heating_delta + cooling_delta

        # Add small measurement noise
        T_measured = T_next + np.random.normal(0, 0.1)

        # Update controller_service's model (adaptive learning with RLS)
        controller_service.update_model(T_current, pred_result.action, T_measured)

        # Calculate costs
        timestep_hours = 1.0 / timesteps_per_hour  # Dynamic based on configuration
        energy_kwh = (watts_on / 1000) * timestep_hours if heater_on else 0.0
        cost = energy_kwh * spot_prices[t]

        cum_cost += cost

        # Store results
        results["time"].append(t * timestep_hours)
        results["temperature"].append(T_measured)
        results["heater_on"].append(heater_on)
        results["spot_price"].append(spot_prices[t])
        results["electricity_cost"].append(cost)
        results["heating_rate"].append(controller_service.theta[0])
        results["cooling_coeff"].append(controller_service.theta[1])

        # Progress update
        if t % (timesteps_per_hour * 2) == 0:  # Every 2 hours
            print(
                f"Hour {current_hour:2d}: T={T_measured-273.15:.1f}°C, "
                f"Heater={'ON' if heater_on else 'OFF'}, "
                f"Cost={cum_cost:.2f} DKK, "
                f"Learned: h={controller_service.theta[0]:.3f}, c={controller_service.theta[1]:.4f}"
            )

        T_current = T_measured

    # Calculate statistics
    temps_celsius = np.array(results["temperature"]) - 273.15
    temp_min_c = config.temp_min - 273.15
    temp_max_c = config.temp_max - 273.15

    print(f"\n{'='*60}")
    print(f"SIMULATION RESULTS - Adaptive Controller Service with RLS")
    print(f"{'='*60}")
    print(f"Total electricity cost: {cum_cost:.2f} DKK")
    print(f"\nTemperature Statistics:")
    print(f"  Range: {temps_celsius.min():.1f}°C - {temps_celsius.max():.1f}°C")
    print(f"  Mean: {temps_celsius.mean():.1f}°C")
    print(f"  Allowed range: {temp_min_c:.1f}°C - {temp_max_c:.1f}°C")
    print(f"  Range utilization: {temps_celsius.max() - temps_celsius.min():.1f}°C of {temp_max_c - temp_min_c:.1f}°C available")
    print(f"\nHeater Usage:")
    heater_on_pct = np.mean(results["heater_on"]) * 100
    print(f"  On: {heater_on_pct:.1f}% of time")
    print(f"  Total energy: {cum_cost / np.mean(results['spot_price']):.2f} kWh")
    print(f"\nAdaptive Learning:")
    print(f"  Initial heating rate: 0.3 K/step")
    print(f"  Final heating rate: {controller_service.theta[0]:.3f} K/step (true: 0.5)")
    print(f"  Initial cooling coeff: 0.015")
    print(f"  Final cooling coeff: {controller_service.theta[1]:.4f} (true: 0.02)")
    print(f"  Mean prediction error: {np.mean(np.abs(controller_service.prediction_errors)):.3f} K")
    print(f"{'='*60}\n")
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