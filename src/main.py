"""Simple example usage of the PricesService to fetch Danish electricity prices."""

from datetime import date, timedelta
import logging
from prices_service import PricesService
from controller_service import EconomicMPC, HeaterParams
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
    print(f"\nRetrieved {len(prices)} hourly prices:\n")

    horizon_hours = max(len(prices), 24) # Up to 24 hours horizon
    timesteps_per_hour = 12
    total_timesteps = horizon_hours * timesteps_per_hour
    fcrd_prices = np.array(prices) * 2 # Assuming FCR-D prices are double the spot prices TODO: Replace with real forecast
    fcrd_prices = np.repeat(fcrd_prices, timesteps_per_hour)

    spot_prices = np.repeat(prices, timesteps_per_hour)

    results = {
        "time": [],
        "temperature": [],
        "heater_on": [],
        "fcrd_capacity": [],
        "spot_price": [],
        "fcrd_price": [],
        "electricity_cost": [],
        "fcrd_revenue": [],
    }

    cum_cost = 0.0
    cum_revenue = 0.0

    params = HeaterParams(
        temp_comfort_weight=20.0,
    )
    controller = EconomicMPC(params=params, horizon_steps=24)
    T_current = 50.0 # Initial temperature

    for t in range(total_timesteps):
        current_hour = (t // timesteps_per_hour) % 24

        # Run MPC optimization every timestep (in practice, could be less frequent)
        optimal_u, info = controller.optimize(
            T_current, spot_prices[t:], fcrd_prices[t:], current_hour
        )

        # Get control action
        heater_on, fcrd_capacity = controller.get_control_action(optimal_u)

        # Apply control and simulate physics
        T_next = controller.predict_temperature(T_current, heater_on, timesteps=1)

        # Add small measurement noise
        T_measured = T_next + np.random.normal(0, 0.2)

        # Calculate costs
        energy_kwh = params.power_kw * controller.timestep_hours if heater_on else 0.0
        cost = energy_kwh * spot_prices[t]
        revenue = fcrd_capacity * fcrd_prices[t] * controller.timestep_hours

        cum_cost += cost
        cum_revenue += revenue

        # Store results
        results["time"].append(t * 5 / 60)  # Convert to hours
        results["temperature"].append(T_measured)
        results["heater_on"].append(heater_on)
        results["fcrd_capacity"].append(fcrd_capacity)
        results["spot_price"].append(spot_prices[t])
        results["fcrd_price"].append(fcrd_prices[t])
        results["electricity_cost"].append(cost)
        results["fcrd_revenue"].append(revenue)

        # Update model (adaptive learning)
        controller.update_model(T_measured, heater_on)

        # Progress update
        if t % (timesteps_per_hour * 2) == 0:  # Every 2 hours
            print(
                f"Hour {current_hour:2d}: T={T_measured:.1f}°C, "
                f"Heater={'ON' if heater_on else 'OFF'}, "
                f"FCR-D={fcrd_capacity*1000:.1f}kW, "
                f"Net Cost={cum_cost-cum_revenue:.2f} DKK"
            )

        T_current = T_measured

    print(f"\nTotal electricity cost: {cum_cost:.2f} DKK")
    print(f"Total FCR-D revenue: {cum_revenue:.2f} DKK")
    print(f"Net cost: {cum_cost - cum_revenue:.2f} DKK")
    print("\nSimulation complete. Plotting results...")

    plot_results(results, params)


if __name__ == "__main__":
    main()


def plot_results(results, params):
    """Create visualization of simulation results"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    time = np.array(results["time"])

    # Temperature
    axes[0].plot(time, results["temperature"], "b-", linewidth=2)
    axes[0].axhline(params.temp_target, color="g", linestyle="--", label="Target")
    axes[0].axhline(params.temp_min, color="r", linestyle="--", label="Min")
    axes[0].axhline(params.temp_max, color="r", linestyle="--", label="Max")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Water Heater Economic MPC - 24h Simulation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Heater status and FCR-D
    ax_heater = axes[1]
    ax_fcrd = ax_heater.twinx()

    heater_on = np.array(results["heater_on"])
    ax_heater.fill_between(
        time, 0, heater_on, alpha=0.3, color="orange", label="Heater ON"
    )
    ax_fcrd.plot(
        time,
        np.array(results["fcrd_capacity"]) * 1000,
        "g-",
        linewidth=2,
        label="FCR-D Capacity",
    )

    ax_heater.set_ylabel("Heater Status")
    ax_fcrd.set_ylabel("FCR-D Capacity (kW)", color="g")
    ax_heater.legend(loc="upper left")
    ax_fcrd.legend(loc="upper right")
    ax_heater.grid(True, alpha=0.3)

    # Prices
    ax_spot = axes[2]
    ax_fcrd_price = ax_spot.twinx()

    ax_spot.plot(time, results["spot_price"], "b-", linewidth=2, label="Spot Price")
    ax_fcrd_price.plot(
        time, results["fcrd_price"], "g-", linewidth=2, label="FCR-D Price"
    )

    ax_spot.set_ylabel("Spot Price (DKK/kWh)", color="b")
    ax_fcrd_price.set_ylabel("FCR-D Price (DKK/MW/h)", color="g")
    ax_spot.legend(loc="upper left")
    ax_fcrd_price.legend(loc="upper right")
    ax_spot.grid(True, alpha=0.3)

    # Cumulative costs and revenues
    cumulative_cost = np.cumsum(results["electricity_cost"])
    cumulative_revenue = np.cumsum(results["fcrd_revenue"])
    net_cost = cumulative_cost - cumulative_revenue

    axes[3].plot(time, cumulative_cost, "r-", linewidth=2, label="Electricity Cost")
    axes[3].plot(time, cumulative_revenue, "g-", linewidth=2, label="FCR-D Revenue")
    axes[3].plot(time, net_cost, "b-", linewidth=2, label="Net Cost")
    axes[3].set_ylabel("Cumulative (DKK)")
    axes[3].set_xlabel("Time (hours)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mpc_simulation_results.png")
    plt.show()