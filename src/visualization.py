import matplotlib.pyplot as plt
import numpy as np
from config import price_grid, cost, competitor_price, sku_types
from simulator import calculate_profit_advanced
from deep_demand_model import predict_demand, AdvancedDemandModel
from typing import Dict, List, Tuple

def plot_price_vs_profit_curve(
    model: AdvancedDemandModel,
    sku: str,
    current_cost: float,
    current_competitor_price: float,
    sku_type: str,
    current_seasonality: float,
    current_promo: int,
    current_day: int
):
    """
    Plots the profit curve for a single SKU across the defined price grid.
    Includes demand prediction and advanced profit calculation.

    Args:
        model (AdvancedDemandModel): The trained demand model for the SKU.
        sku (str): The SKU identifier.
        current_cost (float): Cost of the SKU.
        current_competitor_price (float): Competitor's price for the SKU.
        sku_type (str): Type of the SKU ("Vendor Product" or "Private Label").
        current_seasonality (float): Current seasonality factor.
        current_promo (int): Current promotional flag.
        current_day (int): Current day for time decay.
    """
    profits = []
    demands = []
    cpi_values = []
    for p in price_grid:
        d = predict_demand(
            model,
            price=float(p),
            competitor_price=current_competitor_price,
            seasonality=current_seasonality,
            promo=current_promo
        )
        prof, _, _, _ = calculate_profit_advanced(
            price=float(p),
            cost=current_cost,
            demand=d,
            competitor_price=current_competitor_price,
            day=current_day,
            sku_type=sku_type,
            is_promo_period=bool(current_promo)
        )
        profits.append(prof)
        demands.append(d)
        cpi_values.append((p / current_competitor_price) * 100 if current_competitor_price > 0 else 0)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Profit', color=color)
    ax1.plot(price_grid, profits, color=color, label='Profit')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(x=current_competitor_price * 1.00, color='gray', linestyle='--', label='CPI Min (100%)')
    ax1.axvline(x=current_competitor_price * 1.07, color='gray', linestyle=':', label='CPI Max (107%)')
    ax1.axvline(x=current_competitor_price, color='red', linestyle='-', label='Competitor Price')


    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel('Demand', color=color)
    ax2.plot(price_grid, demands, color=color, linestyle='--', label='Demand')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.title(f"Price vs Profit/Demand for {sku} (Type: {sku_type})")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.grid(True)
    plt.show()

def plot_pareto_front(F: np.ndarray, title: str = "PyMOO Pareto Front"):
    """
    Plots the Pareto front from PyMOO results.

    Args:
        F (np.ndarray): The objective values of the non-dominated solutions (res.F).
                        Expected shape (num_solutions, num_objectives).
        title (str): Title for the plot.
    """
    if F is None or F.shape[0] == 0:
        print("No Pareto front data to plot.")
        return

    plt.figure(figsize=(10, 7))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue', label='Pareto Front Solutions')
    plt.title(title)
    plt.xlabel('Total Negative Profit (Minimize)')
    plt.ylabel('Total CPI Deviation from Target (Minimize)')
    plt.grid(True)
    plt.legend()
    plt.show()