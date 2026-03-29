from data_generation import generate_data
from deep_demand_model import train_models
from optimization import run_ortools_optimization
from pymoo_optimization import DFLPricingProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from config import cost, competitor_price, sku_types, skus
from visualization import plot_price_vs_profit_curve, plot_pareto_front
import random
import numpy as np
from typing import Dict 

def run_pymoo_optimization(models: Dict[str, 'AdvancedDemandModel']):
    """
    Runs Multi-Objective Optimization using PyMOO (NSGA2 algorithm).

    Args:
        models (Dict[str, AdvancedDemandModel]): Dictionary of trained demand models.
    """
    print("\n--- PyMOO Multi-Objective Optimization ---")

    # Simulate current environment for optimization (these would be real-time inputs)
    current_promo_flag = random.choice([0, 1])
    current_day = random.randint(5, 25)
    current_seasonality = random.uniform(-30, 50)

    problem = DFLPricingProblem(
        models=models,
        current_seasonality=current_seasonality,
        current_promo=current_promo_flag,
        current_day=current_day
    )

    # NSGA2 is a popular multi-objective evolutionary algorithm
    algorithm = NSGA2(
        pop_size=100, # Population size
        # sampling=LHS(), # Latin Hypercube Sampling for initial population
        # crossover=SBX(prob=0.9, eta=15),
        # mutation=PM(eta=20)
    )

    # Termination criteria (e.g., number of generations)
    termination = get_termination("n_gen", 100) # Run for 100 generations

    # Run the optimization
    print(f"Running PyMOO with {problem.n_obj} objectives and {problem.n_constr} constraints...")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1, # For reproducibility
        verbose=False,
        save_history=True
    )

    print("\n=== PyMOO Optimization Results ===")
    if res.X is not None and res.F is not None:
        print(f"Found {len(res.X)} non-dominated solutions (Pareto Front size).")
        print("Example Pareto Solutions (Price Vector, Negative Profit, CPI Deviation):")
        for i in range(min(5, len(res.X))): # Print first 5 solutions
            prices = [int(p) for p in res.X[i]]
            profit = -res.F[i, 0] # Convert back to positive profit
            cpi_dev = res.F[i, 1]
            print(f"Solution {i+1}: Prices={prices[:5]}... (for {len(prices)} SKUs), Profit={profit:.2f}, CPI_Dev={cpi_dev:.2f}")

        # Plot the Pareto front
        plot_pareto_front(res.F)

        # Optional: Analyze a specific solution from the Pareto front
        # E.g., the one with highest profit (lowest negative profit)
        best_profit_idx = np.argmin(res.F[:, 0])
        best_profit_prices = res.X[best_profit_idx]
        best_profit_val = -res.F[best_profit_idx, 0]
        best_profit_cpi_dev = res.F[best_profit_idx, 1]
        print(f"\nSolution with Highest Profit: Prices={best_profit_prices[:5]}... Profit={best_profit_val:.2f}, CPI_Dev={best_profit_cpi_dev:.2f}")

        # E.g., the one with lowest CPI deviation
        best_cpi_idx = np.argmin(res.F[:, 1])
        best_cpi_prices = res.X[best_cpi_idx]
        best_cpi_val = -res.F[best_cpi_idx, 0]
        best_cpi_cpi_dev = res.F[best_cpi_idx, 1]
        print(f"Solution with Lowest CPI Deviation: Prices={best_cpi_prices[:5]}... Profit={best_cpi_val:.2f}, CPI_Dev={best_cpi_cpi_dev:.2f}")


    else:
        print("PyMOO did not find any feasible solutions.")


def main():
    print("--- Starting DFL Retail Pricing Optimization ---")

    print("\n1. Generating synthetic demand data...")
    df = generate_data()
    print(f"Data generated for {len(df['sku'].unique())} SKUs over {df['day'].nunique()} days.")

    print("\n2. Training Advanced Deep Learning Demand Models (one per SKU)...")
    models = train_models(df)
    if not models:
        print("No demand models trained. Exiting.")
        return
    print(f"Trained models for {len(models)} SKUs.")

    # Simulate a current scenario for optimization runs
    current_day_for_opt = random.randint(5, 25)
    current_seasonality_for_opt = random.uniform(-30, 50)
    current_promo_for_opt = random.choice([0, 1])

    # 3. Run OR-Tools Optimization (Single-Objective: Maximize Profit with Hard Constraints)
    run_ortools_optimization(models)

    # 4. Run PyMOO Multi-Objective Optimization (Profit vs. CPI Deviation with Constraints)
    run_pymoo_optimization(models)

    # 5. Plotting Price vs Profit curve for a few example SKUs
    print("\n5. Plotting Price vs Profit curve for example SKUs...")
    # Select a few SKUs to visualize
    example_skus = list(skus)[:3] # Take first 3 SKUs
    for sku in example_skus:
        if sku in models:
            plot_price_vs_profit_curve(
                model=models[sku],
                sku=sku,
                current_cost=cost[sku],
                current_competitor_price=competitor_price[sku],
                sku_type=sku_types[sku],
                current_seasonality=current_seasonality_for_opt,
                current_promo=current_promo_for_opt,
                current_day=current_day_for_opt
            )
        else:
            print(f"Warning: Model not found for SKU {sku}, skipping plot.")

    print("\n--- DFL Retail Pricing Optimization Completed ---")

if __name__ == "__main__":
    main()