from ortools.sat.python import cp_model
from config import skus, cost, competitor_price, inventory, price_grid, sku_types, \
                   CPI_MIN_PERCENT, CPI_MAX_PERCENT
from simulator import calculate_profit_advanced
from deep_demand_model import predict_demand
import random
from typing import Dict, List, Tuple

def run_ortools_optimization(models: Dict[str, 'AdvancedDemandModel']):
    """
    Runs an OR-Tools CP-SAT optimization to maximize total profit for all SKUs,
    subject to inventory and CPI constraints.

    Args:
        models (Dict[str, AdvancedDemandModel]): Dictionary of trained demand models.
    """
    print("\n--- OR-Tools CP-SAT Optimization ---")

    # Simulate current environment (these would come from real-time data)
    current_promo_flag = random.choice([0, 1])
    current_day = random.randint(5, 25)
    current_seasonality = random.uniform(-30, 50) # Varies by SKU in data_gen, here a global avg for simplicity

    # Precompute (Simulator) - This step is crucial for DFL to feed the optimizer
    # We pre-calculate demand and profit for each possible price point for each SKU
    # using the learned demand models and the advanced simulator.
    profit_map: Dict[Tuple[str, int], float] = {}
    demand_map: Dict[Tuple[str, int], int] = {}

    for sku in skus:
        for p in price_grid:
            # Predict demand using the trained model
            d = predict_demand(
                models[sku],
                price=float(p),
                competitor_price=competitor_price[sku],
                seasonality=current_seasonality,
                promo=current_promo_flag
            )

            # Calculate profit using the advanced simulator, including vendor and CPI logic
            prof, _, _, _ = calculate_profit_advanced(
                price=float(p),
                cost=cost[sku],
                demand=d,
                competitor_price=competitor_price[sku],
                day=current_day,
                sku_type=sku_types[sku],
                is_promo_period=bool(current_promo_flag)
            )
            profit_map[(sku, p)] = int(prof) # Store as int for CP-SAT
            demand_map[(sku, p)] = d

    # OR-Tools Model Formulation
    model = cp_model.CpModel()

    # Decision Variables: x[(sku, p)] is 1 if SKU 'sku' is priced at 'p', 0 otherwise
    x = {
        (sku, p): model.NewBoolVar(f"x_{sku}_{p}")
        for sku in skus for p in price_grid
    }

    # Constraint 1: Exactly one price must be chosen for each SKU
    for sku in skus:
        model.Add(sum(x[(sku, p)] for p in price_grid) == 1)

    # Constraint 2: Inventory constraint
    # The total predicted demand for a chosen price must not exceed available inventory
    for sku in skus:
        # Sum of (demand for chosen price * 1) should be <= inventory
        model.Add(
            sum(demand_map[(sku, p)] * x[(sku, p)] for p in price_grid) <= inventory[sku]
        )

    # Constraint 3: CPI Compliance (100-107% of competitor price)
    for sku in skus:
        # Get the chosen price variable for the current SKU
        chosen_price_var = model.NewIntVar(min(price_grid), max(price_grid), f"chosen_price_{sku}")
        model.Add(
            chosen_price_var == sum(p * x[(sku, p)] for p in price_grid)
        )

        # Lower bound: price >= CPI_MIN_PERCENT/100 * competitor_price
        model.Add(chosen_price_var * 100 >= CPI_MIN_PERCENT * competitor_price[sku])
        # Upper bound: price <= CPI_MAX_PERCENT/100 * competitor_price
        model.Add(chosen_price_var * 100 <= CPI_MAX_PERCENT * competitor_price[sku])


    # Objective: Maximize total profit
    model.Maximize(
        sum(profit_map[(sku, p)] * x[(sku, p)]
            for sku in skus for p in price_grid)
    )

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"\nOptimization Status: {solver.StatusName(status)}")
        total_optimized_profit = 0
        print("\n=== DFL OPTIMIZED PRICING (OR-Tools) ===")
        for sku in skus:
            for p in price_grid:
                if solver.Value(x[(sku, p)]):
                    d = demand_map[(sku, p)]
                    prof = profit_map[(sku, p)]
                    total_optimized_profit += prof
                    cpi = (p / competitor_price[sku]) * 100 if competitor_price[sku] > 0 else 0
                    print(f"{sku} (Type: {sku_types[sku]}) → Price={p}, Comp_Price={competitor_price[sku]}, CPI={cpi:.2f}%, Demand={d}, Profit={prof}")
        print(f"\nTotal Profit (OR-Tools): {total_optimized_profit}")
    else:
        print(f"\nOR-Tools did not find an optimal solution. Status: {solver.StatusName(status)}")
        if status == cp_model.INFEASIBLE:
            print("The problem is infeasible. Check constraints.")