from ortools.sat.python import cp_model
from config import skus, cost, competitor_price, inventory, price_grid
from simulator import calculate_profit

def run_optimization(models):

    season = 1
    promo = 1

    profit_dict = {}
    demand_dict = {}

    # Precompute
    for sku in skus:
        for p in price_grid:
            pred_demand = int(models[sku].predict([[p, season, promo]])[0])
            pred_demand = max(0, pred_demand)

            profit, vendor, penalty = calculate_profit(
                p,
                cost[sku],
                pred_demand,
                competitor_price[sku]
            )

            profit_dict[(sku, p)] = int(profit)
            demand_dict[(sku, p)] = pred_demand

    # OR-Tools model
    model_cp = cp_model.CpModel()

    x = {}
    for sku in skus:
        for p in price_grid:
            x[(sku, p)] = model_cp.NewBoolVar(f"x_{sku}_{p}")

    # One price per SKU
    for sku in skus:
        model_cp.Add(sum(x[(sku, p)] for p in price_grid) == 1)

    # Inventory constraint
    for sku in skus:
        model_cp.Add(
            sum(demand_dict[(sku, p)] * x[(sku, p)] for p in price_grid)
            <= inventory[sku]
        )

    # Objective
    model_cp.Maximize(
        sum(profit_dict[(sku, p)] * x[(sku, p)]
            for sku in skus for p in price_grid)
    )

    solver = cp_model.CpSolver()
    solver.Solve(model_cp)

    print("\n=== OPTIMIZED PRICING ===\n")

    total_profit = 0

    for sku in skus:
        for p in price_grid:
            if solver.Value(x[(sku, p)]) == 1:
                d = demand_dict[(sku, p)]
                prof = profit_dict[(sku, p)]
                total_profit += prof

                print(f"{sku} → Price: {p}, Demand: {d}, Profit: {prof}")

    print(f"\nTotal Profit: {total_profit}")

    # PTO baseline
    print("\n=== PTO BASELINE ===\n")

    pto_profit = 0

    for sku in skus:
        best_price = max(price_grid, key=lambda p: p - cost[sku])
        prof = profit_dict[(sku, best_price)]
        pto_profit += prof

        print(f"{sku} → Price: {best_price}, Profit: {prof}")

    print(f"\nTotal PTO Profit: {pto_profit}")
    print(f"Improvement: {total_profit - pto_profit}")