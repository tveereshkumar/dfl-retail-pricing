# =========================================================
# Combined Python Files - Wed Mar 25 23:34:14 IST 2026
# =========================================================

# ==================== FILE: config.py ====================

import random

# Generate 1000 SKUs
skus = [f"SKU_{i}" for i in range(1, 100)]

cost = {sku: random.randint(20, 100) for sku in skus}
competitor_price = {sku: cost[sku] + random.randint(5, 30) for sku in skus}
inventory = {sku: random.randint(300, 1000) for sku in skus}

price_grid = list(range(30, 121, 5))

# ========================================================

# ==================== FILE: data_generation.py ====================

# data_generation.py
import numpy as np
import pandas as pd
from config import skus, price_grid

np.random.seed(42)

def generate_data(num_days=30):

    data = []

    for sku in skus:
        base = np.random.randint(800, 1200)   # ✅ bigger demand scale

        price_sensitivity = np.random.uniform(6, 10)  # ✅ SKU-level variation

        for day in range(num_days):

            seasonality = 50 * np.sin(day / 5)   # stronger seasonality
            promo = np.random.choice([0, 1], p=[0.7, 0.3])

            for price in price_grid:

                demand = (
                    base
                    - price_sensitivity * (price ** 1.3)   # dynamic slope
                    + seasonality
                    + 100 * promo                # stronger promo effect
                    + np.random.normal(0, 20)    # realistic noise
                )

                data.append([
                    sku,
                    day,
                    price,
                    seasonality,
                    promo,
                    max(0, demand)
                ])

    df = pd.DataFrame(
        data,
        columns=["sku", "day", "price", "seasonality", "promo", "demand"]
    )

    return df

# ========================================================

# ==================== FILE: deep_demand_model.py ====================

import torch
import torch.nn as nn
import numpy as np

SEQ_LEN = 5  # use past 5 days

class DemandModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def create_sequences(df_sku):
    X, y = [], []

    df_sku = df_sku.sort_values("day")

    features = df_sku[["price", "seasonality", "promo"]].values
    target = df_sku["demand"].values

    for i in range(len(df_sku) - SEQ_LEN):
        X.append(features[i:i+SEQ_LEN])
        y.append(target[i+SEQ_LEN])

    return np.array(X), np.array(y)


def train_models(df):
    models = {}

    for sku in df["sku"].unique():
        df_sku = df[df["sku"] == sku]

        X, y = create_sequences(df_sku)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        model = DemandModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()

        for epoch in range(100):
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        models[sku] = model

    return models


def predict(model, price, seasonality, promo):
    seq = []

    for i in range(SEQ_LEN):
        seq.append([
            price,
            seasonality + np.random.normal(0, 2),
            promo
        ])

    seq = torch.tensor([seq], dtype=torch.float32)

    with torch.no_grad():
        return max(0, int(model(seq).item()))

# ========================================================

# ==================== FILE: main.py ====================

from data_generation import generate_data
from deep_demand_model import train_models
from optimization import run_optimization
from pymoo_optimization import PricingProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from config import cost, competitor_price
from visualization import plot_price_vs_profit

def run_pymoo(models):
    problem = PricingProblem(models)
    algo = NSGA2(pop_size=50)

    res = minimize(problem, algo, ('n_gen', 50), verbose=False)

    print("\n=== PyMOO Pareto Solutions ===")
    for sol in res.X[:5]:
        print(sol)


def main():
    print("Generating data...")
    df = generate_data()

    print("Training Deep Learning model...")
    models = train_models(df)

    if not models:
        print("No models trained!")
        return

    print("\nRunning OR-Tools Optimization...")
    run_optimization(models)

    print("\nRunning PyMOO Optimization...")
    run_pymoo(models)

    print("\nPlotting Price vs Profit curve...")
    # for sku in list(models.keys())[:3]:
    #     plot_price_vs_profit(
    #         models[sku],
    #         sku,
    #         cost[sku],
    #         competitor_price[sku]
    #     )


if __name__ == "__main__":
    main()

# ========================================================

# ==================== FILE: optimization.py ====================

from ortools.sat.python import cp_model
from config import skus, cost, competitor_price, inventory, price_grid
from simulator import calculate_profit
from deep_demand_model import predict
import random

def run_optimization(models):

    promo = random.choice([0, 1])
    day = random.randint(5, 25)

    profit_dict = {}
    demand_dict = {}

    # Precompute (Simulator)
    for sku in skus:
        for p in price_grid:
            seasonality = random.uniform(-30, 50)

            d = predict(models[sku], p, seasonality, promo)
            
            if p > competitor_price[sku]:
                d = int(d * 0.5)

            profit, _, _ = calculate_profit(
                p,
                cost[sku],
                d,
                competitor_price[sku],
                day   
            )

            profit_dict[(sku, p)] = int(profit)
            demand_dict[(sku, p)] = d

    # OR-Tools Model
    model = cp_model.CpModel()

    x = {
        (sku, p): model.NewBoolVar(f"x_{sku}_{p}")
        for sku in skus for p in price_grid
    }

    # Constraint: one price per SKU
    for sku in skus:
        model.Add(sum(x[(sku, p)] for p in price_grid) == 1)

    # Constraint: inventory
    for sku in skus:
        model.Add(
            sum(demand_dict[(sku, p)] * x[(sku, p)]
                for p in price_grid) <= inventory[sku]
        )

    # Objective: maximize profit
    model.Maximize(
        sum(profit_dict[(sku, p)] * x[(sku, p)]
            for sku in skus for p in price_grid)
    )

    # Solve
    solver = cp_model.CpSolver()
    solver.Solve(model)

    print("\n=== DFL OPTIMIZED PRICING ===\n")

    total = 0

    for sku in skus:
        for p in price_grid:
            if solver.Value(x[(sku, p)]):
                d = demand_dict[(sku, p)]
                prof = profit_dict[(sku, p)]
                total += prof

                print(f"{sku} → Price={p}, Demand={d}, Profit={prof}")

    print(f"\nTotal Profit (DFL): {total}")

# ========================================================

# ==================== FILE: pymoo_optimization.py ====================

import numpy as np
from pymoo.core.problem import Problem
from config import skus, cost, competitor_price
from deep_demand_model import predict

class PricingProblem(Problem):

    def __init__(self, models):
        super().__init__(
            n_var=len(skus),
            n_obj=2,
            xl=np.array([30]*len(skus)),
            xu=np.array([120]*len(skus))
        )
        self.models = models

    def _evaluate(self, X, out, *args, **kwargs):

        profits, cpi_dev = [], []

        for sol in X:
            total_profit = 0
            total_cpi = 0

            for i, sku in enumerate(skus):
                price = int(sol[i])
                d = predict(self.models[sku], price, 1, 1)

                profit = (price - cost[sku]) * d
                total_profit += profit
                total_cpi += abs(price - competitor_price[sku])

            profits.append(-total_profit)
            cpi_dev.append(total_cpi)

        out["F"] = np.column_stack([profits, cpi_dev])

# ========================================================

# ==================== FILE: simulator.py ====================

def calculate_profit(price, cost, demand, competitor_price, day):
    
    # Vendor funding increases during promo days
    vendor = 0
    if price <= competitor_price and demand > 120:
        vendor = 30 * demand
    
    # CPI penalty
    penalty = max(0, (competitor_price - price) * 50)

    # Time decay (inventory aging)
    decay = 0.98 ** day

    profit = ((price - cost) * demand + vendor - penalty) * decay

    return profit, vendor, penalty

# ========================================================

# ==================== FILE: visualization.py ====================

import matplotlib.pyplot as plt
from config import price_grid
from simulator import calculate_profit
from deep_demand_model import predict


def plot_price_vs_profit(model, sku, cost, competitor_price):

    profits = []
    demands = []

    for p in price_grid:
        d = predict(model, p, 20, 1)
        prof, _, _ = calculate_profit(p, cost, d, competitor_price, day=10)

        profits.append(prof)
        demands.append(d)

    plt.figure()
    plt.plot(price_grid, profits)
    plt.title(f"Price vs Profit for {sku}")
    plt.xlabel("Price")
    plt.ylabel("Profit")
    plt.show()

# ========================================================

