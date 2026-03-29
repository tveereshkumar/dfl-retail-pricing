import torch
import torch.optim as optim

from data.dummy_data import generate_dummy_data
from models.feature_builder import build_features
from models.tft_model import TFTDemandModel

from utils.profit import compute_profit
from utils.vendor import compute_vendor_allowance

from optimization.differentiable_pricing import DifferentiablePricingLayer
from optimization.pricing_problem import RetailPricingProblem
from optimization.solver import run_optimization


# 🔥 HARD CPI ENFORCEMENT (GLOBAL)
def enforce_cpi_iterative(prices, competitor_prices, weights, iters=5, target=1.03):
    for _ in range(iters):
        cpi = (weights * prices).sum() / (weights * competitor_prices).sum()
        scale = target / (cpi + 1e-6)
        prices = prices * scale
    return prices, cpi


def main():

    print("🔹 Generating data...")
    data, price_options = generate_dummy_data(n_skus=20)

    base_features = torch.tensor(data["base_features"]).float()
    costs = torch.tensor(data["costs"]).float()
    competitor_prices = torch.tensor(data["competitor_prices"]).float()
    weights = torch.tensor(data["weights"]).float()
    thresholds = torch.tensor(data["thresholds"]).float()

    print("🔹 Initializing TFT model...")
    input_size = base_features.shape[1] + 2
    model = TFTDemandModel(input_size=input_size)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("🔹 Starting DFL Training...\n")

    pricing_layer = DifferentiablePricingLayer(price_options)

    for epoch in range(20):

        dummy_prices = torch.zeros_like(costs)
        dummy_promos = torch.zeros_like(costs)

        # ---- Pass 1 ----
        features = build_features(base_features, dummy_prices, dummy_promos)
        demand = model(features.unsqueeze(1)).squeeze()

        # ---- Pricing ----
        prices_t = pricing_layer.forward(demand, costs)

        # 🔥 HARD CPI FIX
        prices_t, cpi = enforce_cpi_iterative(
            prices_t,
            competitor_prices,
            weights
        )

        # ---- Promotions ----
        promos_t = torch.sigmoid(0.1 * (thresholds - demand))

        # ---- Pass 2 ----
        features = build_features(base_features, prices_t, promos_t)
        demand = model(features.unsqueeze(1)).squeeze()

        # ---- Elasticity ----
        demand = demand * torch.exp(-0.05 * prices_t)

        # ---- Margin awareness ----
        demand = demand * torch.sigmoid(prices_t - costs)

        # ---- Clamp demand (stability) ----
        demand = torch.clamp(demand, 0, 200)

        # ---- Vendor ----
        allowance = compute_vendor_allowance(
            demand,
            prices_t,
            promos_t,
            thresholds
        )

        # ---- Profit ----
        profit = compute_profit(
            prices_t,
            demand,
            costs,
            allowance,
            markdowns=0
        )

        loss = -profit  # CPI already enforced

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_cpi = (weights * prices_t).sum() / (weights * competitor_prices).sum()
        print(
            f"Epoch {epoch+1} | Profit: {profit.item():.2f} | "
            f"CPI: {final_cpi.item():.3f} | Price Avg: {prices_t.mean().item():.2f}"
        )

    # ---------------- FINAL DISCRETE OPT ----------------

    print("\n✅ Final Optimization Run...")

    final_problem = RetailPricingProblem(
        tft_model=model,
        base_features=base_features,
        costs=costs,
        competitor_prices=competitor_prices,
        weights=weights,
        thresholds=thresholds.numpy(),
        price_options=price_options
    )

    final_result = run_optimization(final_problem)

    best_solution = final_result.X[0]
    prices, promos = final_problem.decode_solution(best_solution)

    print("\n📊 Final Prices:", prices)
    print("🎯 Promos:", promos)


if __name__ == "__main__":
    main()