import torch
import torch.optim as optim

from data.dummy_data import generate_dummy_data
from models.feature_builder import build_features
from models.tft_model import TFTDemandModel
from optimization.pricing_problem import RetailPricingProblem
from optimization.solver import run_optimization

from utils.profit import compute_profit
from utils.vendor import compute_vendor_allowance


def main():

    print("🔹 Generating data...")
    data, price_options = generate_dummy_data(n_skus=20)

    base_features = torch.tensor(data["base_features"]).float()
    costs = torch.tensor(data["costs"]).float()
    competitor_prices = torch.tensor(data["competitor_prices"]).float()
    weights = torch.tensor(data["weights"]).float()
    thresholds = data["thresholds"]

    print("🔹 Initializing TFT model...")
    model = TFTDemandModel(input_size=12)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("🔹 Starting DFL Training...\n")

    for epoch in range(20):

        problem = RetailPricingProblem(
            tft_model=model,
            base_features=base_features,
            costs=costs,
            competitor_prices=competitor_prices,
            weights=weights,
            thresholds=thresholds,
            price_options=price_options
        )

        result = run_optimization(problem)

        best_solution = result.X[0]

        prices, promos = problem.decode_solution(best_solution)

        # Convert to tensors
        prices_t = torch.tensor(prices).float()
        promos_t = torch.tensor(promos).float()

        # Predict demand
        features = build_features(base_features, prices_t, promos_t)
        demand = model(features.unsqueeze(1)).squeeze()

        allowance = compute_vendor_allowance(
            demand.detach().numpy(),  # still numpy logic
            prices,
            promos,
            thresholds
        )

        allowance_t = torch.tensor(allowance).float()

        profit = compute_profit(
            prices_t,
            demand,
            costs,
            allowance_t,
            markdowns=0
        )

        loss = -profit

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1} | Profit: {profit.item():.2f}")

    print("\n✅ Final Optimization Run...")

    final_problem = RetailPricingProblem(
        tft_model=model,
        base_features=base_features,
        costs=costs,
        competitor_prices=competitor_prices,
        weights=weights,
        thresholds=thresholds,
        price_options=price_options
    )

    final_result = run_optimization(final_problem)

    best_solution = final_result.X[0]
    prices, promos = final_problem.decode_solution(best_solution)

    print("\n📊 Final Prices:", prices)
    print("🎯 Promos:", promos)


if __name__ == "__main__":
    main()