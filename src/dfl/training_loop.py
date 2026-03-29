import torch
import torch.optim as optim

def train_dfl(model, problem_builder, train_data, epochs=10):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):

        for batch in train_data:

            base_features = batch["features"]

            problem = problem_builder(model, base_features)

            result = run_optimization(problem)

            best_solution = result.X[0]

            prices, promos = problem.decode_solution(best_solution)

            model(features(price, promo))

            allowance = compute_vendor_allowance(
                demand, prices, promos, problem.thresholds
            )

            profit = compute_profit(
                prices, demand, problem.costs, allowance, 0
            )

            loss = -profit  # maximize profit

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} | Profit: {-loss.item()}")