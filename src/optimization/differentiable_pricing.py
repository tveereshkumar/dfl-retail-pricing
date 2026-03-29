import torch
import torch.nn.functional as F

class DifferentiablePricingLayer:
    def __init__(self, price_options, temperature=5.0):
        self.price_options = [torch.tensor(p).float() for p in price_options]
        self.temperature = temperature

    def forward(self, demand, costs, competitor_prices, weights):

        prices = []

        for i, options in enumerate(self.price_options):

            d = demand[i]
            cost = costs[i]
            comp_price = competitor_prices[i]
            weight = weights[i]

            # 🔥 Elastic demand per option
            demand_adj = d * torch.exp(-0.4 * options)

            # 🔥 Margin
            margin = torch.clamp(options - cost, min=0.01)

            # 🔥 Profit
            profit = margin * demand_adj

            # 🔥 CPI penalty per SKU (local approximation)
            cpi_local = options / comp_price
            cpi_penalty = torch.relu(cpi_local - 1.07) + torch.relu(1.00 - cpi_local)

            # 🔥 FINAL objective (THIS IS KEY)
            score = profit - 50 * cpi_penalty

            probs = torch.softmax(score / self.temperature, dim=0)

            expected_price = (probs * options).sum()

            prices.append(expected_price)

        return torch.stack(prices)