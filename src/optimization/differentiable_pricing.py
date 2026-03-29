import torch
import torch.nn.functional as F

class DifferentiablePricingLayer:
    def __init__(self, price_options, temperature=1.0):
        self.price_options = [torch.tensor(p).float() for p in price_options]
        self.temperature = temperature

    def forward(self, demand, costs, weights, competitor_prices):

        prices = []

        # Precompute denominator for CPI
        cpi_denominator = (weights * competitor_prices).sum()

        for i, options in enumerate(self.price_options):

            d = demand[i]
            cost = costs[i]
            w = weights[i]

            # Base profit
            profit_options = (options - cost) * d

            # ---- CPI-aware penalty ----
            # simulate CPI if this SKU changes
            approx_cpi = (w * options) / cpi_denominator

            cpi_penalty = torch.relu(approx_cpi - 1.07) + torch.relu(1.00 - approx_cpi)

            # 🔥 FINAL objective per option
            score = profit_options - 20 * cpi_penalty

            probs = F.softmax(score / self.temperature, dim=0)

            expected_price = (probs * options).sum()

            prices.append(expected_price)

        return torch.stack(prices)