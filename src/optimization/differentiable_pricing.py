import torch
import torch.nn.functional as F


class DifferentiablePricingLayer:
    def __init__(self, price_options, temperature=1.0):
        self.price_options = [torch.tensor(p).float() for p in price_options]
        self.temperature = temperature

    def forward(self, demand, costs):

        prices = []

        for i, options in enumerate(self.price_options):

            d = demand[i]
            cost = costs[i]

            # Profit-aware scoring
            profit_options = (options - cost) * d

            probs = F.softmax(profit_options / self.temperature, dim=0)

            expected_price = (probs * options).sum()

            prices.append(expected_price)

        return torch.stack(prices)