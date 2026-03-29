import numpy as np
from pymoo.core.problem import Problem
import torch
from utils.profit import compute_profit
from utils.cpi import compute_cpi
from utils.vendor import compute_vendor_allowance
from models.feature_builder import build_features

class RetailPricingProblem(Problem):
    def __init__(self, 
                 tft_model,
                 base_features,
                 costs,
                 competitor_prices,
                 weights,
                 thresholds,
                 price_options):

        self.tft_model = tft_model
        self.base_features = base_features
        self.costs = costs
        self.competitor_prices = competitor_prices
        self.weights = weights
        self.thresholds = thresholds
        self.price_options = price_options

        n_skus = len(costs)

        super().__init__(
            n_var=n_skus * 2,
            n_obj=2,
            n_constr=2,
            xl=np.array([0]*n_skus + [0]*n_skus),
            xu=np.array(
                [len(price_options[0]) - 1]*n_skus + [1]*n_skus
            ),
            type_var=int
        )

    def _evaluate(self, X, out, *args, **kwargs):

        F, G = [], []

        for sol in X:

            prices, promos = self.decode_solution(sol)

            prices_t = torch.tensor(prices).float()
            promos_t = torch.tensor(promos).float()

            features = build_features(self.base_features, prices_t, promos_t)

            # FIX: add time dimension
            features = features.unsqueeze(1)

            with torch.no_grad():
                demand = self.tft_model(features).squeeze().numpy()
                demand = demand * 100

            allowance = compute_vendor_allowance(
                demand, prices, promos, self.thresholds
            )

            profit = compute_profit(
                prices,
                demand,
                self.costs.numpy(),
                allowance,
                markdowns=0
            )

            cpi = compute_cpi(
                prices,
                self.competitor_prices.numpy(),
                self.weights.numpy()
            )

            F.append([
                -profit,
                abs(cpi - 1.03)
            ])

            G.append([
                1.00 - cpi,
                cpi - 1.07
            ])
        # print("DEBUG:", type(prices), type(self.costs), type(self.weights))
        out["F"] = np.array(F)
        out["G"] = np.array(G)

    def decode_solution(self, sol):
        n = len(sol) // 2

        price_idx = sol[:n]
        promos = sol[n:]

        prices = []

        for i in range(n):
            # ✅ FORCE INTEGER + CLIP RANGE
            idx = int(round(price_idx[i]))
            idx = max(0, min(idx, len(self.price_options[i]) - 1))

            prices.append(self.price_options[i][idx])

        prices = np.array(prices)

        # ✅ Ensure promos are 0/1
        promos = np.array([1 if p >= 0.5 else 0 for p in promos])

        return prices, promos

    def predict_demand(self, prices, promos):
        features = build_features(self.base_features, prices, promos)

        with torch.no_grad():
            demand = self.tft_model(features).numpy()

        return demand