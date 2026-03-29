import numpy as np

def compute_cpi(prices, competitor_prices, weights):

    prices = np.array(prices)
    competitor_prices = np.array(competitor_prices)
    weights = np.array(weights)

    denominator = (weights * competitor_prices).sum()

    if denominator == 0:
        return 1.0  # safe fallback

    return (weights * prices).sum() / denominator