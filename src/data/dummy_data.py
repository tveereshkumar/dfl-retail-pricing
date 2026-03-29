import numpy as np

def generate_dummy_data(n_skus=20):

    data = {
        "base_features": np.random.rand(n_skus, 10),
        "costs": np.random.uniform(5, 15, n_skus),
        "competitor_prices": np.random.uniform(10, 25, n_skus),
        "weights": np.random.uniform(0.5, 2.0, n_skus),
        "thresholds": np.random.randint(50, 200, n_skus),
    }

    # Each SKU has 5 candidate prices
    price_options = [
        np.linspace(10, 30, 5) for _ in range(n_skus)
    ]

    return data, price_options