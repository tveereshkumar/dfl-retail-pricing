import numpy as np
import pandas as pd
from config import skus, price_grid

np.random.seed(42)

def generate_data():
    data = []

    for sku in skus:
        for p in price_grid:
            for season in [0, 1]:
                for promo in [0, 1]:

                    base = {
                        "SKU_1": 300,
                        "SKU_2": 250,
                        "SKU_3": 200
                    }[sku]

                    demand = (
                        base
                        - 2.5 * p   # strong price sensitivity
                        + 30 * season
                        + 50 * promo
                        + np.random.normal(0, 5)
                    )

                    data.append([
                        sku, p, season, promo, max(0, demand)
                    ])

    df = pd.DataFrame(
        data,
        columns=["sku", "price", "season", "promo", "demand"]
    )

    df.to_csv("data/synthetic_data.csv", index=False)

    return df