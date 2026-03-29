import numpy as np
import pandas as pd
from config import skus, price_grid, competitor_price, sku_types

np.random.seed(42)

def generate_data(num_days: int = 30) -> pd.DataFrame:
    """
    Generates synthetic demand data for multiple SKUs across a range of days and prices.
    Includes SKU-level price sensitivity, competitor price influence, seasonality,
    promotional effects, and realistic noise.

    Args:
        num_days (int): The number of days for which to generate data.

    Returns:
        pd.DataFrame: A DataFrame containing generated demand data.
    """
    data = []
    for sku in skus:
        # SKU-specific base demand and sensitivities
        base_demand = np.random.randint(800, 1200)
        price_sensitivity = np.random.uniform(6, 10)
        competitor_price_sensitivity = np.random.uniform(0.5, 2.0) # How much competitor's price affects demand

        for day in range(num_days):
            # Stronger seasonality effect
            seasonality = 50 * np.sin(day / 5)

            # Promotional flag (30% chance of promo)
            promo = np.random.choice([0, 1], p=[0.7, 0.3])

            for price in price_grid:
                # Get competitor price for the SKU
                current_competitor_price = competitor_price[sku]

                # Dynamic demand function:
                # - Base demand
                # - Stronger price elasticity (price ** 1.3)
                # - Influence from competitor price (e.g., if our price is much higher, demand drops)
                # - Seasonality
                # - Stronger promotional uplift
                # - Gaussian noise
                demand = (
                    base_demand
                    - price_sensitivity * (price ** 1.3)
                    + competitor_price_sensitivity * (current_competitor_price - price) # Demand increases if our price is lower
                    + seasonality
                    + 100 * promo
                    + np.random.normal(0, 20)
                )

                data.append([
                    sku,
                    day,
                    price,
                    current_competitor_price, # Include competitor price in features
                    seasonality,
                    promo,
                    max(0, demand) # Demand cannot be negative
                ])

    df = pd.DataFrame(
        data,
        columns=["sku", "day", "price", "competitor_price", "seasonality", "promo", "demand"]
    )
    return df