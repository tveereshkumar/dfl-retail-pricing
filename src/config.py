# Configuration for pricing system

skus = ["SKU_1", "SKU_2", "SKU_3"]

cost = {
    "SKU_1": 90,
    "SKU_2": 50,
    "SKU_3": 30
}

competitor_price = {
    "SKU_1": 100,
    "SKU_2": 60,
    "SKU_3": 40
}

inventory = {
    "SKU_1": 500,
    "SKU_2": 600,
    "SKU_3": 700
}

price_grid = list(range(30, 121, 5))