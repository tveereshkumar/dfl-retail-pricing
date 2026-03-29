import random

# Generate 100 SKUs (reduced from 1000 for faster execution in example)
skus = [f"SKU_{i}" for i in range(1, 101)]

# Assign SKU types (Vendor Product or Private Label)
sku_types = {sku: random.choice(["Vendor Product", "Private Label"]) for sku in skus}

# Base cost for each SKU
# Adjusted min cost to 25 to ensure competitor_price (min 30) allows for CPI compliance
# If competitor_price is 30, CPI range [30, 32.1] which overlaps with price_grid [30, 120]
cost = {sku: random.randint(25, 100) for sku in skus} # Changed min from 20 to 25

competitor_price = {sku: cost[sku] + random.randint(5, 30) for sku in skus}
# Now, minimum competitor_price is 25 (min cost) + 5 (min diff) = 30.
# This ensures that the CPI lower bound (100% of comp_price) is at least 30,
# which is the minimum value in our price_grid.

# Inventory for each SKU
inventory = {sku: random.randint(300, 1000) for sku in skus}

# Price grid for discrete price points
price_grid = list(range(30, 121, 5))

# CPI target range
CPI_MIN_PERCENT = 100
CPI_MAX_PERCENT = 107