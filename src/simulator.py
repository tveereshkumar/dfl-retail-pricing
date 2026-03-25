def calculate_profit(p, cost, demand, competitor_price):
    # Vendor rule
    vendor = 0
    if p <= competitor_price and demand >= 120:
        vendor = 15 * demand

    # CPI penalty
    penalty = 0
    if p < competitor_price:
        penalty = (competitor_price - p) * 10

    profit = (p - cost) * demand + vendor - penalty

    return profit, vendor, penalty