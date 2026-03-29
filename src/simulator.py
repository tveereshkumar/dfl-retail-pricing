from typing import Tuple
from config import sku_types, CPI_MIN_PERCENT, CPI_MAX_PERCENT

def calculate_profit_advanced(
    price: float,
    cost: float,
    demand: int,
    competitor_price: float,
    day: int,
    sku_type: str,
    is_promo_period: bool = False
) -> Tuple[float, float, float, float]:
    """
    Calculates profit considering vendor allowances, CPI penalties, and inventory aging.
    This simulator is designed to be more robust for DFL.

    Args:
        price (float): The selling price of the SKU.
        cost (float): The cost of the SKU.
        demand (int): The predicted demand.
        competitor_price (float): The competitor's price for the SKU.
        day (int): The current day (for time decay).
        sku_type (str): Type of SKU ("Vendor Product" or "Private Label").
        is_promo_period (bool): Flag indicating if it's a promotional period.

    Returns:
        Tuple[float, float, float, float]: (profit, vendor_allowance, cpi_penalty, gross_margin)
    """

    gross_margin = (price - cost) * demand
    vendor_allowance = 0.0
    cpi_penalty = 0.0

    # 1. Vendor Allowances (more complex logic)
    if sku_type == "Vendor Product":
        if is_promo_period:
            # Example: Tiered allowance during promo
            if demand >= 800:
                vendor_allowance = 0.15 * price * demand # 15% of sales revenue
            elif demand >= 400:
                vendor_allowance = 0.10 * price * demand # 10% of sales revenue
            else:
                vendor_allowance = 500 # Flat allowance for lower demand promos
        else:
            # Example: Smaller, volume-based allowance during regular periods
            if demand >= 1000:
                vendor_allowance = 0.02 * price * demand # 2% of sales revenue
    # Private Label products typically have no vendor funding

    # 2. CPI Penalty/Bonus
    if competitor_price > 0:
        cpi = (price / competitor_price) * 100
        if cpi < CPI_MIN_PERCENT:
            # Penalty for being too cheap (race to bottom, brand erosion)
            cpi_penalty = (CPI_MIN_PERCENT - cpi) * 20 * demand # Higher penalty
        elif cpi > CPI_MAX_PERCENT:
            # Penalty for being too expensive (loss of market share)
            cpi_penalty = (cpi - CPI_MAX_PERCENT) * 10 * demand
        # else: CPI is within target, no penalty

    # 3. Time Decay (inventory aging) - stronger effect on profit
    decay_factor = 0.98 ** day # Exponential decay

    # Total Profit
    profit = (gross_margin + vendor_allowance - cpi_penalty) * decay_factor

    return profit, vendor_allowance, cpi_penalty, gross_margin