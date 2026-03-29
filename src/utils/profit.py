import numpy as np
import torch

def compute_profit(prices, demand, costs, allowances, markdowns):

    # If tensors → use torch (for training)
    if isinstance(demand, torch.Tensor):
        revenue = prices * demand
        total_cost = costs * demand
        return (revenue - total_cost + allowances - markdowns).sum()

    # Else → fallback to numpy (for optimization)
    prices = np.array(prices)
    demand = np.array(demand)
    costs = np.array(costs)

    revenue = prices * demand
    total_cost = costs * demand

    return (revenue - total_cost + allowances - markdowns).sum()