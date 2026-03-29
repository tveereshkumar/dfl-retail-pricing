import torch
import numpy as np


def compute_vendor_allowance(demand, prices, promos, thresholds):

    if isinstance(demand, torch.Tensor):

        condition = (promos > 0.5) & (demand >= thresholds)
        allowance = 0.2 * prices * demand * condition.float()

        return allowance.sum()

    else:
        demand = np.array(demand)
        prices = np.array(prices)
        promos = np.array(promos)
        thresholds = np.array(thresholds)

        condition = (promos > 0.5) & (demand >= thresholds)

        return np.sum(0.2 * prices * demand * condition)