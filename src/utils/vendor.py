import torch
import numpy as np

def compute_vendor_allowance(demand, prices, promos, thresholds):

    # -------- TORCH PATH (DFL training) --------
    if isinstance(demand, torch.Tensor):

        allowance = torch.zeros_like(demand)

        for i in range(len(demand)):
            condition = (promos[i] > 0.5) & (demand[i] >= thresholds[i])

            allowance[i] = torch.where(
                condition,
                0.2 * prices[i] * demand[i],
                torch.tensor(0.0, device=demand.device)
            )

        return allowance.sum()

    # -------- NUMPY PATH (PyMOO inference) --------
    else:

        demand = np.array(demand)
        prices = np.array(prices)
        promos = np.array(promos)
        thresholds = np.array(thresholds)

        allowance = 0.0

        for i in range(len(demand)):
            if promos[i] > 0.5 and demand[i] >= thresholds[i]:
                allowance += 0.2 * prices[i] * demand[i]

        return allowance