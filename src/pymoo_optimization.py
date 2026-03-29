import numpy as np
from pymoo.core.problem import Problem
from config import skus, cost, competitor_price, inventory, sku_types, \
                   CPI_MIN_PERCENT, CPI_MAX_PERCENT, price_grid
from deep_demand_model import predict_demand
from simulator import calculate_profit_advanced
from typing import Dict, List

class DFLPricingProblem(Problem):
    """
    PyMOO Problem formulation for Multi-Objective Optimization (MOO) of pricing.
    Objectives: Maximize Total Gross Profit, Minimize CPI Deviation from Target Range.
    Constraints: CPI bounds (100-107%), Inventory.
    """
    def __init__(
        self,
        models: Dict[str, 'AdvancedDemandModel'],
        current_seasonality: float,
        current_promo: int,
        current_day: int
    ):
        # Number of decision variables: one price for each SKU
        num_skus = len(skus)

        # Number of objectives:
        # F1: Negative Total Gross Profit (to minimize)
        # F2: Total CPI Deviation from Target Range (to minimize)
        n_obj = 2

        # Number of constraints per solution:
        # For each SKU:
        #   1. CPI lower bound constraint (price must be >= 100% of competitor_price)
        #   2. CPI upper bound constraint (price must be <= 107% of competitor_price)
        #   3. Inventory constraint (predicted_demand <= inventory)
        n_constr = num_skus * 3

        # Lower and upper bounds for decision variables (prices)
        # Ensure prices are within the defined price_grid range
        xl = np.array([min(price_grid)] * num_skus)
        xu = np.array([max(price_grid)] * num_skus)

        super().__init__(
            n_var=num_skus,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu
        )
        self.models = models
        self.current_seasonality = current_seasonality
        self.current_promo = current_promo
        self.current_day = current_day

        # Store config data locally for easier access in _evaluate
        self._skus = skus
        self._cost = cost
        self._competitor_price = competitor_price
        self._inventory = inventory
        self._sku_types = sku_types

    def _evaluate(self, X: np.ndarray, out: Dict, *args, **kwargs):
        """
        Evaluates a batch of solutions (price vectors) for their objectives and constraints.

        Args:
            X (np.ndarray): A 2D array where each row is a solution (price vector for all SKUs).
                            Shape: (num_solutions, num_skus)
            out (Dict): Dictionary to store objectives ("F") and constraints ("G").
        """
        # Initialize lists to store objectives and constraints for all solutions in the batch
        all_neg_profits: List[float] = []
        all_cpi_deviations: List[float] = []
        all_constraints: List[List[float]] = []

        for sol_idx, sol in enumerate(X): # Iterate through each solution (a price vector for all SKUs)
            total_profit = 0.0
            total_cpi_deviation = 0.0
            current_solution_constraints: List[float] = []

            for sku_idx, sku in enumerate(self._skus):
                # Ensure price is an integer and within the valid grid (or closest to it)
                # PyMOO can propose continuous values, round them to nearest grid point or integer
                price = int(np.round(sol[sku_idx]))
                price = min(max(price, min(price_grid)), max(price_grid)) # Clamp to bounds
                # If using exact price grid points, one might snap to the nearest grid point:
                # price = min(price_grid, key=lambda x: abs(x - price))


                comp_price = self._competitor_price[sku]

                # 1. Predict Demand using the trained Deep Learning Model
                predicted_demand = predict_demand(
                    self.models[sku],
                    price=float(price),
                    competitor_price=comp_price,
                    seasonality=self.current_seasonality, # Simplified as global for all SKUs in this run
                    promo=self.current_promo
                )

                # 2. Calculate Profit using the Advanced Simulator
                profit, _, _, _ = calculate_profit_advanced(
                    price=float(price),
                    cost=self._cost[sku],
                    demand=predicted_demand,
                    competitor_price=comp_price,
                    day=self.current_day,
                    sku_type=self._sku_types[sku],
                    is_promo_period=bool(self.current_promo)
                )
                total_profit += profit

                # 3. Calculate CPI and its Deviation (for Objective and Constraints)
                cpi_value = (price / comp_price) * 100 if comp_price > 0 else 0.0

                # Objective F2: Minimize total deviation from the CPI target range [100, 107]
                # Deviation is 0 if within range, otherwise the absolute distance to the closest bound.
                total_cpi_deviation += max(0, CPI_MIN_PERCENT - cpi_value, cpi_value - CPI_MAX_PERCENT)

                # Constraints G: These must be <= 0 for a solution to be feasible

                # Constraint G1: CPI Lower Bound (price must be at least 100% of competitor_price)
                # If cpi_value < CPI_MIN_PERCENT, then (CPI_MIN_PERCENT - cpi_value) > 0, violating constraint.
                current_solution_constraints.append(CPI_MIN_PERCENT - cpi_value)

                # Constraint G2: CPI Upper Bound (price must be at most 107% of competitor_price)
                # If cpi_value > CPI_MAX_PERCENT, then (cpi_value - CPI_MAX_PERCENT) > 0, violating constraint.
                current_solution_constraints.append(cpi_value - CPI_MAX_PERCENT)

                # Constraint G3: Inventory Constraint (predicted_demand must not exceed inventory)
                # If predicted_demand > inventory, then (predicted_demand - inventory) > 0, violating constraint.
                current_solution_constraints.append(predicted_demand - self._inventory[sku])

            all_neg_profits.append(-total_profit) # PyMOO minimizes, so minimize negative profit
            all_cpi_deviations.append(total_cpi_deviation)
            all_constraints.append(current_solution_constraints)

        # Store objectives and constraints in the 'out' dictionary
        out["F"] = np.column_stack([all_neg_profits, all_cpi_deviations])
        out["G"] = np.array(all_constraints)