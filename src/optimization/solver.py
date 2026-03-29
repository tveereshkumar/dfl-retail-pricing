from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

def run_optimization(problem):

    algorithm = NSGA2(
        pop_size=200,  # increased
        eliminate_duplicates=True
    )

    result = minimize(
        problem,
        algorithm,
        termination=('n_gen', 30),
        seed=42,
        verbose=False,
        save_history=False
    )

    return result