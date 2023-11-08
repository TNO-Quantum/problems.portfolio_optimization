from datetime import datetime

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler

from tno.quantum.problems.portfolio_optimization import (
    PortfolioOptimizer,
    plot_front,
    plot_points,
)

# Define the precision of the portfolio sizes.
kmax = 2  # 2 #number of values
kmin = 0  # minimal value 2**kmin

# Choose sampler and solve qubo.
sampler = SimulatedAnnealingSampler()
sampler_kwargs = {"num_reads": 20, "num_sweeps": 200}

# Set up penalty coefficients for the constraints
lambdas1 = np.logspace(-17, 2, 99, endpoint=False, base=10.0)
lambdas2 = np.logspace(-17, -2, 99, endpoint=False, base=10.0)
lambdas3 = np.array([1])


portfolio_optimizer = PortfolioOptimizer("rabobank", kmin, kmax)
portfolio_optimizer.add_minimize_HHI(weights=lambdas1)
portfolio_optimizer.add_maximize_ROC(formulation=4, weights_roc=lambdas2)
portfolio_optimizer.add_emission_constraint(weights=lambdas3)
results = portfolio_optimizer.run(sampler, sampler_kwargs)
results.slice_results()

# Make a plot of the results.
timestamp = datetime.now().strftime(r"%Y-%m-%d %H_%M_%S.%f")

fig = plot_points(results, "mediumblue", "mediumorchid", "crimson")
fig.savefig(f"figures/Port_Opt_Rabo_v7_points_{timestamp}.png")

fig = plot_front(results, "mediumblue", "mediumorchid", "crimson")
fig.savefig(f"figures/Port_Opt_Rabo_v7_front_{timestamp}.png")
