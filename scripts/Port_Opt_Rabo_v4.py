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
growth_target = 1.55


# Choose sampler and solve qubo.
sampler = SimulatedAnnealingSampler()
sampler_kwargs = {"num_reads": 20, "num_sweeps": 200}

# Set up penalty coefficients for the constraints
labdas1 = np.logspace(-16, 0, 21, endpoint=False, base=10.0)
labdas2 = np.logspace(-16, 0, 21, endpoint=False, base=10.0)
labdas3 = np.array([1])
labdas4 = np.logspace(-16, 0, 21, endpoint=False, base=10.0)


portfolio_optimizer = PortfolioOptimizer("rabobank", kmin, kmax)
portfolio_optimizer.add_minimize_HHI(weights=labdas1)
portfolio_optimizer.add_maximize_ROC(formulation=1, weights_roc=labdas2)
portfolio_optimizer.add_emission_constraint(weights=labdas3)
portfolio_optimizer.add_growth_factor_constraint(growth_target, weights=labdas4)
results = portfolio_optimizer.run(sampler, sampler_kwargs)
results.slice_results(growth_target)


# Make a plot of the results.
timestamp = datetime.now().strftime(r"%Y-%m-%d %H_%M_%S.%f")

fig = plot_points(results, "coral", "gold", "dodgerblue")
fig.savefig(f"figures/Port_Opt_Rabo_v4_GT{growth_target}_points_{timestamp}.png")

fig = plot_front(results, "coral", "gold", "dodgerblue")
fig.savefig(f"figures/Port_Opt_Rabo_v4_GT{growth_target}_fronts_{timestamp}.png")
