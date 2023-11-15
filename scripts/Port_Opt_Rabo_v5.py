import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
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
capital_growth_factor = 1.6

# Choose sampler and solve qubo.
sampler = SimulatedAnnealingSampler()
sampler_kwargs = {"num_reads": 20, "num_sweeps": 200}

# Set up penalty coefficients for the constraints
lambdas1 = np.logspace(-16, 0, 15, endpoint=False, base=10.0)
lambdas2 = np.logspace(-16, 0, 15, endpoint=False, base=10.0)
lambdas4 = np.array([1])
lambdas3 = np.logspace(-16, 0, 15, endpoint=False, base=10.0)


portfolio_optimizer = PortfolioOptimizer("rabobank", kmin, kmax)
portfolio_optimizer.add_minimize_HHI(weights=lambdas1)
portfolio_optimizer.add_maximize_ROC(
    formulation=2,
    capital_growth_factor=capital_growth_factor,
    weights_roc=lambdas2,
    weights_stabilize=lambdas3,
)
portfolio_optimizer.add_emission_constraint(weights=lambdas4)
results = portfolio_optimizer.run(sampler, sampler_kwargs)
(x1, y1), (x2, y2), (x3, y3) = results.slice_results()


# Make a plot of the results.

# x/y_rabo1 corresponds to a front optimized including the emission target.
# x/y_rabo2 corresponds to a front optimized without the emission target.
with (Path(__file__).parent / "rabo_matlab.json").open(encoding="utf-8") as json_file:
    x_rabo1, y_rabo1, x_rabo2, y_rabo2 = json.load(json_file)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
colors = ["salmon", "goldenrod", "turquoise", "blue", "gray"]
labels = ["QUBO constraint not met", "QUBO reduced", "QUBO constraint met"]
labels += ["classical constrained", "classical unconstrained"]
x_values = [x3, x2, x1, x_rabo1, x_rabo2]
y_values = [y3, y2, y1, y_rabo1, y_rabo2]

for x_val, y_val, color, label in zip(x_values, y_values, colors, labels):
    plot_points(x_val, y_val, color=color, label=label, ax=ax1)
    plot_front(x_val, y_val, color=color, label=label, ax=ax2)

ax1.set_title("Points")
ax2.set_title("Pareto Front")
fig.tight_layout()

timestamp = datetime.now().strftime(r"%Y-%m-%d %H_%M_%S.%f")
fig.savefig(
    f"figures/Port_Opt_Rabo_v5_Cgf{100 * capital_growth_factor:.0f}_{timestamp}.png"
)
