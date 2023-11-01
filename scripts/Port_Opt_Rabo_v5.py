from datetime import datetime

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler

from tno.quantum.problems.portfolio_optimization.io import read_portfolio_data
from tno.quantum.problems.portfolio_optimization.portfolio_optimizer import (
    PortfolioOptimizer,
)
from tno.quantum.problems.portfolio_optimization.preprocessing import print_info
from tno.quantum.problems.portfolio_optimization.visualization import (
    plot_front,
    plot_points,
)

# Define the precision of the portfolio sizes.
kmax = 2  # 2 #number of values
kmin = 0  # minimal value 2**kmin
capital_growth_factor = 1.6

# Algorithm variables
steps1 = 15
steps2 = 15
steps3 = 1
steps4 = 15

df = read_portfolio_data("rabodata.xlsx")
print_info(df)


# Creating the actual model to optimize using the annealer.
print("Status: creating model")

print("Status: calculating")
starttime = datetime.now()

# Choose sampler and solve qubo.
sampler = SimulatedAnnealingSampler()
sampler_kwargs = {"num_reads": 20, "num_sweeps": 200}

# Set up penalty coefficients for the constraints
labdas1 = np.logspace(-16, 0, steps1, endpoint=False, base=10.0)
labdas2 = np.logspace(-16, 0, steps2, endpoint=False, base=10.0)
labdas4 = np.array([1])
labdas3 = np.logspace(-16, 0, steps4, endpoint=False, base=10.0)


portfolio_optimizer = PortfolioOptimizer(df, kmin, kmax)
portfolio_optimizer.add_minimize_HHI(weights=labdas1)
portfolio_optimizer.add_maximize_ROC(
    formulation=2,
    capital_growth_factor=capital_growth_factor,
    weights_roc=labdas2,
    weights_stabilize=labdas3,
)
portfolio_optimizer.add_emission_constraint(weights=labdas4)
results = portfolio_optimizer.run(sampler, sampler_kwargs)


print(
    "Number of generated samples: ", len(results.x1), len(results.x2), len(results.x3)
)
print("Time consumed:", datetime.now() - starttime)


# Make a plot of the results.
fig = plot_points(results, "turquoise", "goldenrod", "salmon")
# Name to save the figure under.
string_format = r"%Y-%m-%d %H_%M_%S.%f"
name = f"figures/Port_Opt_Rabo_v5_Cgf{int(100 * capital_growth_factor)}_points_"
name += f"{datetime.now().strftime(string_format)}.png"
fig.savefig(name)

fig = plot_front(results, "turquoise", "goldenrod", "salmon")
# Name to save the figure under.
name = f"figures/Port_Opt_Rabo_v5_Cgf{int(100 * capital_growth_factor)}_front_"
name += f"{datetime.now().strftime(string_format)}.png"
fig.savefig(name)
