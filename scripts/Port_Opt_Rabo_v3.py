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
kmax = 2  # 2 # number of values
kmin = 0  # minimal value 2**kmin

# Algorithm variables
steps1 = 25
steps2 = 25
steps3 = 1

df = read_portfolio_data("rabodata.xlsx")
print_info(df)

# Creating the actual model to optimize using the annealer.
print("Status: creating model")

print("Status: calculating")
starttime = datetime.now()

# Choose sampler and solve qubo. This is the actual optimization with either a DWave
# system or a simulated annealer.
sampler = SimulatedAnnealingSampler()
sampler_kwargs = {"num_reads": 20, "num_sweeps": 200}


# Set up penalty coefficients for the constraints
labdas1 = np.logspace(-16, 1, steps1, endpoint=False, base=10.0)
labdas2 = np.logspace(-16, 1, steps2, endpoint=False, base=10.0)
labdas3 = np.array([1])


portfolio_optimizer = PortfolioOptimizer(df, kmin, kmax)
portfolio_optimizer.add_minimize_HHI(weights=labdas1)
portfolio_optimizer.add_maximize_ROC(formulation=1, weights_roc=labdas1)
portfolio_optimizer.add_emission_constraint(weights=labdas3)
results = portfolio_optimizer.run(sampler, sampler_kwargs)


print(
    "Number of generated samples: ", len(results.x1), len(results.x2), len(results.x3)
)
print("Time consumed:", datetime.now() - starttime)


# Make a plot of the results.
fig = plot_points(results, "green", "orange", "red")
# Name to save the figure under.
string_format = r"%Y-%m-%d %H_%M_%S.%f"
name = f"figures/Port_Opt_Rabo_v3_points_{datetime.now().strftime(string_format)}.png"
fig.savefig(name)
print(name)

starttime = datetime.now()

fig = plot_front(results, "green", "orange", "red")
# Name to save the figure under.
name = f"figures/Port_Opt_Rabo_v3_front_{datetime.now().strftime(string_format)}.png"
fig.savefig(name)
print(name)
