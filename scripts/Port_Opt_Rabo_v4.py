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
Growth_target = 1.55

# Algorithm variables
steps1 = 21
steps2 = 21
steps3 = 1
steps4 = 21

df = read_portfolio_data("rabodata.xlsx")
print_info(df)
print("Growth target:", round(100.0 * (Growth_target - 1), 1), "%")


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
labdas3 = np.array([1])
labdas4 = np.logspace(-16, 0, steps4, endpoint=False, base=10.0)


portfolio_optimizer = PortfolioOptimizer(
    df, kmin, kmax, labdas1, labdas2, labdas3, labdas4, Growth_target
)
portfolio_optimizer.add_minimize_HHI()
portfolio_optimizer.add_maximize_ROC(formulation=1)
portfolio_optimizer.add_emission_constraint()
portfolio_optimizer.add_growth_factor_constraint(Growth_target)
results = portfolio_optimizer.run(sampler, sampler_kwargs)

print(
    f"Number of generated samples: ", len(results.x1), len(results.x2), len(results.x3)
)
print("Time consumed:", datetime.now() - starttime)


# Make a plot of the results.
fig = plot_points(results, "coral", "gold", "dodgerblue")
# Name to save the figure under.
string_format = r"%Y-%m-%d %H_%M_%S.%f"
name = f"figures/Port_Opt_Rabo_v4_GT{Growth_target}_points_"
name += f"{datetime.now().strftime(string_format)}.png"
fig.savefig(name)

fig = plot_front(results, "coral", "gold", "dodgerblue")
# Name to save the figure under.
name = f"figures/Port_Opt_Rabo_v4_GT{Growth_target}_fronts_"
name += f"{datetime.now().strftime(string_format)}.png"
fig.savefig(name)
