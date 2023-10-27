import itertools
from datetime import datetime

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import FixedEmbeddingComposite, LeapHybridSampler
from dwave.system.samplers import DWaveSampler  # Library to interact with the QPU
from minorminer import find_embedding
from tqdm import tqdm

from tno.quantum.problems.portfolio_optimization.containers import Results
from tno.quantum.problems.portfolio_optimization.io import (
    get_rabo_fronts,
    read_portfolio_data,
)
from tno.quantum.problems.portfolio_optimization.postprocess import Decoder
from tno.quantum.problems.portfolio_optimization.preprocessing import print_info
from tno.quantum.problems.portfolio_optimization.qubo_factories import QUBOFactory1
from tno.quantum.problems.portfolio_optimization.visualization import (
    plot_front,
    plot_points,
)

# Number of assets
N = 52

# Define the precision of the portfolio sizes.
kmax = 2  # 2 #number of values
kmin = 0  # minimal value 2**kmin

out2021, _, _, e, income, capital, df = read_portfolio_data("rabodata.xlsx")

# Compute the returns per outstanding amount in 2021.
returns = income / out2021

print_info(df)
Out2021 = np.sum(out2021)
ROC2021 = np.sum(income) / np.sum(capital)
HHI2021 = np.sum(out2021**2) / np.sum(out2021) ** 2
bigE = np.sum(e * out2021) / Out2021


Growth_target = 1.55
print("Growth target:", round(100.0 * (Growth_target - 1), 1), "%")


# Creating the actual model to optimize using the annealer.
print("Status: creating model")
# Initialize variable vector of the required size
size_of_variable_array = N * kmax

# Defining constraints/HHI2030tives in the model
# HHI
qubo_factory = QUBOFactory1(
    portfolio_data=df, n_vars=size_of_variable_array, kmin=kmin, kmax=kmax
)

# These are the variables to store 3 kinds of results.
results = Results(df, Growth_target)
parameters = []

qubo_factory.compile(Growth_target)

# Quantum computing options
useQPU = False  # true = QPU, false = SA
Option1 = False

# Algorithm variables
steps1 = 21
steps2 = 21
steps3 = 1
steps4 = 21

print("Status: calculating")
starttime = datetime.now()
decoder = Decoder(portfolio_data=df, kmin=kmin, kmax=kmax)

# Choose sampler and solve qubo. This is the actual optimization with either a DWave
# system or a simulated annealer.
if useQPU and Option1:
    sampler = LeapHybridSampler()
    sampler_kwargs = {}
elif useQPU:
    qubo, _ = qubo_factory.make_qubo(1, 1, 1, 1)
    solver = DWaveSampler()
    __, target_edgelist, target_adjacency = solver.structure
    emb = find_embedding(qubo, target_edgelist, verbose=1)
    sampler = FixedEmbeddingComposite(solver, emb)
    sampler_kwargs = {"num_reads": 20}
else:
    sampler = SimulatedAnnealingSampler()
    sampler_kwargs = {"num_reads": 20, "num_sweeps": 200}

# Set up penalty coefficients for the constraints
labdas1 = np.logspace(-16, 0, steps1, endpoint=False, base=10.0)
labdas2 = np.logspace(-16, 0, steps2, endpoint=False, base=10.0)
labdas3 = np.array([1])
labdas4 = np.logspace(-16, 0, steps4, endpoint=False, base=10.0)
total_steps = steps1 * steps2 * steps3 * steps4
labdas_iterator = tqdm(
    itertools.product(labdas1, labdas2, labdas3, labdas4), total=total_steps
)

for labdas in labdas_iterator:
    # Compile the model and generate QUBO
    qubo, offset = qubo_factory.make_qubo(*labdas)

    response = sampler.sample_qubo(qubo, **sampler_kwargs)

    # Postprocess solution.Iterate over all found solutions. (Compute 2030 portfolios)
    out2030 = decoder.decode_sampleset(response)
    results.add_result(out2030)


print(
    "Number of generated samples: ", len(results.x1), len(results.x2), len(results.x3)
)
print("Time consumed:", datetime.now() - starttime)

# Comparing with Rabobank's fronts.
# x/y_rabo1 corresponds to a front optimized including the emission target.
# x/y_rabo2 corresponds to a front optimized without the emission target.
x_rabo1, y_rabo1, x_rabo2, y_rabo2 = get_rabo_fronts()


# Make a plot of the results.
fig = plot_points(
    results, "coral", "gold", "dodgerblue", x_rabo1, y_rabo1, x_rabo2, y_rabo2
)
# Name to save the figure under.
string_format = r"%Y-%m-%d %H_%M_%S.%f"
name = f"figures/Port_Opt_Rabo_v4_GT{Growth_target}_points_"
name += f"{datetime.now().strftime(string_format)}.png"
fig.savefig(name)

fig = plot_front(
    results, "coral", "gold", "dodgerblue", x_rabo1, y_rabo1, x_rabo2, y_rabo2
)
# Name to save the figure under.
name = f"figures/Port_Opt_Rabo_v4_GT{Growth_target}_fronts_"
name += f"{datetime.now().strftime(string_format)}.png"
fig.savefig(name)
