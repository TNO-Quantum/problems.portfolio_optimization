import itertools
from collections import deque
from datetime import datetime

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import FixedEmbeddingComposite, LeapHybridSampler
from dwave.system.samplers import DWaveSampler  # Library to interact with the QPU
from minorminer import find_embedding
from tqdm import tqdm

from tno.quantum.problems.portfolio_optimization.io import (
    get_rabo_fronts,
    read_portfolio_data,
)
from tno.quantum.problems.portfolio_optimization.postprocess import Decoder
from tno.quantum.problems.portfolio_optimization.preprocessing import print_info
from tno.quantum.problems.portfolio_optimization.qubo_factories import QUBOFactory3
from tno.quantum.problems.portfolio_optimization.visualization import (
    plot_front,
    plot_points,
)

# Number of assets
N = 52

# Define the precision of the portfolio sizes.
kmax = 2  # 4#2 #number of values
kmin = 0  # minimal value 2**kmin

out2021, _, _, e, income, capital, df = read_portfolio_data("rabodata.xlsx")

# Compute the returns per outstanding amount in 2021.
returns = income / out2021

print_info(df)
ROC2021 = np.sum(income) / np.sum(capital)
HHI2021 = np.sum(out2021**2) / np.sum(out2021) ** 2
bigE = np.sum(e * out2021) / np.sum(out2021)


# Creating the actual model to optimize using the annealer.
print("Status: creating model")
# Initialize variable vector of the required size
solution_qubits = N * kmax
ancilla_qubits = 5
size_of_variable_array = solution_qubits + ancilla_qubits

# Defining constraints/HHI2030tives in the model
# HHI
qubo_factory = QUBOFactory3(
    portfolio_data=df, n_vars=size_of_variable_array, kmin=kmin, kmax=kmax
)

# These are the variables to store 3 kinds of results.
x1, y1 = deque(), deque()  # Emission target met
x2, y2 = deque(), deque()  # Reduced emission
x3, y3 = deque(), deque()  # Targets not met

qubo_factory.compile(ancilla_qubits)

# Quantum computing options
useQPU = False  # true = QPU, false = SA
Option1 = False

# Algorithm variables
steps1 = 20
steps2 = 12
steps3 = 1
steps4 = 12

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
labdas1 = np.logspace(-4.25, -1.75, steps1, endpoint=False, base=10.0)
labdas2 = np.logspace(-4, -2.5, steps2, endpoint=False, base=10.0)
labdas3 = np.array([1])
labdas4 = np.logspace(-11, -9.5, steps4, endpoint=False, base=10.0)
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

    Out2030 = np.sum(out2030, axis=1)
    # Compute the 2030 HHI.
    HHI2030 = np.sum(out2030**2, axis=1) / Out2030**2
    # Compute the 2030 ROC
    ROC = np.sum(out2030 * returns, axis=1) / np.sum(
        out2030 * capital / out2021, axis=1
    )
    # Compute the emissions from the resulting 2030 portfolio.
    res_emis = 0.76 * np.sum(e * out2030, axis=1)
    x = 100 * (1 - (HHI2030 / HHI2021))
    y = 100 * (ROC / ROC2021 - 1)

    norm1 = bigE * 0.70 * Out2030
    norm2 = 1.020 * norm1

    x1_slice = res_emis < norm1
    x1.extend(x[x1_slice])
    y1.extend(y[x1_slice])

    x2_slice = (res_emis >= norm1) & (res_emis < norm2)
    x2.extend(x[x2_slice])
    y2.extend(y[x2_slice])

    x3_slice = res_emis >= norm2
    x3.extend(x[x3_slice])
    y3.extend(y[x3_slice])

print("Number of generated samples: ", len(x1), len(x2), len(x3))
print("Time consumed:", datetime.now() - starttime)

# Comparing with Rabobank's fronts.
# x/y_rabo1 corresponds to a front optimized including the emission target.
# x/y_rabo2 corresponds to a front optimized without the emission target.
x_rabo1, y_rabo1, x_rabo2, y_rabo2 = get_rabo_fronts()

# Make a plot of the results.
fig = plot_points(
    x1,
    y1,
    "mediumseagreen",
    x2,
    y2,
    "darkkhaki",
    x3,
    y3,
    "crimson",
    x_rabo1,
    y_rabo1,
    x_rabo2,
    y_rabo2,
)
# Name to save the figure under.
string_format = r"%Y-%m-%d %H_%M_%S.%f"
name = f"figures/Port_Opt_Rabo_v6_points_{datetime.now().strftime(string_format)}.png"
fig.savefig(name)
print(name)

fig = plot_front(
    x1,
    y1,
    "mediumseagreen",
    x2,
    y2,
    "darkkhaki",
    x3,
    y3,
    "crimson",
    x_rabo1,
    y_rabo1,
    x_rabo2,
    y_rabo2,
)
# Name to save the figure under.
name = f"figures/Port_Opt_Rabo_v6_front_{datetime.now().strftime(string_format)}.png"
fig.savefig(name)
print(name)
