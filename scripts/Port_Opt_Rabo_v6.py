from datetime import datetime

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import FixedEmbeddingComposite, LeapHybridSampler
from dwave.system.samplers import DWaveSampler  # Library to interact with the QPU
from minorminer import find_embedding

from tno.quantum.problems.portfolio_optimization.io import read_portfolio_data
from tno.quantum.problems.portfolio_optimization.portfolio_optimizer import (
    PortfolioOptimizer,
)
from tno.quantum.problems.portfolio_optimization.preprocessing import print_info
from tno.quantum.problems.portfolio_optimization.qubo_factories import QUBOFactory3
from tno.quantum.problems.portfolio_optimization.visualization import (
    plot_front,
    plot_points,
)

# Quantum computing options
useQPU = False  # true = QPU, false = SA
Option1 = False

# Define the precision of the portfolio sizes.
kmax = 2  # 4#2 #number of values
kmin = 0  # minimal value 2**kmin\
ancilla_qubits = 5

# Algorithm variables
steps1 = 20
steps2 = 12
steps3 = 1
steps4 = 12

df = read_portfolio_data("rabodata.xlsx")
print_info(df)


# Creating the actual model to optimize using the annealer.
print("Status: creating model")
qubo_factory = QUBOFactory3(
    portfolio_data=df, kmin=kmin, kmax=kmax, ancilla_qubits=ancilla_qubits
).compile()

print("Status: calculating")
starttime = datetime.now()

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
labdas4 = np.array([1])
labdas3 = np.logspace(-11, -9.5, steps4, endpoint=False, base=10.0)


portfolio_optimizer = PortfolioOptimizer(
    df,
    kmin,
    kmax,
    qubo_factory,
    sampler,
    sampler_kwargs,
    labdas1,
    labdas2,
    labdas3,
    labdas4,
)
portfolio_optimizer.add_minimize_HHI()
portfolio_optimizer.add_maximize_ROC(formulation=3, ancilla_qubits=ancilla_qubits)
portfolio_optimizer.add_emission_constraint()
results = portfolio_optimizer.run()

print(
    "Number of generated samples: ", len(results.x1), len(results.x2), len(results.x3)
)
print("Time consumed:", datetime.now() - starttime)


# Make a plot of the results.
fig = plot_points(results, "mediumseagreen", "darkkhaki", "crimson")
# Name to save the figure under.
string_format = r"%Y-%m-%d %H_%M_%S.%f"
name = f"figures/Port_Opt_Rabo_v6_points_{datetime.now().strftime(string_format)}.png"
fig.savefig(name)
print(name)

fig = plot_front(results, "mediumseagreen", "darkkhaki", "crimson")
# Name to save the figure under.
name = f"figures/Port_Opt_Rabo_v6_front_{datetime.now().strftime(string_format)}.png"
fig.savefig(name)
print(name)
