import itertools
from datetime import datetime

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import FixedEmbeddingComposite, LeapHybridSampler
from dwave.system.samplers import DWaveSampler  # Library to interact with the QPU
from minorminer import find_embedding
from tqdm import tqdm

from tno.quantum.problems.portfolio_optimization.containers import Results
from tno.quantum.problems.portfolio_optimization.io import read_portfolio_data
from tno.quantum.problems.portfolio_optimization.postprocess import Decoder
from tno.quantum.problems.portfolio_optimization.preprocessing import print_info
from tno.quantum.problems.portfolio_optimization.qubo_factories import QUBOFactory2
from tno.quantum.problems.portfolio_optimization.visualization import (
    plot_front,
    plot_points,
)

# Quantum computing options
useQPU = False  # true = QPU, false = SA
Option1 = False

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
qubo_factory = QUBOFactory2(portfolio_data=df, kmin=kmin, kmax=kmax).compile(
    capital_growth_factor
)

results = Results(df)

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
