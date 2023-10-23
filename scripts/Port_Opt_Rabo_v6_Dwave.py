import datetime
import itertools

import matplotlib.pyplot as plt
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
from tno.quantum.problems.portfolio_optimization.pareto_front import pareto_front
from tno.quantum.problems.portfolio_optimization.postprocess import Decoder
from tno.quantum.problems.portfolio_optimization.preprocessing import print_info
from tno.quantum.problems.portfolio_optimization.qubo_factories import QUBOFactory3

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
x1, y1 = [], []  # Emission target met
z1 = {}
e1 = {}
# Counter variables for the numer of outcomes found.
res_ctr2 = 0
res_ctr3 = 0

qubo_factory.compile(ancilla_qubits)

# Quantum computing options
useQPU = True  # true = QPU, false = SA
Option1 = False

# Algorithm variables
steps1 = 2
steps2 = 2
steps3 = 1
steps4 = 2

z1 = {
    i: {j: {k: [] for k in range(steps4)} for j in range(steps2)} for i in range(steps1)
}
e1 = {
    i: {j: {k: 0.0 for k in range(steps4)} for j in range(steps2)}
    for i in range(steps1)
}
decoder = Decoder(portfolio_data=df, kmin=kmin, kmax=kmax)
starttime = datetime.datetime.now()

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
    sampler_kwargs = {"num_reads": 20, "num_sweeps": 10}

for counter1, counter2, counter3, counter4 in tqdm(
    itertools.product(range(steps1), range(steps2), range(steps3), range(steps4)),
    total=steps1 * steps2 * steps3 * steps4,
):

    # Set up penalty coefficients for the constraints
    # Vary A
    A = 10.0 ** (-7.0 + (2.0 / steps1) * counter1)
    C = 10.0 ** (-8.0 + (2.0 / steps2) * counter2)
    P = 10.0**2
    Q = 10.0 ** (-11.0 + (2.0 / steps4) * counter4)

    # Compile the model and generate QUBO
    qubo, offset = qubo_factory.make_qubo(A, C, P, Q)
    response = sampler.sample_qubo(qubo, **sampler_kwargs)

    # Postprocess solution.Iterate over all found solutions.
    dummy_ctr = 0
    for sample in response.samples():

        Energy = response.record[dummy_ctr][-2]
        dummy_ctr += 1
        # Compute the 2030 portfolio
        out2030 = decoder.decode_sample(sample)
        Out2030 = np.sum(out2030)
        HHI2030 = out2030**2 / Out2030**2
        # Compute the 2030 ROC
        ROC = np.sum(out2030 * returns) / np.sum(out2030 * capital / out2021)
        # Compute the emissions from the resulting 2030 portfolio.
        res_emis = 0.76 * np.sum(e * out2030)
        norm1 = bigE * 0.70 * Out2030
        norm2 = 1.020 * norm1

        # Compare the emission with norm1 and norm 2 and store the results accordingly.
        if res_emis < norm1:
            x1.append(100 * (1 - (HHI2030 / HHI2021)))
            y1.append(100 * (ROC / ROC2021 - 1))
            z1[counter1][counter2][counter4].append(len(x1))
            e1[counter1][counter2][counter4] += Energy
        elif res_emis < norm2:
            res_ctr2 += 1
        else:
            res_ctr3 += 1

print("Number of generated samples: ", len(x1), res_ctr2, res_ctr3)
print("Time consumed:", datetime.datetime.now() - starttime)

# Comparing with Rabobank's fronts.
# x/y_rabo1 corresponds to a front optimized including the emission target.
# x/y_rabo2 corresponds to a front optimized without the emission target.
x_rabo1, y_rabo1, x_rabo2, y_rabo2 = get_rabo_fronts()
# Make a plot of the results.
fig, ax = plt.subplots()

legend = []
# ax.scatter(list(x1.values()),list(y1.values()), color='green')
for counter1 in range(steps1):
    for counter2 in range(steps2):
        for counter4 in range(steps4):
            X = []
            Y = []
            for ctr in z1[counter1][counter2][counter4]:
                X.append(x1[ctr])
                Y.append(y1[ctr])
            legend.append(f"{counter1},{counter2},{counter4}")
            print(
                counter1,
                counter2,
                counter4,
                len(z1[counter1][counter2][counter4]),
                (
                    e1[counter1][counter2][counter4]
                    / len(z1[counter1][counter2][counter4])
                    if (len(z1[counter1][counter2][counter4]) > 0)
                    else 0.0
                ),
            )
            ax.scatter(X, Y)
ax.scatter(x_rabo1, y_rabo1, color="blue")
ax.scatter(x_rabo2, y_rabo2, color="gray")
legend.append("cc")
legend.append("cu")
ax.legend(legend, loc="upper right")
"""
ax.scatter(list(x1.values()),list(y1.values()), color='limegreen')
ax.scatter(list(x_rabo1.values()),list(y_rabo1.values()), color='blue')
ax.scatter(list(x_rabo2.values()),list(y_rabo2.values()), color='gray')
ax.legend(['QUBO constrained', 'classical constrained', 'classical unconstrained'])
"""
ax.scatter(0, 0)
ax.set_xlabel("Diversification")
ax.set_ylabel("ROC")
plt.grid()
# Name to save the figure under.
name = (
    "figures/Port_Opt_Rabo_v6_Dwave_points_"
    + datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S.%f")
    + ".png"
)
# plt.savefig(name)
print(name)
plt.show()
"""
starttime = datetime.datetime.now()
x1, y1 = pareto_front(x1,y1)
print("Time consumed:", datetime.datetime.now()-starttime)

#Make a plot of the results.
fig, ax = plt.subplots()
ax.scatter(list(x1.values()),list(y1.values()), color='limegreen')
ax.scatter(x_rabo1,y_rabo1, color='blue')
ax.scatter(x_rabo2,y_rabo2, color='gray')
ax.legend(['QUBO constrained', 'classical constrained', 'classical unconstrained'])
ax.scatter(0,0)
ax.set_xlabel('Diversification')
ax.set_ylabel('ROC')
plt.grid()
#Name to save the figure under.
name = "figures/Port_Opt_Rabo_v6_Dwave_front_"+datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S.%f") +".png"
plt.savefig(name)
print(name)
#plt.show()
"""
