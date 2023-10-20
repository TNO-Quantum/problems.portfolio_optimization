import datetime
import itertools

import matplotlib.pyplot as plt
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import FixedEmbeddingComposite, LeapHybridSampler
from dwave.system.samplers import DWaveSampler  # Library to interact with the QPU
from minorminer import find_embedding
from pyqubo import Array
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
maxk = 2 ** (kmax + kmin) - 1 + (2 ** (-kmin) - 1) / (2 ** (-kmin))

out2021, LB, UB, e, income, capital = read_portfolio_data("rabodata.xlsx")

# Compute the returns per outstanding amount in 2021.
returns = income / out2021

print_info(out2021, LB, UB, e, income, capital)
Out2021 = np.sum(out2021)
ROC2021 = np.sum(income) / np.sum(capital)
HHI2021 = np.sum(out2021**2) / np.sum(out2021) ** 2
emis2021 = np.sum(e * out2021)
bigE = emis2021 / Out2021


# Creating the actual model to optimize using the annealer.
print("Status: creating model")
# Initialize variable vector of the required size
solution_qubits = N * kmax
ancilla_qubits = 5
size_of_variable_array = solution_qubits + ancilla_qubits
var = Array.create("vector", size_of_variable_array, "BINARY")

# Defining constraints/HHI2030tives in the model
# HHI
qubo_factory = QUBOFactory3(
    var=var,
    N=N,
    out2021=out2021,
    LB=LB,
    UB=UB,
    e=e,
    income=income,
    capital=capital,
    kmin=kmin,
    kmax=kmax,
    maxk=maxk,
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
eerste = True
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
decoder = Decoder(
    N=N,
    out2021=out2021,
    LB=LB,
    UB=UB,
    e=e,
    income=income,
    capital=capital,
    kmin=kmin,
    kmax=kmax,
    maxk=maxk,
)
starttime = datetime.datetime.now()
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

    # Choose sampler and solve qubo. This is the actual optimization with either a DWave system or a simulated annealer.
    if useQPU:
        if Option1:
            sampler = LeapHybridSampler()
            response = sampler.sample_qubo(qubo)
        else:
            if eerste:
                solver = DWaveSampler()
                __, target_edgelist, target_adjacency = solver.structure
                emb = find_embedding(qubo, target_edgelist, verbose=1)
                eerste = False
            sampler = FixedEmbeddingComposite(solver, emb)
            response = sampler.sample_qubo(qubo, num_reads=20)
    else:
        sampler = SimulatedAnnealingSampler()
        response = sampler.sample_qubo(qubo, num_sweeps=20, num_reads=10)

    # Postprocess solution.Iterate over all found solutions.
    dummy_ctr = 0
    for sample in response.samples():

        Energy = response.record[dummy_ctr][-2]
        dummy_ctr += 1
        # Compute the 2030 portfolio
        out2030 = decoder.decode_sample(sample)
        Out2030 = sum(out2030[i] for i in range(N))
        # Compute the 2030 HHI.
        HHI2030 = 0
        for i in range(N):
            HHI2030 += out2030[i] ** 2
        HHI2030 = HHI2030 / (Out2030**2)
        # Compute the 2030 ROC
        ROC = sum(out2030[i] * returns[i] for i in range(N)) / sum(
            out2030[i] * capital[i] / out2021[i] for i in range(N)
        )
        # Compute the emissions from the resulting 2030 portfolio.
        res_emis = 0
        res_emis = 0.76 * sum(e[i] * out2030[i] for i in range(N))
        norm1 = bigE * 0.70 * Out2030  # Out2021
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
