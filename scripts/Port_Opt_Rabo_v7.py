import datetime

import matplotlib.pyplot as plt
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler  # Library to interact with the QPU
from minorminer import find_embedding
from pyqubo import Array, Constraint, Placeholder

import tno.quantum.problems.portfolio_optimization.qubo_factory as qubo_factory
from tno.quantum.problems.portfolio_optimization.io import read_portfolio_data
from tno.quantum.problems.portfolio_optimization.pareto_front import pareto_front

# Number of assets
N = 52

# Define the precision of the portfolio sizes.
kmax = 2  # 2 #number of values
kmin = 0  # minimal value 2**kmin
maxk = 2 ** (kmax + kmin) - 1 + (2 ** (-kmin) - 1) / (2 ** (-kmin))

out2021, LB, UB, e, income, capital = read_portfolio_data("rabodata.xlsx")

# Compute the returns per outstanding amount in 2021.
returns = {}
for i in range(N):
    returns[i] = income[i] / out2021[i]

# Calculate the total outstanding amount in 2021
Out2021 = sum(out2021[i] for i in range(N))
print("Total outstanding 2021: ", Out2021)
# Calculate the ROC for 2021
ROC2021 = sum(income[i] for i in range(N)) / sum(capital[i] for i in range(N))
print("ROC 2021: ", ROC2021)
# Calculate the HHI diversification for 2021
HHI2021 = sum(out2021[j] ** 2 for j in range(N)) / (
    sum(out2021[j] for j in range(N)) ** 2
)
print("HHI 2021: ", HHI2021)
# Calculate the total emissions for 2021
emis2021 = sum(e[i] * out2021[i] for i in range(N))
print("Emission 2021: ", emis2021)
# Calculate the average emission intensity 2021
bigE = emis2021 / Out2021
print("Emission intensity 2021:", bigE)

# Estimate the total outstanding amount and its standard deviation for 2030. This follows from the assumption of a symmetric probability distribution on the interval [LB,UB] and the central limit theorem. A correction factor can be used to tweak the results.
Correctiefactor = 1.00
Exp_total_out2030 = Correctiefactor * sum((UB[j] + LB[j]) / 2 for j in range(N))
Exp_stddev_total_out2030 = Correctiefactor * np.sqrt(
    sum((((UB[j] - LB[j]) / 2) ** 2) for j in range(N))
)
print(
    "Expected total outstanding 2030: ",
    Exp_total_out2030,
    "Std dev:",
    Exp_stddev_total_out2030,
)

# Creating the actual model to optimize using the annealer.
print("Status: creating model")
# Initialize variable vector of the required size
size_of_variable_array = N * kmax
var = Array.create("vector", size_of_variable_array, "BINARY")

# Defining constraints/HHI2030tives in the model
# HHI
sumi = qubo_factory.calc_sumi(
    var=var, N=N, LB=LB, UB=UB, kmin=kmin, kmax=kmax, maxk=maxk
)
minimize_HHI = qubo_factory.calc_minimize_HHI(
    var=var,
    N=N,
    LB=LB,
    UB=UB,
    kmin=kmin,
    kmax=kmax,
    maxk=maxk,
    Exp_total_out2030=Exp_total_out2030,
)


# ROC
maximize_ROC = qubo_factory.calc_maximize_ROC4(
    var=var,
    N=N,
    out2021=out2021,
    LB=LB,
    UB=UB,
    capital=capital,
    kmin=kmin,
    kmax=kmax,
    maxk=maxk,
    returns=returns,
)

# Emissions
emission_model = 0
for i in range(N):
    emission_model += (
        0.76
        * e[i]
        * (
            LB[i]
            + (UB[i] - LB[i])
            * sum(2 ** (k + kmin) * var[i * kmax + k] for k in range(kmax))
            / maxk
        )
        / emis2021
    )
emission_model += (-0.7 * bigE * sumi) / emis2021
emission = Constraint(emission_model**2, label="minimize_emission")

# These are the variables to store 3 kinds of results.
x1 = {}  # Emission target met
y1 = {}
x2 = {}  # Reduced emission
y2 = {}
x3 = {}  # Targets not met
y3 = {}
# Counter variables for the numer of outcomes found.
res_ctr1 = 0
res_ctr2 = 0
res_ctr3 = 0

# Variables to combine the 3 HHI2030tives to optimize.
labda1 = Placeholder("labda1")
labda2 = Placeholder("labda2")
labda3 = Placeholder("labda3")

# Define Hamiltonian as a weighted sum of individual constraints
H = labda1 * minimize_HHI - labda2 * maximize_ROC + labda3 * emission
model = H.compile()

# Quantum computing options
eerste = True
useQPU = False  # true = QPU, false = SA
Option1 = False

# Algorithm variables
steps1 = 99
steps2 = 99
steps3 = 1

print("Status: calculating")
starttime = datetime.datetime.now()
percent = 0
for counter1 in range(steps1):
    for counter2 in range(steps2):
        for counter3 in range(steps3):

            # Progress estimation
            if (100.0 * (steps2 * counter1 + counter2) * steps3 + counter3 + 1) / (
                steps1 * steps2 * steps3
            ) >= percent + 5:
                percent += 5
                print("Iterations:", percent, "%")

            # Set up penalty coefficients for the constraints
            A = 10 ** (-17.0 + (18.0 / steps1) * counter1)
            C = 10 ** (-17.0 + (18.0 / steps2) * counter2)
            P = 1

            # Compile the model and generate QUBO
            qubo, offset = model.to_qubo(
                feed_dict={"labda1": A, "labda2": C, "labda3": P}
            )

            # Choose sampler and solve qubo. This is the actual optimization with either a DWave system or a simulated annealer.
            if useQPU:
                if Option1:
                    from dwave.system import LeapHybridSampler

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
                response = sampler.sample_qubo(qubo, num_sweeps=400, num_reads=80)

            # Postprocess solution.Iterate over all found solutions.
            for sample in response.samples():
                # Compute the 2030 portfolio
                Out2030 = 0
                out2030 = {}
                for i in range(N):
                    out2030[i] = (
                        LB[i]
                        + (UB[i] - LB[i])
                        * sum(
                            (
                                2 ** (k + kmin)
                                * sample["vector[" + str(i * kmax + k) + "]"]
                            )
                            for k in range(kmax)
                        )
                        / maxk
                    )
                    Out2030 += out2030[i]
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
                    x1[res_ctr1] = 100 * (1 - (HHI2030 / HHI2021))
                    y1[res_ctr1] = 100 * (ROC / ROC2021 - 1)
                    res_ctr1 += 1
                elif res_emis < norm2:
                    x2[res_ctr2] = 100 * (1 - (HHI2030 / HHI2021))
                    y2[res_ctr2] = 100 * (ROC / ROC2021 - 1)
                    res_ctr2 += 1
                else:
                    x3[res_ctr3] = 100 * (1 - (HHI2030 / HHI2021))
                    y3[res_ctr3] = 100 * (ROC / ROC2021 - 1)
                    res_ctr3 += 1

print("Number of generated samples: ", res_ctr1, res_ctr2, res_ctr3)
print("Time consumed:", datetime.datetime.now() - starttime)

# Comparing with Rabobank's fronts.
# x/y_rabo1 corresponds to a front optimized including the emission target.
x_rabo1 = {}
y_rabo1 = {}
if 1 == 1:
    x_rabo1[0] = 3.44
    y_rabo1[0] = 0.0

    x_rabo1[1] = 3.66
    y_rabo1[1] = 0.5

    x_rabo1[2] = 3.84
    y_rabo1[2] = 1.0

    x_rabo1[3] = 3.96
    y_rabo1[3] = 1.5

    x_rabo1[4] = 4.01
    y_rabo1[4] = 2.0

    x_rabo1[5] = 3.99
    y_rabo1[5] = 2.5

    x_rabo1[6] = 3.94
    y_rabo1[6] = 3.0

    x_rabo1[7] = 3.83
    y_rabo1[7] = 3.5

    x_rabo1[8] = 3.62
    y_rabo1[8] = 4.0

    x_rabo1[9] = 3.31
    y_rabo1[9] = 4.5

    x_rabo1[10] = 2.90
    y_rabo1[10] = 5.0

    x_rabo1[11] = 0.40
    y_rabo1[11] = 5.5

    x_rabo1[12] = -3.53
    y_rabo1[12] = 6.0

    x_rabo1[13] = 0.90
    y_rabo1[13] = -2.0

    x_rabo1[14] = 2.02
    y_rabo1[14] = -1.5

    x_rabo1[15] = 2.73
    y_rabo1[15] = -1.0

    x_rabo1[16] = 3.14
    y_rabo1[16] = -0.5

# x/y_rabo1 corresponds to a front optimized without the emission target.
x_rabo2 = {}
y_rabo2 = {}
if 1 == 1:
    x_rabo2[0] = 3.275
    y_rabo2[0] = 8

    x_rabo2[1] = 3.634
    y_rabo2[1] = 7.5

    x_rabo2[2] = 3.89
    y_rabo2[2] = 7

    x_rabo2[3] = 4.293
    y_rabo2[3] = 6.5

    x_rabo2[4] = 4.447
    y_rabo2[4] = 6.0

    x_rabo2[5] = 4.753
    y_rabo2[5] = 5.5

    x_rabo2[6] = 4.897
    y_rabo2[6] = 5.0

    x_rabo2[7] = 5.034
    y_rabo2[7] = 4.5

    x_rabo2[8] = 5.148
    y_rabo2[8] = 4.0

    x_rabo2[9] = 5.149
    y_rabo2[9] = 3.5

    x_rabo2[10] = 5.198
    y_rabo2[10] = 3.0

    x_rabo2[11] = 5.179
    y_rabo2[11] = 2.5

# Make a plot of the results.
fig, ax = plt.subplots()
ax.scatter(list(x3.values()), list(y3.values()), color="crimson")
ax.scatter(list(x2.values()), list(y2.values()), color="mediumorchid")
ax.scatter(list(x1.values()), list(y1.values()), color="mediumblue")
ax.scatter(list(x_rabo1.values()), list(y_rabo1.values()), color="blue")
ax.scatter(list(x_rabo2.values()), list(y_rabo2.values()), color="gray")
ax.legend(
    [
        "QUBO constraint not met",
        "QUBO reduced",
        "QUBO constraint met",
        "classical constrained",
        "classical unconstrained",
    ]
)
ax.scatter(0, 0)
ax.set_xlabel("Diversification")
ax.set_ylabel("ROC")
plt.grid()
# Name to save the figure under.
name = (
    "figures/Port_Opt_Rabo_v7_points_"
    + datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S.%f")
    + ".png"
)
plt.savefig(name)
print(name)
# plt.show()

starttime = datetime.datetime.now()
x1, y1 = pareto_front(x1, y1, res_ctr1)
x2, y2 = pareto_front(x2, y2, res_ctr2)
x3, y3 = pareto_front(x3, y3, res_ctr3)
print("Time consumed:", datetime.datetime.now() - starttime)

# Make a plot of the results.
fig, ax = plt.subplots()
ax.scatter(list(x3.values()), list(y3.values()), color="crimson")
ax.scatter(list(x2.values()), list(y2.values()), color="mediumorchid")
ax.scatter(list(x1.values()), list(y1.values()), color="mediumblue")
ax.scatter(list(x_rabo1.values()), list(y_rabo1.values()), color="blue")
ax.scatter(list(x_rabo2.values()), list(y_rabo2.values()), color="gray")
ax.legend(
    [
        "QUBO constraint not met",
        "QUBO reduced",
        "QUBO constraint met",
        "classical constrained",
        "classical unconstrained",
    ]
)
ax.scatter(0, 0)
ax.set_xlabel("Diversification")
ax.set_ylabel("ROC")
plt.grid()
# Name to save the figure under.
name = (
    "figures/Port_Opt_Rabo_v7_front_"
    + datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S.%f")
    + ".png"
)
plt.savefig(name)
print(name)
# plt.show()
