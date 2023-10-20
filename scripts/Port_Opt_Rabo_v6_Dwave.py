import datetime
import itertools

import matplotlib.pyplot as plt
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler  # Library to interact with the QPU
from minorminer import find_embedding
from pyqubo import Array, Constraint, Placeholder
from tqdm import tqdm

from tno.quantum.problems.portfolio_optimization.io import read_portfolio_data
from tno.quantum.problems.portfolio_optimization.pareto_front import pareto_front
from tno.quantum.problems.portfolio_optimization.qubo_factory import QUBOFactory

# Number of assets
N = 52

# Define the precision of the portfolio sizes.
kmax = 2  # 4#2 #number of values
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
solution_qubits = N * kmax
ancilla_qubits = 5
size_of_variable_array = solution_qubits + ancilla_qubits
var = Array.create("vector", size_of_variable_array, "BINARY")

# Defining constraints/HHI2030tives in the model
# HHI
qubo_factory = QUBOFactory(
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
minimize_HHI = qubo_factory.calc_minimize_HHI()

# ROC
cap_growth_fac = 1 + sum(
    var[k] * (2 ** (solution_qubits - k - 1))
    for k in range(solution_qubits, size_of_variable_array)
)
capital2021 = sum(capital[i] for i in range(N))
capital_target = cap_growth_fac * capital2021

reg_capital = 0
for i in range(N):
    reg_capital += (
        capital[i]
        * (
            LB[i]
            + (UB[i] - LB[i])
            * sum(2 ** (k + kmin) * var[i * kmax + k] for k in range(kmax))
            / maxk
        )
        / out2021[i]
    )
reg_capital += -1 * capital_target
stabilize_C = Constraint(reg_capital**2, label="stabilize_C")

maximize_R = qubo_factory.calc_maximize_ROC3(ancilla_qubits=ancilla_qubits)

# Emissions
emission = qubo_factory.calc_emission()


# These are the variables to store 3 kinds of results.
x1 = {}  # Emission target met
y1 = {}
z1 = {}
e1 = {}
# Counter variables for the numer of outcomes found.
res_ctr1 = 0
res_ctr2 = 0
res_ctr3 = 0

# Variables to combine the 3 HHI2030tives to optimize.
labda1 = Placeholder("labda1")
labda2 = Placeholder("labda2")
labda3 = Placeholder("labda3")
labda4 = Placeholder("labda4")

# Define Hamiltonian as a weighted sum of individual constraints
H = (
    labda1 * minimize_HHI
    - labda2 * maximize_R
    + labda4 * stabilize_C
    + labda3 * emission
)
model = H.compile()

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
    qubo, offset = model.to_qubo(
        feed_dict={"labda1": A, "labda2": C, "labda3": P, "labda4": Q}
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
        response = sampler.sample_qubo(qubo, num_sweeps=20, num_reads=10)

    # Postprocess solution.Iterate over all found solutions.
    dummy_ctr = 0
    for sample in response.samples():

        Energy = response.record[dummy_ctr][-2]
        dummy_ctr += 1
        # Compute the 2030 portfolio
        Out2030 = 0
        out2030 = {}
        for i in range(N):
            out2030[i] = (
                LB[i]
                + (UB[i] - LB[i])
                * sum(
                    (2 ** (k + kmin) * sample["vector[" + str(i * kmax + k) + "]"])
                    for k in range(kmax)
                )
                / maxk
            )
            if (LB[i] > out2030[i]) | (UB[i] < out2030[i]):
                print("Bounds not obeyed.\\", i, LB[i], out2030[i], UB[i])
                quit()
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
            z1[counter1][counter2][counter4].append(res_ctr1)
            e1[counter1][counter2][counter4] += Energy
            res_ctr1 += 1
        elif res_emis < norm2:
            res_ctr2 += 1
        else:
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
            legend.append(str(counter1) + "," + str(counter2) + "," + str(counter4))
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
            ax.scatter(list(X), list(Y))
ax.scatter(list(x_rabo1.values()), list(y_rabo1.values()), color="blue")
ax.scatter(list(x_rabo2.values()), list(y_rabo2.values()), color="gray")
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
x1, y1 = pareto_front(x1,y1, res_ctr1)
print("Time consumed:", datetime.datetime.now()-starttime)

#Make a plot of the results.
fig, ax = plt.subplots()
ax.scatter(list(x1.values()),list(y1.values()), color='limegreen')
ax.scatter(list(x_rabo1.values()),list(y_rabo1.values()), color='blue')
ax.scatter(list(x_rabo2.values()),list(y_rabo2.values()), color='gray')
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
