import datetime
import itertools

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import FixedEmbeddingComposite, LeapHybridSampler
from dwave.system.samplers import DWaveSampler  # Library to interact with the QPU
from minorminer import find_embedding
from pyqubo import Array, Constraint, Placeholder
from tqdm import tqdm

from tno.quantum.problems.portfolio_optimization.io import (
    get_rabo_fronts,
    read_portfolio_data,
)
from tno.quantum.problems.portfolio_optimization.postprocess import Decoder
from tno.quantum.problems.portfolio_optimization.qubo_factory import QUBOFactory
from tno.quantum.problems.portfolio_optimization.visualization import (
    plot_front,
    plot_points,
)

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

Growth_target = 1.55
print("Growth target:", round(100.0 * (Growth_target - 1), 1), "%")

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

# Estimate a average growth factor and its standard deviation for 2021-2030. This consists of the (averaged) amount per asset in 2030, which is the outcome of the optimization, divided by the amount for 2021.
Exp_avr_growth_fac = Correctiefactor * sum(
    (UB[j] + LB[j]) / (2 * out2021[j]) for j in range(N)
)
Exp_stddev_avr_growth_fac = Correctiefactor * np.sqrt(
    sum((((UB[j] - LB[j]) / (2 * out2021[j])) ** 2) for j in range(N))
)
print(
    "Expected average growth factor: ",
    Exp_avr_growth_fac,
    "Std dev:",
    Exp_stddev_avr_growth_fac,
)


# Creating the actual model to optimize using the annealer.
print("Status: creating model")
# Initialize variable vector of the required size
size_of_variable_array = N * kmax
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
sumi = qubo_factory._calc_sumi()
minimize_HHI = qubo_factory.calc_minimize_HHI()

# ROC
maximize_ROC = qubo_factory.calc_maximize_ROC(Exp_avr_growth_fac=Exp_avr_growth_fac)

# Emissions
minimize_emission = qubo_factory.calc_emission()

# Growth condition
growth_factor = Constraint(
    ((sumi / Out2021) - Growth_target) ** 2, label="growth_factor"
)

# These are the variables to store 3 kinds of results.
x1, y1 = [], []  # Emission target met
x2, y2 = [], []  # Reduced emission
x3, y3 = [], []  # Targets not met
parameters = []

# Variables to combine the 3 HHI2030tives to optimize.
labda1 = Placeholder("labda1")
labda2 = Placeholder("labda2")
labda3 = Placeholder("labda3")
labda4 = Placeholder("labda4")


# Define Hamiltonian as a weighted sum of individual constraints
H = (
    labda1 * minimize_HHI
    - labda2 * maximize_ROC
    + labda3 * minimize_emission
    + labda4 * growth_factor
)
model = H.compile()

# Quantum computing options
eerste = True
useQPU = False  # true = QPU, false = SA
Option1 = False

# Algorithm variables
steps1 = 21
steps2 = 21
steps3 = 1
steps4 = 21

print("Status: calculating")
starttime = datetime.datetime.now()
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

for counter1, counter2, counter3, counter4 in tqdm(
    itertools.product(range(steps1), range(steps2), range(steps3), range(steps4)),
    total=steps1 * steps2 * steps3 * steps4,
):

    # Set up penalty coefficients for the constraints
    A = 10 ** (-16.0 + (16.0 / steps1) * counter1)
    C = 10 ** (-16.0 + (16.0 / steps2) * counter2)
    P = 1
    Q = 10 ** (-16.0 + (16.0 / steps4) * counter4)

    # Compile the model and generate QUBO
    qubo, offset = model.to_qubo(
        feed_dict={"labda1": A, "labda2": C, "labda3": P, "labda4": Q}
    )

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
        response = sampler.sample_qubo(qubo, num_sweeps=200, num_reads=20)

    # Postprocess solution.Iterate over all found solutions.
    for sample in response.samples():
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

        Realized_growth = Out2030 / Out2021

        # Compare the emission with norm1 and norm 2 and store the results accordingly.
        if res_emis < norm1:
            if Realized_growth > Growth_target:
                x1.append(100 * (1 - (HHI2030 / HHI2021)))
                y1.append(100 * (ROC / ROC2021 - 1))
                parameters.append([counter1, counter2, counter4].copy())
            elif Realized_growth > 0.98 * Growth_target:
                x2.append(100 * (1 - (HHI2030 / HHI2021)))
                y2.append(100 * (ROC / ROC2021 - 1))
            else:
                x3.append(100 * (1 - (HHI2030 / HHI2021)))
                y3.append(100 * (ROC / ROC2021 - 1))

print("Number of generated samples: ", len(x1), len(x2), len(x3))
print("Time consumed:", datetime.datetime.now() - starttime)

# Comparing with Rabobank's fronts.
# x/y_rabo1 corresponds to a front optimized including the emission target.
# x/y_rabo2 corresponds to a front optimized without the emission target.
x_rabo1, y_rabo1, x_rabo2, y_rabo2 = get_rabo_fronts()


# Make a plot of the results.
fig = fig = plot_points(
    x1,
    y1,
    "coral",
    x2,
    y2,
    "gold",
    x3,
    y3,
    "dodgerblue",
    x_rabo1,
    y_rabo1,
    x_rabo2,
    y_rabo2,
)
# Name to save the figure under.
name = (
    "figures/Port_Opt_Rabo_v4_GT"
    + str(Growth_target)
    + "_points_"
    + datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S.%f")
    + ".png"
)
fig.savefig(name)

fig = plot_front(
    x1,
    y1,
    "coral",
    x2,
    y2,
    "gold",
    x3,
    y3,
    "dodgerblue",
    x_rabo1,
    y_rabo1,
    x_rabo2,
    y_rabo2,
)
# Name to save the figure under.
name = (
    "figures/Port_Opt_Rabo_v4_GT"
    + str(Growth_target)
    + "_fronts_"
    + datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S.%f")
    + ".png"
)
fig.savefig(name)
