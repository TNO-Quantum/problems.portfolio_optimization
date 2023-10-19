from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pyqubo import Array, Constraint


def calc_sumi(
    var: Array,
    N: int,
    LB: NDArray[np.float_],
    UB: NDArray[np.float_],
    kmin: int,
    kmax: int,
    maxk: int,
):
    sumi = 0
    for i in range(N):
        sumi += (
            LB[i]
            + (UB[i] - LB[i])
            * sum(2 ** (k + kmin) * var[i * kmax + k] for k in range(kmax))
            / maxk
        )
    return sumi


def calc_minimize_HHI(
    var: Array,
    N: int,
    LB: NDArray[np.float_],
    UB: NDArray[np.float_],
    kmin: int,
    kmax: int,
    maxk: int,
    Exp_total_out2030: float,
):
    minimize_HHI = Constraint(
        sum(
            (
                LB[i]
                + (UB[i] - LB[i])
                * sum(2 ** (k + kmin) * var[i * kmax + k] for k in range(kmax))
                / maxk
            )
            ** 2
            for i in range(N)
        )
        / (Exp_total_out2030**2),
        label="minimize_HHI",
    )
    return minimize_HHI


def calc_maximize_ROC(
    var: Array,
    N: int,
    out2021: NDArray[np.float_],
    LB: NDArray[np.float_],
    UB: NDArray[np.float_],
    income: NDArray[np.float_],
    capital: NDArray[np.float_],
    kmin: int,
    kmax: int,
    maxk: int,
    Exp_avr_growth_fac: float,
):
    maximize_ROC = Constraint(
        sum(
            (
                (
                    LB[i]
                    + (UB[i] - LB[i])
                    * sum(2 ** (k + kmin) * var[i * kmax + k] for k in range(kmax))
                    / maxk
                )
                * income[i]
                / (capital[i] * out2021[i] * Exp_avr_growth_fac)
            )
            for i in range(N)
        ),
        label="maximize_ROC",
    )
    return maximize_ROC


def calc_maximize_ROC2(
    var: Array,
    N: int,
    out2021: NDArray[np.float_],
    LB: NDArray[np.float_],
    UB: NDArray[np.float_],
    income: NDArray[np.float_],
    kmin: int,
    kmax: int,
    maxk: int,
    capital_target: float,
):
    max_R = 0
    for i in range(N):
        max_R += (
            income[i]
            * (
                LB[i]
                + (UB[i] - LB[i])
                * sum(2 ** (k + kmin) * var[i * kmax + k] for k in range(kmax))
                / maxk
            )
            / out2021[i]
        )
    maximize_R = Constraint(max_R / capital_target, label="maximize_R")
    return maximize_R


def calc_maximize_ROC3(
    var: Array,
    N: int,
    out2021: NDArray[np.float_],
    LB: NDArray[np.float_],
    UB: NDArray[np.float_],
    income: NDArray[np.float_],
    kmin: int,
    kmax: int,
    maxk: int,
    capital2021: float,
    ancilla_qubits: int,
):
    solution_qubits = N * kmax
    size_of_variable_array = solution_qubits + ancilla_qubits
    app_inv_cap_growth_fac = 1 + sum(
        var[k]
        * (2 ** (solution_qubits - k - 1))
        * (-1 + (2 ** (solution_qubits - k - 1)))
        for k in range(solution_qubits, size_of_variable_array)
    )

    max_R = 0
    for i in range(N):
        max_R += (
            income[i]
            * (
                LB[i]
                + (UB[i] - LB[i])
                * sum(2 ** (k + kmin) * var[i * kmax + k] for k in range(kmax))
                / maxk
            )
            / out2021[i]
        )
    maximize_R = Constraint(
        app_inv_cap_growth_fac * max_R / capital2021, label="maximize_R"
    )
    return maximize_R


def calc_maximize_ROC4(
    var: Array,
    N: int,
    out2021: NDArray[np.float_],
    LB: NDArray[np.float_],
    UB: NDArray[np.float_],
    capital: NDArray[np.float_],
    kmin: int,
    kmax: int,
    maxk: int,
    returns: dict[int, float],
):
    maximize_ROC = Constraint(
        sum(
            (
                (
                    LB[i]
                    + (UB[i] - LB[i])
                    * sum(2 ** (k + kmin) * var[i * kmax + k] for k in range(kmax))
                    / maxk
                )
                * returns[i]
            )
            for i in range(N)
        )
        / sum((((LB[i] + UB[i]) / (2.0 * out2021[i])) * capital[i]) for i in range(N)),
        label="maximize_ROC",
    )
    return maximize_ROC


def calc_emission(
    var: Array,
    N: int,
    LB: NDArray[np.float_],
    UB: NDArray[np.float_],
    e: NDArray[np.float_],
    kmin: int,
    kmax: int,
    maxk: int,
    emis2021: float,
    bigE: float,
    sumi,
):
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
