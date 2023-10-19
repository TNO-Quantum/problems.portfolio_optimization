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
