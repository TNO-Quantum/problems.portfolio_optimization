import numpy as np
from numpy.typing import NDArray
from pyqubo import Array


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
