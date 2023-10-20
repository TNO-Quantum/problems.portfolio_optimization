from typing import Mapping

import numpy as np
from numpy.typing import NDArray


class Decoder:
    def __init__(
        self,
        N: int,
        out2021: NDArray[np.float_],
        LB: NDArray[np.float_],
        UB: NDArray[np.float_],
        e: NDArray[np.float_],
        income: NDArray[np.float_],
        capital: NDArray[np.float_],
        kmin: int,
        kmax: int,
        maxk: int,
    ) -> None:
        self.N = N
        self.out2021 = out2021
        self.LB = LB
        self.UB = UB
        self.e = e
        self.income = income
        self.capital = capital
        self.kmin = kmin
        self.kmax = kmax
        self.maxk = maxk
        self.returns = income / out2021
        emis2021 = np.sum(e * out2021)
        self.bigE = emis2021 / np.sum(out2021)

    def decode_sample(self, sample: Mapping[str, int]):
        # Compute the 2030 portfolio
        out2030 = np.zeros(self.N)
        for i in range(self.N):
            out2030[i] = (
                self.LB[i]
                + (self.UB[i] - self.LB[i])
                * sum(
                    (
                        2 ** (k + self.kmin)
                        * sample["vector[" + str(i * self.kmax + k) + "]"]
                    )
                    for k in range(self.kmax)
                )
                / self.maxk
            )
            if self.LB[i] > out2030[i] or self.UB[i] < out2030[i]:
                raise ValueError(
                    f"Bounds not obeyed. {i} {self.LB[i]} {self.out2030[i]} {self.UB[i]}"
                )

        return out2030
