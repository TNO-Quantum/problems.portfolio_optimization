from typing import Mapping

import numpy as np
from dimod import SampleSet
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

    def decode_sample(self, sample: Mapping[int, int]):
        # Compute the 2030 portfolio
        sample = np.array(
            [sample[i] for i in range(self.N * self.kmax)], dtype=np.uint8
        )
        mantissa = np.power(2, np.arange(self.kmax) - self.kmin)
        ints = np.sum(sample.reshape((self.N, self.kmax)) * mantissa, axis=1)
        out2030 = self.LB + (self.UB - self.LB) / self.maxk * ints
        if (self.LB > out2030).any() or (self.UB < out2030).any():
            raise ValueError("Bounds not obeyed.")

        return out2030

    def decode_sampleset(self, sampleset: SampleSet):
        # Compute the 2030 portfolio
        samples_matrix = sampleset.record.sample[: self.N * self.kmax]
        samples_reshaped = samples_matrix.reshape((len(sampleset), self.N, self.kmax))
        mantissa = np.power(2, np.arange(self.kmax) - self.kmin)
        ints = np.sum(samples_reshaped * mantissa, axis=2)
        out2030 = self.LB + (self.UB - self.LB) * ints / self.maxk

        if (self.LB > out2030).any() or (self.UB < out2030).any():
            raise ValueError("Bounds not obeyed.")

        return out2030
