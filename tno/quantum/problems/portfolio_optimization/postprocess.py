from typing import Mapping

import numpy as np
from dimod import SampleSet
from numpy.typing import NDArray
from pandas import DataFrame


class Decoder:
    def __init__(
        self,
        portfolio_data: DataFrame,
        kmin: int,
        kmax: int,
    ) -> None:
        self.N = len(portfolio_data)

        self.kmax = kmax
        maxk = 2 ** (kmax + kmin) - 1 + (2 ** (-kmin) - 1) / (2 ** (-kmin))
        self.mantissa = np.power(2, np.arange(kmax) - kmin)

        self.LB = portfolio_data["out_2030_min"].to_numpy()
        self.UB = portfolio_data["out_2030_max"].to_numpy()
        self.multiplier = (self.UB - self.LB) / maxk

    def decode_sample(self, sample: Mapping[int, int]) -> NDArray[np.float_]:
        # Compute the 2030 portfolio
        sample = np.array(
            [sample[i] for i in range(self.N * self.kmax)], dtype=np.uint8
        )
        sample_reshaped = sample.reshape((self.N, self.kmax))
        ints = np.sum(sample_reshaped * self.mantissa, axis=1)
        out2030 = self.LB + self.multiplier * ints
        if (self.LB > out2030).any() or (self.UB < out2030).any():
            raise ValueError("Bounds not obeyed.")

        return out2030

    def decode_sampleset(self, sampleset: SampleSet) -> NDArray[np.float_]:
        # Compute the 2030 portfolio
        samples_matrix = sampleset.record.sample[:, : self.N * self.kmax]
        samples_reshaped = samples_matrix.reshape((len(sampleset), self.N, self.kmax))

        ints = np.sum(samples_reshaped * self.mantissa, axis=2)
        out2030 = self.LB + self.multiplier * ints

        if (self.LB > out2030).any() or (self.UB < out2030).any():
            raise ValueError("Bounds not obeyed.")

        return out2030
