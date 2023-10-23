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
        self.out2021 = portfolio_data["out_2021"].to_numpy()
        self.LB = portfolio_data["out_2030_min"].to_numpy()
        self.UB = portfolio_data["out_2030_max"].to_numpy()
        self.e = (portfolio_data["emis_intens_2021"].to_numpy() / 100).astype(float)
        self.income = portfolio_data["income_2021"].to_numpy()
        self.capital = portfolio_data["regcap_2021"].to_numpy()
        self.kmin = kmin
        self.kmax = kmax
        self.maxk = 2 ** (kmax + kmin) - 1 + (2 ** (-kmin) - 1) / (2 ** (-kmin))
        self.returns = self.income / self.out2021
        emis2021 = np.sum(self.e * self.out2021)
        self.bigE = emis2021 / np.sum(self.out2021)

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
        samples_matrix = sampleset.record.sample[:, : self.N * self.kmax]
        samples_reshaped = samples_matrix.reshape((len(sampleset), self.N, self.kmax))
        mantissa = np.power(2, np.arange(self.kmax) - self.kmin)
        ints = np.sum(samples_reshaped * mantissa, axis=2)
        out2030 = self.LB + (self.UB - self.LB) * ints / self.maxk

        if (self.LB > out2030).any() or (self.UB < out2030).any():
            raise ValueError("Bounds not obeyed.")

        return out2030
