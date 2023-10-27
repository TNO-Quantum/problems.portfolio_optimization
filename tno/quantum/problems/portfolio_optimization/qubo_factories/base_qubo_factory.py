from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

BaseQUBOFactoryT = TypeVar("BaseQUBOFactoryT", bound="BaseQUBOFactory")


class BaseQUBOFactory(ABC):
    def __init__(
        self, portfolio_data: DataFrame, kmin: int, kmax: int, ancilla_qubits: int = 0
    ) -> None:
        self.N = len(portfolio_data)
        self.n_vars = self.N * kmax + ancilla_qubits
        self.out2021 = portfolio_data["out_2021"].to_numpy()
        self.LB = portfolio_data["out_2030_min"].to_numpy()
        self.UB = portfolio_data["out_2030_max"].to_numpy()
        self.e = (portfolio_data["emis_intens_2021"].to_numpy() / 100).astype(float)
        self.income = portfolio_data["income_2021"].to_numpy()
        self.capital = portfolio_data["regcap_2021"].to_numpy()
        self.kmin = kmin
        self.kmax = kmax
        self.maxk = 2 ** (kmax + kmin) - 1 + (2 ** (-kmin) - 1) / (2 ** (-kmin))
        self.ancilla_qubits = ancilla_qubits

    def calc_minimize_HHI(self):
        Exp_total_out2030 = np.sum((self.UB + self.LB)) / 2

        qubo = np.zeros((self.n_vars, self.n_vars))
        offset = np.sum(self.LB**2) / Exp_total_out2030**2
        for i in range(self.N):
            multiplier = (self.UB[i] - self.LB[i]) / self.maxk
            for k in range(self.kmax):
                idx_1 = i * self.kmax + k
                value_1 = multiplier * 2 ** (k + self.kmin)
                qubo[idx_1, idx_1] = value_1 * (value_1 + 2 * self.LB[i])
                for l in range(k + 1, self.kmax):
                    idx_2 = i * self.kmax + l
                    value_2 = multiplier * 2 ** (l + self.kmin)
                    qubo[idx_1, idx_2] = 2 * value_1 * value_2

        qubo = qubo / Exp_total_out2030**2
        return qubo, offset

    @abstractmethod
    def calc_maximize_ROC(self):
        ...

    @abstractmethod
    def compile(self: BaseQUBOFactoryT) -> BaseQUBOFactoryT:
        ...

    def calc_emission(self):
        emis2021 = np.sum(self.e * self.out2021)
        bigE = emis2021 / np.sum(self.out2021)

        alpha = (0.76 * self.e - 0.7 * bigE) * self.LB
        tmp = (0.76 * self.e - 0.7 * bigE) * (self.UB - self.LB) / self.maxk
        beta = np.array([tmp * 2 ** (k + self.kmin) for k in range(self.kmax)]).T

        qubo = np.zeros((self.n_vars, self.n_vars))

        alpha_tot = np.sum(alpha)
        offset = alpha_tot**2 / emis2021**2
        for idx1 in range(self.N * self.kmax):
            i, k = divmod(idx1, self.kmax)
            qubo[idx1, idx1] += 2 * alpha_tot * beta[i, k] + beta[i, k] ** 2
            for idx2 in range(idx1 + 1, self.N * self.kmax):
                j, l = divmod(idx2, self.kmax)
                qubo[idx1, idx2] += 2 * beta[i, k] * beta[j, l]

        qubo = qubo / emis2021**2

        return qubo, offset
