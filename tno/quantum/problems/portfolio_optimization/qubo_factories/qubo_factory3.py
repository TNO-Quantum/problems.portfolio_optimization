from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .base_qubo_factory import BaseQUBOFactory

QUBOFactory3T = TypeVar("QUBOFactory3T", bound="QUBOFactory3")


class QUBOFactory3(BaseQUBOFactory):
    def calc_maximize_ROC(self):
        capital2021 = np.sum(self.capital)

        alpha = np.sum(self.LB * self.income / self.out2021)
        beta = np.array(
            [
                self.income
                / self.out2021
                * (self.UB - self.LB)
                / self.maxk
                * 2 ** (k + self.kmin)
                for k in range(self.kmax)
            ]
        ).T
        gamma = np.array(
            [
                (2 ** (-l - 1)) * (-1 + (2 ** (-l - 1)))
                for l in range(self.ancilla_qubits)
            ]
        )

        qubo = np.zeros((self.n_vars, self.n_vars))
        offset = alpha / capital2021
        for idx1 in range(self.N * self.kmax):
            i, k = divmod(idx1, self.kmax)
            qubo[idx1, idx1] = beta[i, k]
            for l in range(self.ancilla_qubits):
                idx2 = self.N * self.kmax + l
                qubo[idx1, idx2] = gamma[l] * beta[i, k]

        for l in range(self.ancilla_qubits):
            idx2 = self.N * self.kmax + l
            qubo[idx2, idx2] = gamma[l] * alpha

        qubo = qubo / capital2021

        return qubo, offset

    def calc_stabalize_c(self):

        alpha = np.sum(self.capital * self.LB / self.out2021) - np.sum(self.capital)
        beta = np.array(
            [
                self.capital
                / self.out2021
                * (self.UB - self.LB)
                / self.maxk
                * 2 ** (k + self.kmin)
                for k in range(self.kmax)
            ]
        ).T
        gamma = -np.array(
            [(2 ** (-n - 1)) for n in range(self.ancilla_qubits)]
        ) * np.sum(self.capital)

        qubo = np.zeros((self.n_vars, self.n_vars))
        offset = alpha**2
        for idx1 in range(self.N * self.kmax):
            i, k = divmod(idx1, self.kmax)
            qubo[idx1, idx1] = 2 * alpha * beta[i, k] + beta[i, k] ** 2
            for idx2 in range(idx1 + 1, self.N * self.kmax):
                j, l = divmod(idx2, self.kmax)
                qubo[idx1, idx2] = 2 * beta[i, k] * beta[j, l]
            for n in range(self.ancilla_qubits):
                idx2 = n + self.N * self.kmax
                qubo[idx1, idx2] = 2 * beta[i, k] * gamma[n]

        for n in range(self.ancilla_qubits):
            idx1 = n + self.N * self.kmax
            qubo[idx1, idx1] = gamma[n] ** 2 + 2 * alpha * gamma[n]
            for m in range(n + 1, self.ancilla_qubits):
                idx2 = m + self.N * self.kmax
                qubo[idx1, idx2] = 2 * gamma[n] * gamma[m]

        return qubo, offset

    def compile(self: QUBOFactory3T) -> QUBOFactory3T:
        self.minimize_HHI, _ = self.calc_minimize_HHI()
        self.stabilize_C, _ = self.calc_stabalize_c()
        self.maximize_R, _ = self.calc_maximize_ROC()
        self.emission, _ = self.calc_emission()
        return self

    def make_qubo(self, labda1: float, labda2: float, labda3: float, labda4: float):
        # Define Hamiltonian as a weighted sum of individual constraints
        qubo = (
            labda1 * self.minimize_HHI
            - labda2 * self.maximize_R
            + labda4 * self.stabilize_C
            + labda3 * self.emission
        )

        return qubo, float("nan")
