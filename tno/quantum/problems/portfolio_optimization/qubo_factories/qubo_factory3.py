from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .base_qubo_factory import BaseQUBOFactory

QUBOFactory3T = TypeVar("QUBOFactory3T", bound="QUBOFactory3")


class QUBOFactory3(BaseQUBOFactory):
    def calc_maximize_ROC(self):
        capital2021 = np.sum(self.capital)

        alpha = np.sum(self.LB * self.income / self.out2021)
        mantisse = mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = self.income * (self.UB - self.LB) / (self.out2021 * self.maxk)
        beta = np.kron(multiplier, mantisse)

        gamma = np.power(2.0, np.arange(-1, -self.ancilla_qubits - 1, -1))
        gamma = gamma**2 - gamma

        qubo = np.zeros((self.n_vars, self.n_vars))
        np.fill_diagonal(qubo[: self.N * self.kmax, : self.N * self.kmax], beta)
        qubo[: self.N * self.kmax, self.N * self.kmax :] += np.outer(beta, gamma)
        np.fill_diagonal(
            qubo[self.N * self.kmax :, self.N * self.kmax :], alpha * gamma
        )
        offset = alpha / capital2021

        qubo = qubo / capital2021

        return qubo, offset

    def calc_stabalize_c(self):

        alpha = np.sum(self.capital * self.LB / self.out2021) - np.sum(self.capital)

        mantisse = mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = self.capital * (self.UB - self.LB) / (self.out2021 * self.maxk)
        beta = np.kron(multiplier, mantisse)

        gamma = -np.power(2.0, np.arange(-1, -self.ancilla_qubits - 1, -1)) * np.sum(
            self.capital
        )

        qubo = np.zeros((self.n_vars, self.n_vars))
        qubo_upper_left = qubo[: self.N * self.kmax, : self.N * self.kmax]
        qubo_upper_left += np.triu(2 * np.outer(beta, beta), k=1)
        np.fill_diagonal(qubo_upper_left, 2 * alpha * beta + beta**2)

        qubo_upper_right = qubo[: self.N * self.kmax, self.N * self.kmax :]
        qubo_upper_right += 2 * np.outer(beta, gamma)

        qubo_lower_right = qubo[self.N * self.kmax :, self.N * self.kmax :]
        qubo_lower_right += np.triu(2 * np.outer(gamma, gamma), k=1)
        np.fill_diagonal(qubo_lower_right, 2 * alpha * gamma + gamma**2)
        offset = alpha**2

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
