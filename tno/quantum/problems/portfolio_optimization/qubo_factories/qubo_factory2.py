import numpy as np
from numpy.typing import NDArray

from .base_qubo_factory import BaseQUBOFactory


class QUBOFactory2(BaseQUBOFactory):
    def calc_maximize_ROC(self, capital_growth_factor: float):
        capital_target = capital_growth_factor * np.sum(self.capital)

        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = (self.UB - self.LB) * self.income / (self.out2021 * self.maxk)
        qubo_diag = np.kron(multiplier, mantisse) / capital_target
        qubo = np.diag(qubo_diag)
        offset = np.sum(self.LB * self.income / self.out2021) / capital_target
        return qubo, offset

    def calc_stabelize_c(self, capital_growth_factor: float):
        capital_target = capital_growth_factor * np.sum(self.capital)
        alpha = np.sum(self.capital * self.LB / self.out2021) - capital_target

        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = self.capital * (self.UB - self.LB) / (self.out2021 * self.maxk)
        beta = np.kron(multiplier, mantisse)

        qubo = np.triu(2 * np.outer(beta, beta), k=1)
        np.fill_diagonal(qubo, beta**2 + 2 * alpha * beta)
        offset = alpha**2
        return qubo, offset

    def compile(self, capital_growth_factor):
        self.minimize_HHI, _ = self.calc_minimize_HHI()
        self.maximize_ROC, _ = self.calc_maximize_ROC(capital_growth_factor)
        self.emission, _ = self.calc_emission()
        self.stabilize_C, _ = self.calc_stabelize_c(capital_growth_factor)

    def make_qubo(self, labda1: float, labda2: float, labda3: float, labda4: float):
        qubo = (
            labda1 * self.minimize_HHI
            - labda2 * self.maximize_ROC
            + labda3 * self.emission
            + labda4 * self.stabilize_C
        )
        return qubo, float("nan")
