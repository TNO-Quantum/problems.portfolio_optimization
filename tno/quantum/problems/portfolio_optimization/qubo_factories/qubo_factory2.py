import numpy as np
from numpy.typing import NDArray
from pyqubo import Constraint, Placeholder

from .base_qubo_factory import BaseQUBOFactory


class QUBOFactory2(BaseQUBOFactory):
    def calc_maximize_ROC(self, capital_growth_factor: float):
        capital_target = capital_growth_factor * np.sum(self.capital)

        qubo_diag = np.array(
            [
                (self.UB - self.LB)
                * self.income
                / (self.out2021 * self.maxk)
                * 2 ** (k + self.kmin)
                for k in range(self.kmax)
            ]
        ).flatten("F")
        qubo = np.diag(qubo_diag / capital_target)
        offset = np.sum(self.LB * self.income / self.out2021) / capital_target
        return qubo, offset

    def calc_stabelize_c(self, capital_growth_factor: float):
        capital_target = capital_growth_factor * np.sum(self.capital)
        alpha = np.sum(self.capital * self.LB / self.out2021) - capital_target
        beta = np.array(
            [
                self.capital
                * (self.UB - self.LB)
                / (self.maxk * self.out2021)
                * 2 ** (k + self.kmin)
                for k in range(self.kmax)
            ]
        ).T

        qubo = np.zeros((len(self.var), len(self.var)))
        offset = alpha**2
        for idx1 in range(self.N * self.kmax):
            i, k = divmod(idx1, self.kmax)
            qubo[idx1, idx1] = 2 * alpha * beta[i, k] + beta[i, k] ** 2
            for idx2 in range(idx1 + 1, self.N * self.kmax):
                j, l = divmod(idx2, self.kmax)
                qubo[idx1, idx2] = 2 * beta[i, k] * beta[j, l]

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
