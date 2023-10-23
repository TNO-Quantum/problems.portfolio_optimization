from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .base_qubo_factory import BaseQUBOFactory


class QUBOFactory1(BaseQUBOFactory):
    def calc_maximize_ROC(self):
        Exp_avr_growth_fac = np.sum((self.UB + self.LB) / (2 * self.out2021))
        qubo = np.zeros((self.n_vars, self.n_vars))
        offset = np.sum(self.LB)
        for i in range(self.N):
            for k in range(self.kmax):
                idx = i * self.kmax + k
                qubo[idx, idx] += (
                    (self.UB[i] - self.LB[i])
                    * 2 ** (k + self.kmin)
                    / self.maxk
                    * self.income[i]
                    / (self.capital[i] * self.out2021[i] * Exp_avr_growth_fac)
                )

        return qubo, offset

    def calc_growth_factor(self, Growth_target: float):
        out2021_tot = np.sum(self.out2021)
        alpha = np.sum(self.LB) / out2021_tot - Growth_target
        beta = np.array(
            [
                (self.UB - self.LB) / (self.maxk * out2021_tot) * 2 ** (k + self.kmin)
                for k in range(self.kmax)
            ]
        ).T

        qubo = np.zeros((self.n_vars, self.n_vars))
        offset = alpha**2
        for idx1 in range(self.N * self.kmax):
            i, k = divmod(idx1, self.kmax)
            qubo[idx1, idx1] = 2 * alpha * beta[i, k] + beta[i, k] ** 2
            for idx2 in range(idx1 + 1, self.N * self.kmax):
                j, l = divmod(idx2, self.kmax)
                qubo[idx1, idx2] += 2 * beta[i, k] * beta[j, l]

        return qubo, offset

    def compile(self, Growth_target: Optional[float] = None):
        self.minimize_HHI, _ = self.calc_minimize_HHI()
        self.maximize_ROC, _ = self.calc_maximize_ROC()
        self.emission, _ = self.calc_emission()
        if Growth_target is not None:
            self.growth_factor, _ = self.calc_growth_factor(Growth_target)

    def make_qubo(
        self,
        labda1: float,
        labda2: float,
        labda3: float,
        labda4: Optional[float] = None,
    ):
        qubo = (
            labda1 * self.minimize_HHI
            - labda2 * self.maximize_ROC
            + labda3 * self.emission
        )
        if labda4 is not None:
            qubo += labda4 * self.growth_factor
        return qubo, float("nan")
