from typing import Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from .base_qubo_factory import BaseQUBOFactory

QUBOFactory1T = TypeVar("QUBOFactory1T", bound="QUBOFactory1")


class QUBOFactory1(BaseQUBOFactory):
    def calc_maximize_ROC(self):
        Exp_avr_growth_fac = 0.5 * np.sum((self.UB + self.LB) / self.out2021)
        returns = self.income / self.out2021
        offset = np.sum(self.LB / (self.capital * self.out2021 * Exp_avr_growth_fac))
        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = (
            (self.UB - self.LB)
            * returns
            / (self.maxk * self.capital * Exp_avr_growth_fac)
        )
        qubo_diag = np.kron(multiplier, mantisse)

        qubo = np.diag(qubo_diag)
        return qubo, offset

    def calc_growth_factor(self, Growth_target: float):
        out2021_tot = np.sum(self.out2021)
        alpha = np.sum(self.LB) / out2021_tot - Growth_target

        mantisse = np.power(2, np.arange(self.kmax) - self.kmin)
        multiplier = (self.UB - self.LB) / (self.maxk * out2021_tot)
        beta = np.kron(multiplier, mantisse)

        qubo = np.triu(2 * np.outer(beta, beta), k=1)
        np.fill_diagonal(qubo, beta**2 + 2 * alpha * beta)
        offset = alpha**2
        return qubo, offset

    def compile(
        self: QUBOFactory1T, Growth_target: Optional[float] = None
    ) -> QUBOFactory1T:
        self.minimize_HHI, _ = self.calc_minimize_HHI()
        self.maximize_ROC, _ = self.calc_maximize_ROC()
        self.emission, _ = self.calc_emission()
        if Growth_target is not None:
            self.growth_factor, _ = self.calc_growth_factor(Growth_target)
        return self

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
