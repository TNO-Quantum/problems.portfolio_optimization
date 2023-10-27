from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .base_qubo_factory import BaseQUBOFactory

QUBOFactory4T = TypeVar("QUBOFactory4T", bound="QUBOFactory4")


class QUBOFactory4(BaseQUBOFactory):
    def calc_maximize_ROC(self):
        returns = self.income / self.out2021
        multiplier = 2 / (np.sum((self.LB + self.UB) * self.capital / self.out2021))
        offset = np.sum(returns * self.LB) * multiplier
        qubo_diag = (
            np.array(
                [
                    returns * (self.UB - self.LB) / self.maxk * 2 ** (k + self.kmin)
                    for k in range(self.kmax)
                ]
            ).flatten("F")
            * multiplier
        )
        qubo = np.diag(qubo_diag)
        return qubo, offset

    def compile(self: QUBOFactory4T) -> QUBOFactory4T:
        self.minimize_HHI, _ = self.calc_minimize_HHI()
        self.maximize_ROC, _ = self.calc_maximize_ROC()
        self.emission, _ = self.calc_emission()
        return self

    def make_qubo(self, labda1: float, labda2: float, labda3: float):
        qubo = (
            labda1 * self.minimize_HHI
            - labda2 * self.maximize_ROC
            + labda3 * self.emission
        )
        return qubo, float("nan")
