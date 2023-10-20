from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pyqubo import Constraint, Placeholder

from .base_qubo_factory import BaseQUBOFactory


class QUBOFactory1(BaseQUBOFactory):
    def calc_maximize_ROC(self):
        Exp_avr_growth_fac = np.sum((self.UB + self.LB) / (2 * self.out2021))
        maximize_ROC = Constraint(
            sum(
                (
                    (
                        self.LB[i]
                        + (self.UB[i] - self.LB[i])
                        * sum(
                            2 ** (k + self.kmin) * self.var[i * self.kmax + k]
                            for k in range(self.kmax)
                        )
                        / self.maxk
                    )
                    * self.income[i]
                    / (self.capital[i] * self.out2021[i] * Exp_avr_growth_fac)
                )
                for i in range(self.N)
            ),
            label="maximize_ROC",
        )
        return maximize_ROC

    def calc_growth_factor(self, Growth_target: float):
        sumi = self._calc_sumi()
        Out2021 = np.sum(self.out2021)
        growth_factor = Constraint(
            ((sumi / Out2021) - Growth_target) ** 2, label="growth_factor"
        )
        return growth_factor

    def compile(self, Growth_target: Optional[float] = None):
        minimize_HHI = self.calc_minimize_HHI()
        maximize_ROC = self.calc_maximize_ROC()
        emission = self.calc_emission()

        # Variables to combine the objectives to optimize.
        labda1 = Placeholder("labda1")
        labda2 = Placeholder("labda2")
        labda3 = Placeholder("labda3")

        # Define Hamiltonian as a weighted sum of individual constraints
        H = labda1 * minimize_HHI - labda2 * maximize_ROC + labda3 * emission

        if Growth_target is not None:
            growth_factor = self.calc_growth_factor(Growth_target)
            labda4 = Placeholder("labda4")
            H += labda4 * growth_factor
        self.model = H.compile()

    def make_qubo(
        self,
        labda1: float,
        labda2: float,
        labda3: float,
        labda4: Optional[float] = None,
    ):
        feed_dict = {"labda1": labda1, "labda2": labda2, "labda3": labda3}
        if labda4 is not None:
            feed_dict["labda4"] = labda4
        return self.model.to_qubo(feed_dict=feed_dict)
