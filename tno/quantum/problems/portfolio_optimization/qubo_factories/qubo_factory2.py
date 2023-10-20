import numpy as np
from numpy.typing import NDArray
from pyqubo import Constraint, Placeholder

from .base_qubo_factory import BaseQUBOFactory


class QUBOFactory2(BaseQUBOFactory):
    def calc_maximize_ROC(self, capital_growth_factor: float):
        capital_target = capital_growth_factor * np.sum(self.capital)
        max_R = 0
        for i in range(self.N):
            max_R += (
                self.income[i]
                * (
                    self.LB[i]
                    + (self.UB[i] - self.LB[i])
                    * sum(
                        2 ** (k + self.kmin) * self.var[i * self.kmax + k]
                        for k in range(self.kmax)
                    )
                    / self.maxk
                )
                / self.out2021[i]
            )
        maximize_R = Constraint(max_R / capital_target, label="maximize_R")
        return maximize_R

    def calc_stabelize_c(self, capital_growth_factor: float):
        capital_target = capital_growth_factor * np.sum(self.capital)
        reg_capital = 0
        for i in range(self.N):
            reg_capital += (
                self.capital[i]
                * (
                    self.LB[i]
                    + (self.UB[i] - self.LB[i])
                    * sum(
                        2 ** (k + self.kmin) * self.var[i * self.kmax + k]
                        for k in range(self.kmax)
                    )
                    / self.maxk
                )
                / self.out2021[i]
            )
        reg_capital += -1 * capital_target
        stabilize_C = Constraint(reg_capital**2, label="stabilize_C")
        return stabilize_C

    def compile(self, capital_growth_factor):
        minimize_HHI = self.calc_minimize_HHI()
        stabilize_C = self.calc_stabelize_c(capital_growth_factor)
        maximize_R = self.calc_maximize_ROC(capital_growth_factor)
        emission = self.calc_emission()

        # Variables to combine the 4 objectives to optimize.
        labda1 = Placeholder("labda1")
        labda2 = Placeholder("labda2")
        labda3 = Placeholder("labda3")
        labda4 = Placeholder("labda4")

        # Define Hamiltonian as a weighted sum of individual constraints
        H = (
            labda1 * minimize_HHI
            - labda2 * maximize_R
            + labda4 * stabilize_C
            + labda3 * emission
        )
        self.model = H.compile()

    def make_qubo(self, labda1: float, labda2: float, labda3: float, labda4: float):
        feed_dict = {
            "labda1": labda1,
            "labda2": labda2,
            "labda3": labda3,
            "labda4": labda4,
        }
        return self.model.to_qubo(feed_dict=feed_dict)
