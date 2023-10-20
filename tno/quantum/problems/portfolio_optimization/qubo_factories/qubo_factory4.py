import numpy as np
from numpy.typing import NDArray
from pyqubo import Constraint, Placeholder

from .base_qubo_factory import BaseQUBOFactory


class QUBOFactory4(BaseQUBOFactory):
    def calc_maximize_ROC(self):
        returns = self.income / self.out2021
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
                    * returns[i]
                )
                for i in range(self.N)
            )
            / sum(
                (
                    ((self.LB[i] + self.UB[i]) / (2.0 * self.out2021[i]))
                    * self.capital[i]
                )
                for i in range(self.N)
            ),
            label="maximize_ROC",
        )

        return maximize_ROC

    def compile(self):
        minimize_HHI = self.calc_minimize_HHI()
        maximize_ROC = self.calc_maximize_ROC()
        emission = self.calc_emission()

        # Variables to combine the 3 HHI2030tives to optimize.
        labda1 = Placeholder("labda1")
        labda2 = Placeholder("labda2")
        labda3 = Placeholder("labda3")

        # Define Hamiltonian as a weighted sum of individual constraints
        H = labda1 * minimize_HHI - labda2 * maximize_ROC + labda3 * emission
        self.model = H.compile()

    def make_qubo(self, labda1: float, labda2: float, labda3: float):
        feed_dict = {
            "labda1": labda1,
            "labda2": labda2,
            "labda3": labda3,
        }
        return self.model.to_qubo(feed_dict=feed_dict)
