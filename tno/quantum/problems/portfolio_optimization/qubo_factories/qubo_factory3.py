import numpy as np
from numpy.typing import NDArray
from pyqubo import Constraint, Placeholder

from .base_qubo_factory import BaseQUBOFactory


class QUBOFactory3(BaseQUBOFactory):
    def calc_maximize_ROC(self, ancilla_qubits: int):
        capital2021 = np.sum(self.capital)
        solution_qubits = self.N * self.kmax
        size_of_variable_array = solution_qubits + ancilla_qubits
        app_inv_cap_growth_fac = 1 + sum(
            self.var[k]
            * (2 ** (solution_qubits - k - 1))
            * (-1 + (2 ** (solution_qubits - k - 1)))
            for k in range(solution_qubits, size_of_variable_array)
        )

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
        maximize_R = Constraint(
            app_inv_cap_growth_fac * max_R / capital2021, label="maximize_R"
        )
        return maximize_R

    def calc_stabalize_c(self, ancilla_qubits: int):
        solution_qubits = self.N * self.kmax
        size_of_variable_array = solution_qubits + ancilla_qubits
        cap_growth_fac = 1 + sum(
            self.var[k] * (2 ** (solution_qubits - k - 1))
            for k in range(solution_qubits, size_of_variable_array)
        )
        capital2021 = sum(self.capital[i] for i in range(self.N))
        capital_target = cap_growth_fac * capital2021

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

    def compile(self, ancilla_qubits: int):
        minimize_HHI = self.calc_minimize_HHI()
        stabilize_C = self.calc_stabalize_c(ancilla_qubits)
        maximize_R = self.calc_maximize_ROC(ancilla_qubits)
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
