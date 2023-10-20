from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pyqubo import Array, Constraint


class QUBOFactory:
    def __init__(
        self,
        var: Array,
        N: int,
        out2021: NDArray[np.float_],
        LB: NDArray[np.float_],
        UB: NDArray[np.float_],
        e: NDArray[np.float_],
        income: NDArray[np.float_],
        capital: NDArray[np.float_],
        kmin: int,
        kmax: int,
        maxk: int,
    ) -> None:
        self.var = var
        self.N = N
        self.out2021 = out2021
        self.LB = LB
        self.UB = UB
        self.e = e
        self.income = income
        self.capital = capital
        self.kmin = kmin
        self.kmax = kmax
        self.maxk = maxk

    def _calc_sumi(self):
        sumi = 0
        for i in range(self.N):
            sumi += (
                self.LB[i]
                + (self.UB[i] - self.LB[i])
                * sum(
                    2 ** (k + self.kmin) * self.var[i * self.kmax + k]
                    for k in range(self.kmax)
                )
                / self.maxk
            )

        return sumi

    def calc_minimize_HHI(self):
        Correctiefactor = 1.00
        Exp_total_out2030 = Correctiefactor * np.sum((self.UB + self.LB)) / 2

        minimize_HHI = Constraint(
            sum(
                (
                    self.LB[i]
                    + (self.UB[i] - self.LB[i])
                    * sum(
                        2 ** (k + self.kmin) * self.var[i * self.kmax + k]
                        for k in range(self.kmax)
                    )
                    / self.maxk
                )
                ** 2
                for i in range(self.N)
            )
            / (Exp_total_out2030**2),
            label="minimize_HHI",
        )
        return minimize_HHI

    def calc_maximize_ROC(
        self,
    ):
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

    def calc_maximize_ROC2(
        self,
        capital_target: float,
    ):
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

    def calc_maximize_ROC3(self, ancilla_qubits: int):
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

    def calc_maximize_ROC4(self):
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

    def calc_emission(self):
        emis2021 = np.sum(self.e * self.out2021)
        bigE = emis2021 / np.sum(self.out2021)
        sumi = self._calc_sumi()

        emission_model = 0
        for i in range(self.N):
            emission_model += (
                0.76
                * self.e[i]
                * (
                    self.LB[i]
                    + (self.UB[i] - self.LB[i])
                    * sum(
                        2 ** (k + self.kmin) * self.var[i * self.kmax + k]
                        for k in range(self.kmax)
                    )
                    / self.maxk
                )
                / emis2021
            )
        emission_model += (-0.7 * bigE * sumi) / emis2021
        emission = Constraint(emission_model**2, label="minimize_emission")
        return emission
