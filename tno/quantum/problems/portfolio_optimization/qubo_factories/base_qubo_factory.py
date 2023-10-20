from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from pyqubo import Array, Constraint


class BaseQUBOFactory(ABC):
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

    @abstractmethod
    def calc_maximize_ROC(self):
        ...

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
