from __future__ import annotations

from functools import partial
from typing import Callable, TypeVar

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from .qubo_factory import QuboFactory

QuboCompilerT = TypeVar("QuboCompilerT", bound="QuboCompiler")


class QuboCompiler:
    def __init__(self, portfolio_data: DataFrame, kmin: int, kmax: int) -> None:
        self._qubo_factory = QuboFactory(portfolio_data, kmin, kmax)

        self._to_compile: list[Callable[[], tuple[NDArray[np.float_], float]]] = []
        self._compiled_qubos: list[NDArray[np.float_]] = []

    def add_minimize_HHI(self: QuboCompilerT) -> QuboCompilerT:
        self._to_compile.append(self._qubo_factory.calc_minimize_HHI)
        return self

    def add_maximize_ROC(
        self: QuboCompilerT,
        formulation: int,
        capital_growth_factor: float = 0,
        ancilla_qubits: int = 0,
    ) -> QuboCompilerT:
        """
        formulation 1:
            add 1 qubo term
        formulation 2:
            add 2 qubo terms, requires extra arg capital_growth_factor
        formulation 3:
            add 2 qubo terms, requires extra arg ancilla_qubits
        formulation 4:
            add 1 qubo term
        """
        if formulation == 1:
            self._to_compile.append(self._qubo_factory.calc_maximize_ROC1)
        elif formulation == 2:
            roc_method = partial(
                self._qubo_factory.calc_maximize_ROC2, capital_growth_factor
            )
            stabalize_method = partial(
                self._qubo_factory.calc_stabilize_c1, capital_growth_factor
            )
            self._to_compile.append(roc_method)
            self._to_compile.append(stabalize_method)
        elif formulation == 3:
            self._qubo_factory.n_vars += ancilla_qubits
            self._to_compile.append(self._qubo_factory.calc_maximize_ROC3)
            self._to_compile.append(self._qubo_factory.calc_stabilize_c2)
        elif formulation == 4:
            self._to_compile.append(self._qubo_factory.calc_maximize_ROC4)

        return self

    def add_emission_constraint(self: QuboCompilerT) -> QuboCompilerT:
        self._to_compile.append(self._qubo_factory.calc_emission_constraint)
        return self

    def add_growth_factor_constraint(
        self: QuboCompilerT, growth_target: float
    ) -> QuboCompilerT:
        """Add constaint: total_out2030/total_out2021 = growth_target"""
        method = partial(
            self._qubo_factory.calc_growth_factor_constraint, growth_target
        )
        self._to_compile.append(method)
        return self

    def compile(self: QuboCompilerT) -> QuboCompilerT:
        for constructor in self._to_compile:
            qubo, _ = constructor()
            self._compiled_qubos.append(qubo)
        return self

    def make_qubo(self, *labdas: float) -> tuple[NDArray[np.float_], float]:
        if len(labdas) != len(self._compiled_qubos):
            raise ValueError(
                "Number of labdas does not correspond with the number of Hamiltonians."
            )
        qubo = sum(
            (labda_i * qubo_i for labda_i, qubo_i in zip(labdas, self._compiled_qubos)),
            start=np.zeros_like(self._compiled_qubos[0]),
        )

        return qubo, float("nan")
