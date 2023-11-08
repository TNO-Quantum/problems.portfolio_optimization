"""This module contains the ``QuboCompiler`` class."""
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
        """Init of the ``QuboCompiler`` class.

        The ``QuboCompiler`` can create a verity of QUBO formulation by combining
        different objectives and constraints.

        Args:
            portfolio_data: A ``pandas.Dataframe`` containing the portfolio to optimize.
            kmin: Minimum $k$ in the discretization of the variables.
            kmax: Maximum $k$ in the discretization of the variables.
        """
        self._qubo_factory = QuboFactory(portfolio_data, kmin, kmax)

        self._to_compile: list[Callable[[], tuple[NDArray[np.float_], float]]] = []
        self._compiled_qubos: list[NDArray[np.float_]] = []

    def add_minimize_HHI(self: QuboCompilerT) -> QuboCompilerT:
        """Add the minimize HHI objective to the compile list.

        Returns:
            Self.
        """
        self._to_compile.append(self._qubo_factory.calc_minimize_HHI)
        return self

    def add_maximize_ROC(
        self: QuboCompilerT,
        formulation: int,
        capital_growth_factor: float = 0,
        ancilla_qubits: int = 0,
    ) -> QuboCompilerT:
        """Add the maximize ROC objective an based on the input a stabilize c constraint.

        Args:
            formulation: Integer representing which formulation to pick. If formulation
                is 1 or 4, then one QUBO term will be added. If formulation is 2, then
                2 QUBO terms will be added and the argument `capital_growth_factor` must
                be provided. If formulation is 3, then 2 QUBO terms will be added as
                well, but the argument `ancilla_qubits` must be provided.
            capital_growth_factor: Capital growth factor of formulation 2.
            ancilla_qubits: Number of ancilla qubits to use for formulation 3.

        Returns:
            Self.
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
        """Add the emission constraint to the compile list.

        Returns:
            Self.
        """
        self._to_compile.append(self._qubo_factory.calc_emission_constraint)
        return self

    def add_growth_factor_constraint(
        self: QuboCompilerT, growth_target: float
    ) -> QuboCompilerT:
        """Add the capital growth factor constraint to the compile list.

        The constraint is formulated as total_out_future/total_out_now = growth_target.

        Args:
            growth_target: Growth target to use in teh constraint.

        Returns:
            Self.
        """
        method = partial(
            self._qubo_factory.calc_growth_factor_constraint, growth_target
        )
        self._to_compile.append(method)
        return self

    def compile(self: QuboCompilerT) -> QuboCompilerT:
        """Compile all QUBOs in the compile list.

        Returns:
            Self."""
        for constructor in self._to_compile:
            qubo, _ = constructor()
            self._compiled_qubos.append(qubo)
        return self

    def make_qubo(self, *lambdas: float) -> tuple[NDArray[np.float_], float]:
        """Make a QUBO of the entiry problem with the given lambdas.

        Args:
            lambdas: Scaling parameters for each QUBO in the formulation.

        Returns:
            Tuple containing the QUBO matrix and offset.
        """
        if len(lambdas) != len(self._compiled_qubos):
            raise ValueError(
                "Number of lambdas does not correspond with the number of Hamiltonians."
            )
        qubo = sum(
            (lambda_i * qubo_i for lambda_i, qubo_i in zip(lambdas, self._compiled_qubos)),
            start=np.zeros_like(self._compiled_qubos[0]),
        )

        return qubo, float("nan")
