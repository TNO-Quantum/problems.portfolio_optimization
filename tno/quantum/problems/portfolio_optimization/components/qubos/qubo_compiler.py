"""This module contains the ``QuboCompiler`` class."""
from __future__ import annotations

from functools import partial
from typing import Callable, TypeVar

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from .qubo_factory import QuboFactory

from typing import Optional

QuboCompilerT = TypeVar("QuboCompilerT", bound="QuboCompiler")


class QuboCompiler:
    def __init__(self, portfolio_data: DataFrame, k: int) -> None:
        """Init of the ``QuboCompiler`` class.

        The ``QuboCompiler`` can create a verity of QUBO formulation by combining
        different objectives and constraints.

        Args:
            portfolio_data: A ``pandas.Dataframe`` containing the portfolio to optimize.
            k: The number of bits that are used to represent the outstanding amount for
                each asset. A fixed point representation is used to represent `$2^k$`
                different equidistant values in the range `$[LB_i, UB_i]$` for asset i.
        """
        self._qubo_factory = QuboFactory(portfolio_data, k)

        self._to_compile: list[Callable[[], tuple[NDArray[np.float_], float]]] = []
        self._compiled_qubos: list[NDArray[np.float_]] = []

    def add_minimize_HHI(self: QuboCompilerT) -> QuboCompilerT:
        r"""Add the minimize HHI objective to the compile list.

        The HHI objective is given by

        $$HHI = \frac{\sum_i y_i^2}{\left(\sum_i y_i\right)^2},$$

        where:

            - `$y_i$` is the future outstanding amount for asset `$i$`.

        For the QUBO formulation, see the docs of
        :py:meth:`~portfolio_optimization.components.qubos.qubo_factory.QuboFactory.calc_minimize_HHI`.

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

    def add_emission_constraint(
        self: QuboCompilerT,
        column_name_now: str,
        column_name_future: Optional[str] = None,
        reduction_percentage_target: float = 0.7,
    ) -> QuboCompilerT:
        r"""Add the emission constraint to the compile list.

        The constraint is given by

        $$\frac{\sum_if_i \cdot y_i}{\sum_i y_i} = g \frac{\sum_ie_i \cdot x_i}{\sum_i x_i},$$

        where:

            - `$x_i$` is the current outstanding amount for asset `$i$`,
            - `$y_i$` is the future outstanding amount for asset `$i$`,
            - `$e_i$` is the current emission intensity for asset `$i$`,
            - `$f_i$` is the expected emission intensity at the future for asset `$i$`,
            - `$g$` is the target value for the relative emission reduction.

        For the QUBO formulation, see the docs of
        :py:meth:`~portfolio_optimization.components.qubos.qubo_factory.QuboFactory.calc_emission_constraint`.

        Args:
            variable_now: Name of the column in the portfolio dataset corresponding to
                the variables at current time.
            variable_future: Name of the column in the portfolio dataset corresponding
                to the variables at future time. If no value is provided, it is assumed
                that the value is constant over time, i.e., the variable
                ``variable_now`` will be used.
            reduction_percentage_target: target value for reduction percentage amount.

        Returns:
            Self.
        """
        method = partial(
            self._qubo_factory.calc_emission_constraint,
            column_name_now=column_name_now,
            column_name_future=column_name_future,
            reduction_percentage_target=reduction_percentage_target,
        )
        self._to_compile.append(method)
        return self

    def add_growth_factor_constraint(
        self: QuboCompilerT, growth_target: float
    ) -> QuboCompilerT:
        r"""Add the capital growth factor constraint to the compile list.

        The constraint is formulated as total_outstanding_future/total_outstanding_now = growth_target.

        The constraint is given by

        $$\frac{\sum_i y_i}{\sum_i x_i} = g,$$

        where:

            - `$x_i$` is the current outstanding amount for asset `$i$`,
            - `$y_i$` is the future outstanding amount for asset `$i$`,
            - `$g$` is the target value for the total growth factor.

        For the QUBO formulation, see the docs of
        :py:meth:`~portfolio_optimization.components.qubos.qubo_factory.QuboFactory.calc_growth_factor_constraint`.

        Args:
            growth_target: target value for growth factor total outstanding amount.

        Returns:
            Self.
        """
        method = partial(
            self._qubo_factory.calc_growth_factor_constraint,
            growth_target=growth_target,
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
        """Make a QUBO of the entire problem with the given lambdas.

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
            (
                lambda_i * qubo_i
                for lambda_i, qubo_i in zip(lambdas, self._compiled_qubos)
            ),
            start=np.zeros_like(self._compiled_qubos[0]),
        )

        return qubo, float("nan")
